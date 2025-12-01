import torch
import math
import einops
from typing import Dict, List, Tuple, Optional, Any
from my_diffusers.callbacks import PipelineCallback
from src.distill_utils.attn_processor_cache import pop_cached_attn, pop_cached_qk, clear_attn_cache
from src.distill_utils.query_key_cost_metric import COST_METRIC_FN
from torch import nn


import os
import math
import numpy as np
import torch.nn.functional as FNN
import time
import uuid
from PIL import Image as _PILImage, ImageDraw as _PILDraw
import pandas as pd


class Softmax(nn.Module):
    def __init__(self, 
                 softmax_temp=1.0, num_view_for_per_view=None, per_view=False):
        super(Softmax, self).__init__()
        self.num_view = num_view_for_per_view
        self.per_view = per_view
        self.softmax_temp = softmax_temp
    
    def forward(self, x, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // self.num_view
            x = x.reshape(*x.shape[:-1], self.num_view, HW)
        x = x / self.softmax_temp
        return x.softmax(dim=-1)
    
class Softmax_HeadMean(Softmax):
    def forward(self, x, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // self.num_view
            x = x.reshape(*x.shape[:-1], self.num_view, HW)
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        return x.mean(dim=1, keepdim=True)





class AttentionVisualizationCallback(PipelineCallback):
    """
    VGGT와 UNet attention을 비교하여 시각화 로스를 계산하는 callback 클래스
    매 denoising step마다 호출되어 attention logit을 수집하고 distillation loss를 계산합니다.
    """
    
    # Request exactly the tensors we need from the pipeline callback: current x_t and noise prediction
    tensor_inputs = ["latents", "latent_model_input", "noise_pred"]
    
    def __init__(
        self,
        vggt_model: torch.nn.Module,
        visualize_config: Dict[str, Any],
        batch: Dict[str, Any],
        cond_num: int,
        device: torch.device,
        do_attn_visualize: bool = True,
        accelerator=None,
        vae=None,
    ):
        """
        Args:
            vggt_model: VGGT 모델
            visualize_config: 시각화 설정 (pairs, query, key, loss_fn 등)
            batch: 배치 데이터 (image, vggt_attn_cache 등)
            cond_num: condition frame 수
            device: 디바이스
            do_attn_visualize: attention 시각화 활성화 여부
        """
        super().__init__()
        self.vggt_model = vggt_model
        self.visualize_config = visualize_config
        
        # Loss function 설정
        self._setup_loss_function()
        self.batch = batch
        self.cond_num = cond_num
        self.device = device
        self.do_attn_visualize = do_attn_visualize
        self.accelerator = accelerator  # accelerator 객체 저장
        # Tweedie background support
        self.vae = vae
        self._use_tweedie_bg = str(self.visualize_config.get('viz_pred_bg_source', 'final')).lower() == 'tweedie'
        self._tweedie_views_by_step = {}
        self.visualize_loss_dict = {}
        self.step_losses = []
        self.step_layer_losses = {}  # step별 layer별 loss 저장
        self.layer_losses = {}  # layer별 누적 loss 저장
        self.match_precision_records = {}  # layer별 매칭 정밀도 기록
        # detailed per-(sample,step,layer,head,loss_fn) rows for export
        self._detailed_rows = []
        
        self._pending_viz = []   
        self.viz_use_gpu = bool(self.visualize_config.get('viz_use_gpu', True))
        self.viz_dtype   = torch.float16 if self.visualize_config.get('viz_dtype','fp16')=='fp16' else torch.float32

        
        # VGGT attention cache 미리 계산
        if self.do_attn_visualize and self.vggt_model is not None:
            self._prepare_vggt_cache()
    
    def _setup_loss_function(self):
        """Loss function을 문자열에서 실제 함수로 변환"""
        # define canonical loss functions and store mapping for per-pair overrides
        # Helper: normalize inputs and perform head/view collapsing
        def _prepare_for_loss(pred: torch.Tensor, gt: torch.Tensor):
            """
            Accepts pred/gt in either per-view form (B,Head,Q,V,HW) or global form (B,Head,Q,K)
            and returns a tuple (mode, P, G, meta) where
              - mode is 'per_view' or 'global'
              - P, G are tensors already converted to float and on same device
              - meta contains dict with B, Head, Q, V (if per_view) and K
            """
            if pred is None or gt is None:
                raise ValueError("pred and gt must be tensors")
            # ensure same dtype/device
            gt = gt.to(dtype=pred.dtype, device=pred.device)

            if pred.dim() == 5:
                # (B,Head,Q,V,HW)
                B, Head, Q, V, HW = pred.shape
                mode = 'per_view'
                meta = dict(B=B, Head=Head, Q=Q, V=V, HW=HW)
                return mode, pred, gt, meta
            elif pred.dim() == 4:
                # (B,Head,Q,K)
                B, Head, Q, K = pred.shape
                mode = 'global'
                meta = dict(B=B, Head=Head, Q=Q, K=K)
                return mode, pred, gt, meta
            else:
                raise ValueError(f"Unsupported pred shape for loss: {pred.shape}")

        def _collapse_heads_as_batch(x: torch.Tensor):
            # (B,Head,...) -> (B*Head,...)
            B, H = x.shape[0], x.shape[1]
            return x.view(B * H, *x.shape[2:])

        # Core loss implementations operating on collapsed shapes
        def _cross_entropy(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            # pred_coll, gt_coll: (N, Q, K) probabilities over last dim
            eps = 1e-8
            qloss = - (gt_coll * (pred_coll + eps).log()).sum(dim=-1)  # (N, Q)
            per_sample = qloss.mean(dim=1)  # (N,)
            return per_sample.mean()

        def _kl_divergence(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            qloss = (gt_coll * (gt_coll.log() - pred_coll.log())).sum(dim=-1)
            per_sample = qloss.mean(dim=1)
            return per_sample.mean()

        def _l1(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            # mean abs diff over Q and K
            l1 = torch.abs(pred_coll - gt_coll)
            per_sample = l1.mean(dim=(1, 2))
            return per_sample.mean()

        # Using decorator approach to avoid duplicated wrapper logic. Decorator converts a
        # per-sample (N,Q,K)->(N,) function into a full loss callable accepting pred/gt
        # in either per-view (B,Head,Q,V,HW) or global (B,Head,Q,K) formats and applying
        # optional pre/post head-mean semantics.
        def loss_variant(variant: Optional[str] = None):
            def decorator(per_sample_fn):
                def loss_callable(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
                    mode, P, G, meta = _prepare_for_loss(pred, gt)

                    # pre-headmean
                    if variant == 'pre':
                        P = P.mean(dim=1, keepdim=True)
                        G = G.mean(dim=1, keepdim=True)

                    if mode == 'per_view':
                        B, H, Q, V, HW = P.shape
                        P_coll = P.permute(0, 1, 3, 2, 4).contiguous().view(B * H * V, Q, HW)
                        G_coll = G.permute(0, 1, 3, 2, 4).contiguous().view(B * H * V, Q, HW)
                        per_sample = per_sample_fn(P_coll, G_coll)  # (N,) where N == B*H*V

                        # Return per-head values (1D tensor) when no head-mean requested
                        if variant is None:
                            per_sample = per_sample.view(B, H, V)
                            per_head = per_sample.mean(dim=2)  # (B, H)
                            return per_head.view(-1)

                        # For pre/post head-mean variants return a single-value tensor (length 1)
                        scalar = float(per_sample.mean().item())
                        return torch.tensor([scalar], device=P.device, dtype=P.dtype)

                    else:  # global
                        B, H, Q, K = P.shape
                        P_coll = _collapse_heads_as_batch(P)
                        G_coll = _collapse_heads_as_batch(G)
                        per_sample = per_sample_fn(P_coll, G_coll)  # (N,) where N == B*H

                        if variant is None:
                            per_sample = per_sample.view(B, H)
                            return per_sample.view(-1)

                        scalar = float(per_sample.mean().item())
                        return torch.tensor([scalar], device=P.device, dtype=P.dtype)

                return loss_callable

            return decorator

        # Per-sample loss implementations (return per-sample vector of shape (N,))
        def _cross_entropy_per_sample(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            eps = 1e-8
            # pred_coll, gt_coll: (N, Q, K)
            qloss = - (gt_coll * (pred_coll + eps).log()).sum(dim=-1)  # (N, Q)
            per_sample = qloss.mean(dim=1)  # (N,)
            return per_sample

        def _kl_per_sample(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            qloss = (gt_coll * (gt_coll.log() - pred_coll.log())).sum(dim=-1)
            per_sample = qloss.mean(dim=1)
            return per_sample

        def _l1_per_sample(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor:
            l1 = torch.abs(pred_coll - gt_coll)
            per_sample = l1.mean(dim=(1, 2))
            return per_sample

        # Base per-sample loss functions mapping.
        #
        # Helper for adding custom loss functions:
        # - Implement a per-sample function with signature:
        #       def fn(pred_coll: torch.Tensor, gt_coll: torch.Tensor) -> torch.Tensor
        #   where the inputs are "collapsed" tensors as described below and the
        #   output is a 1D tensor of per-sample losses (length N):
        #     * Global mode: pred_coll / gt_coll shape = (N, Q, K)
        #         - N == B * Head (batch times head)
        #     * Per-view mode: the callback will collapse to
        #         pred_coll / gt_coll shape = (N, Q, K') where
        #         - N == B * Head * V (batch * head * num_views)
        #         - K' == HW (flattened per-view spatial keys)
        #   The per-sample fn must return a 1D tensor of length N (per-sample losses).
        #
        # To register an external per-sample fn, pass a dict into
        # `visualize_config['loss_fn_map']` mapping base name -> callable.
        base_per_sample = {
            'cross_entropy': _cross_entropy_per_sample,
            'kl_divergence': _kl_per_sample,
            'l1': _l1_per_sample,
        }

        # Allow validate.py to pass an external mapping via visualize_config['loss_fn_map']
        external_map = self.visualize_config.get('loss_fn_map', None)

        # decorator factory: canonical implementation is defined earlier (keep the first occurrence)

        # Store base per-sample functions (may be overridden by external_map).
        # expose base per-sample map (possibly overridden by external_map)
        base_map = {}
        for base_name, per_sample_fn in base_per_sample.items():
            if external_map is not None and base_name in external_map:
                per_sample_fn = external_map[base_name]
            base_map[base_name] = per_sample_fn

        # also expose ready-to-call wrapped variants for convenience (so callers can request 'cross_entropy_pre' etc.)
        wrapped_map = {}
        for base_name, per_sample_fn in base_map.items():
            wrapped_map[base_name] = loss_variant(None)(per_sample_fn)
            wrapped_map[f"{base_name}_pre"] = loss_variant('pre')(per_sample_fn)
            wrapped_map[f"{base_name}_post"] = loss_variant('post')(per_sample_fn)

        # store both
        self._ATTN_LOSS_BASE = base_map
        self.ATTN_LOSS_FN = wrapped_map

        # # set default loss_fn
        # if callable(loss_fn):
        #     self.loss_fn = loss_fn
        # elif isinstance(loss_fn, str):
        #     key = loss_fn.lower()
        #     if key not in ATTN_LOSS_FN:
        #         raise ValueError(f"Unsupported loss_fn string: {loss_fn}")
        #     self.loss_fn = ATTN_LOSS_FN[key]
        #     print(f"Using loss_fn: {key}")
        # else:
        #     raise ValueError(f"Unsupported loss_fn type: {type(loss_fn)}")
        
    @staticmethod
    def _skew(v):
        vx, vy, vz = v
        return np.array([
            [0, -vz, vy],
            [vz, 0, -vx],
            [-vy, vx, 0]
        ], dtype=np.float32)

    @staticmethod
    def _fundamental_from_w2c(Ks, Kt, w2c_s, w2c_t):
        """
        Ks, Kt: (3,3) or (4,4)
        w2c_s, w2c_t: (4,4) world->cam
        returns F: (3,3)
        """
        Ks = np.array(Ks, dtype=np.float32)
        Kt = np.array(Kt, dtype=np.float32)
        w2c_s = np.array(w2c_s, dtype=np.float32)
        w2c_t = np.array(w2c_t, dtype=np.float32)

        Rs = w2c_s[:3, :3]
        ts = w2c_s[:3, 3:4]
        Rt = w2c_t[:3, :3]
        tt = w2c_t[:3, 3:4]

        # relative pose source -> target
        R_ts = Rt @ Rs.T
        t_ts = tt - R_ts @ ts

        E = AttentionVisualizationCallback._skew(t_ts.reshape(3)) @ R_ts

        Ks_inv = np.linalg.inv(Ks[:3, :3])
        Kt_inv_T = np.linalg.inv(Kt[:3, :3]).T
        F = Kt_inv_T @ E @ Ks_inv
        return F

    @staticmethod
    def _epiline_in_image(F, pt, W, H):
        """
        F: (3,3), pt: (u,v) in source image (pixel coords)
        returns two image-border points ((x1,y1),(x2,y2)) or None if not intersecting
        """
        p = np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float32)
        l = F @ p
        a, b, c = float(l[0]), float(l[1]), float(l[2])

        pts = []
        # x = 0 -> y = -c/b
        if abs(b) > 1e-6:
            y0 = -c / b
            if 0 <= y0 <= H - 1:
                pts.append((0.0, float(y0)))
        # x = W-1 -> y = -(c + a*(W-1))/b
        if abs(b) > 1e-6:
            yW = -(c + a * (W - 1)) / b
            if 0 <= yW <= H - 1:
                pts.append((float(W - 1), float(yW)))
        # y = 0 -> x = -c/a
        if abs(a) > 1e-6:
            x0 = -c / a
            if 0 <= x0 <= W - 1:
                pts.append((float(x0), 0.0))
        # y = H-1 -> x = -(c + b*(H-1))/a
        if abs(a) > 1e-6:
            xH = -(c + b * (H - 1)) / a
            if 0 <= xH <= W - 1:
                pts.append((float(xH), float(H - 1)))

        if len(pts) < 2:
            return None
        return pts[0], pts[1]

    @staticmethod
    def _draw_epiline_pil(img_pil, pt_src, F, color=(0, 255, 0), width=2):
        W, H = img_pil.size
        seg = AttentionVisualizationCallback._epiline_in_image(F, pt_src, W, H)
        if seg is None:
            return img_pil
        (x1, y1), (x2, y2) = seg
        d = _PILDraw.Draw(img_pil)
        d.line((x1, y1, x2, y2), fill=color, width=width)
        return img_pil

    def _pack_viz_record(self, *, step_index, layer_key,
                         pred_mean, gt_mean,  # (1, num_qv, per_q, num_kv, per_k) on cpu
                         F, per_q, per_k, q_side, k_side,
                         query_idx_list, key_idx_list, q_in_views, grid_cols, seq):
        # 나중에 렌더에 필요한 최소 메타/텐서만 저장
        q_abs_list = []
        for vp in query_idx_list:
            for q_in in q_in_views:
                q_abs_list.append(int(vp)*per_q + int(q_in))
        # Store compact CPU half tensors to avoid repeated GPU<->CPU transfers
        try:
            pred_cpu = pred_mean.detach().to(device='cpu', dtype=torch.float16)
            gt_cpu = gt_mean.detach().to(device='cpu', dtype=torch.float16)
        except Exception:
            # fallback to original detach if conversion fails
            pred_cpu = pred_mean.detach()
            gt_cpu = gt_mean.detach()

        return dict(
            step_index=int(step_index), layer_key=layer_key, F=int(F),
            per_q=int(per_q), per_k=int(per_k), q_side=int(q_side), k_side=int(k_side),
            query_idx=list(query_idx_list), key_idx=list(key_idx_list),
            q_in_views=list(q_in_views), q_abs_list=q_abs_list,
            grid_cols=int(grid_cols), seq=seq,
            pred_mean=pred_cpu, gt_mean=gt_cpu
        )

    def _render_viz_record(self, rec, *, pred_views=None, cond_num=None, replace_targets_in_unet=True):
        # --- config / device / dtype ---
        use_gpu = bool(self.visualize_config.get('viz_use_gpu', True))
        dev = self.device if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
        viz_dtype = torch.float16 if str(self.visualize_config.get('viz_dtype', 'fp16')).lower() == 'fp16' else torch.float32
        alpha = float(self.visualize_config.get('viz_alpha', 0.6))

        # ✅ 항상 포인트를 찍도록 강제
        draw_points = True

        # --- unpack geometry / meta ---
        per_q   = int(rec['per_q'])
        per_k   = int(rec['per_k'])
        q_side  = int(rec['q_side'])
        k_side  = int(rec['k_side'])
        q_idx   = list(rec['query_idx'])
        k_idx   = list(rec['key_idx'])
        q_in    = list(rec['q_in_views'])
        grid_cols = max(1, int(rec.get('grid_cols', len(k_idx))))
        seq     = str(rec.get('seq', 'sample'))
        step_index = int(rec.get('step_index', 0))
        layer_key  = str(rec.get('layer_key', 'layer'))
        cond_num = int(self.cond_num if cond_num is None else cond_num)

        # --- viz_softmax_mode: Pred만 적용, GT는 항상 per_view ---
        def _parse_layer_key(s):
            u, v = None, None
            try:
                parts = s.split('_')
                for p in parts:
                    if p.startswith('unet'):
                        u = int(p[4:])
                    elif p.startswith('vggt'):
                        v = int(p[4:])
            except Exception:
                pass
            return u, v

        pred_norm_mode = None
        u_id, v_id = _parse_layer_key(layer_key)
        for p in self.visualize_config.get('pairs', []):
            if isinstance(p, dict) and ('unet_layer' in p and 'vggt_layer' in p):
                if str(p['unet_layer']) == str(u_id) and str(p['vggt_layer']) == str(v_id):
                    pred_norm_mode = p.get('viz_softmax_mode', pred_norm_mode)
                    break
        if pred_norm_mode is None:
            pred_norm_mode = self.visualize_config.get('viz_softmax_mode', 'global')
        pred_norm_mode = str(pred_norm_mode).lower()  # 'per_view' or 'global'
        gt_norm_mode = 'per_view'  # ✅ GT는 항상 per-view

        # --- move packed tensors to render device ---
        with torch.inference_mode():
            pm_t = rec['pred_mean'].to(device=dev, dtype=viz_dtype)  # (1, nqv, per_q, nkv, per_k)
            gm_t = rec['gt_mean'  ].to(device=dev, dtype=viz_dtype)
        B, nqv, pq, nkv, pk = pm_t.shape
        assert B == 1 and pq == per_q and pk == per_k

        # flatten to (Qtot, Ktot)
        pm = pm_t.reshape(1, nqv * pq, nkv * pk)[0]
        gm = gm_t.reshape(1, nqv * pq, nkv * pk)[0]

        # --- prepare GT views (0..1) ---
        imgs = self.batch['image'][0]  # [F,3,H,W]
        if imgs.min() < 0:
            imgs = (imgs + 1.0) / 2.0
        gt_views_t = imgs.to(device=dev, dtype=viz_dtype).clamp_(0, 1)  # (F,3,H,W)
        F_total = gt_views_t.shape[0]
        assert max(q_idx + k_idx) < F_total

        # --- optional: pred_views to torch ---
        def _to_torch_views(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                t = torch.as_tensor(x, device=dev)
                if t.ndim == 4 and t.shape[-1] == 3:
                    t = t.permute(0, 3, 1, 2)
                if t.dtype not in (torch.float16, torch.float32):
                    t = t.float()
                if t.max() > 1.0:
                    t = t / 255.0
                return t.to(viz_dtype).clamp_(0, 1)
            if isinstance(x, (list, tuple)):
                ts = []
                for arr in x:
                    t = torch.as_tensor(arr, device=dev)
                    if t.ndim == 3 and t.shape[-1] == 3: t = t.permute(2, 0, 1)
                    if t.ndim == 2: t = t.unsqueeze(0).repeat(3, 1, 1)
                    t = t.float() if t.max() <= 1.0 else (t.float()/255.0)
                    ts.append(t.to(viz_dtype).clamp_(0, 1).unsqueeze(0))
                return torch.cat(ts, dim=0)
            if torch.is_tensor(x):
                t = x.to(device=dev)
                if t.ndim == 4 and t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
                t = t.float() if t.max() <= 1.0 else (t.float()/255.0)
                return t.to(dtype=viz_dtype).clamp_(0, 1)
            return None

        pred_views_t = _to_torch_views(pred_views)

        # --- LUT 준비 ---
        if not hasattr(self, '_viz_lut'):
            try:
                lut_len = int(self.visualize_config.get('viz_lut_len', 256))
            except Exception:
                lut_len = 256
            try:
                import matplotlib as mpl
                cmap = mpl.cm.get_cmap('turbo', lut_len)
                lut_rgb_np = (cmap(np.linspace(0.0, 1.0, lut_len))[:, :3].astype('float32'))
                lut_rgb = torch.as_tensor(lut_rgb_np, device=self.device, dtype=self.viz_dtype)
            except Exception:
                lut_x = torch.linspace(0.0, 1.0, steps=lut_len, device=self.device, dtype=self.viz_dtype)
                lut_rgb = torch.stack([lut_x, lut_x, lut_x], dim=1).to(device=self.device, dtype=self.viz_dtype)
            self._viz_lut = lut_rgb

        def _attn_gray_to_rgb_torch(gray_nhw):
            L = self._viz_lut.shape[0]
            idx = (gray_nhw.clamp(0.0, 1.0) * (L - 1)).to(dtype=torch.long)
            rgb = self._viz_lut[idx.view(-1)].view(*idx.shape, 3)
            return rgb.permute(0, 3, 1, 2).to(dtype=viz_dtype, device=gray_nhw.device)

        def _make_key_backgrounds(use_pred_for_targets):
            srcs = []
            for vidx in k_idx:
                if use_pred_for_targets and pred_views_t is not None and vidx >= cond_num:
                    src = pred_views_t[vidx] if vidx < pred_views_t.shape[0] else gt_views_t[vidx]
                else:
                    src = gt_views_t[vidx]
                srcs.append(src.unsqueeze(0))
            return torch.cat(srcs, dim=0)  # (K,3,H,W)

        def _resize_like(x, H, W, mode='bilinear'):
            if x.shape[-2] == H and x.shape[-1] == W:
                return x
            return torch.nn.functional.interpolate(x, size=(H, W), mode=mode, align_corners=False if mode == 'bilinear' else None)

        # ✅ 공통: CHW torch → PIL로 파란/빨간 점(흰테두리) 찍고 다시 torch로
        def _draw_dot_chw(img_chw: torch.Tensor, cx: int, cy: int, r: int, fill_rgb: tuple, outline_rgb: tuple):
            # img_chw: (3,H,W) float[0..1]
            arr = (img_chw.clamp(0,1).mul(255).byte()).permute(1,2,0).cpu().numpy()  # HWC uint8
            pil = _PILImage.fromarray(arr)
            d = _PILDraw.Draw(pil)
            # 흰 테두리(바깥): r+2, 안쪽: r
            d.ellipse((cx-(r+2), cy-(r+2), cx+(r+2), cy+(r+2)), fill=outline_rgb)
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=fill_rgb)
            out = torch.from_numpy(np.array(pil)).permute(2,0,1).to(device=img_chw.device, dtype=viz_dtype) / 255.0
            return out

        # --- 정규화 범위를 인자로 받아 per_view/global 선택 + 포인트 그리기 ---
        def _rows_from_prob(prob_2d, q_panel_use_pred, bg_keys_t, norm_mode: str, draw_query_points: bool, draw_max_points: bool):
            rows = []
            K, _, Hc, Wc = bg_keys_t.shape
            per = k_side * k_side
            ncols = max(1, grid_cols)

            for vp in q_idx:
                for q_local in q_in:
                    q_abs = int(vp) * per_q + int(q_local)
                    seg = prob_2d[q_abs]                                 # (K*per,)
                    tiles = seg.view(K, per).view(K, k_side, k_side)     # (K,ks,ks)

                    # ▶ 정규화: 타일 min-max는 유지, 범위만 다르게
                    if norm_mode == 'per_view':
                        tmin = tiles.amin(dim=(1, 2), keepdim=True)
                        tmax = tiles.amax(dim=(1, 2), keepdim=True)
                        tiles_norm = torch.where(
                            (tmax > tmin),
                            (tiles - tmin) / (tmax - tmin + 1e-12),
                            torch.zeros_like(tiles)
                        )
                    else:
                        gmin = tiles.amin()
                        gmax = tiles.amax()
                        tiles_norm = (tiles - gmin) / (gmax - gmin + 1e-12) if (gmax > gmin) else torch.zeros_like(tiles)

                    # ▶ 키 타일별 argmax 좌표(타일 좌표계) 미리 계산
                    argmax_xy = []
                    if draw_max_points:
                        flat = tiles.view(K, -1)
                        idxs = flat.argmax(dim=1).tolist()
                        for i, idx in enumerate(idxs):
                            my, mx = divmod(int(idx), k_side)  # (y,x)
                            argmax_xy.append((mx, my))

                    # colorize + upsample
                    color = _attn_gray_to_rgb_torch(tiles_norm)                             # (K,3,ks,ks)
                    color_up = torch.nn.functional.interpolate(color, size=(Hc, Wc), mode='nearest')  # (K,3,H,W)

                    # blend
                    blended = (1.0 - alpha) * bg_keys_t + alpha * color_up                  # (K,3,H,W)

                    # ▶ 각 키 이미지에 "빨간 점(흰테두리)" 찍기
                    if draw_max_points:
                        r_k = max(3, min(Hc, Wc) // 80)
                        for ki in range(K):
                            mx, my = argmax_xy[ki]
                            cx = int(round((mx + 0.5) * (Wc / float(k_side))))
                            cy = int(round((my + 0.5) * (Hc / float(k_side))))
                            blended[ki] = _draw_dot_chw(blended[ki], cx, cy, r_k,
                                                        fill_rgb=(255, 0, 0), outline_rgb=(255, 255, 255))  # 🔴+흰

                    # ===== Epipolar line 그리기 (필요한 경우) =====
                    # 조건: 배치에 intrinsic/extrinsic 정보가 있어야 함 (B, F, ...)
                    intr_batch = self.batch.get('intrinsic', None)
                    extr_batch = self.batch.get('extrinsic', None)
                    if intr_batch is None or extr_batch is None:
                        raise RuntimeError("Epipolar drawing requested but 'intrinsic'/'extrinsic' missing in batch")

                    # intr_batch/extr_batch may be tensors or numpy arrays
                    intr_arr = intr_batch[0].detach().cpu().numpy() if torch.is_tensor(intr_batch) else np.array(intr_batch[0])
                    extr_arr = extr_batch[0].detach().cpu().numpy() if torch.is_tensor(extr_batch) else np.array(extr_batch[0])

                    # source pixel center in the background tile coordinate system
                    qy = q_local // q_side
                    qx = q_local % q_side
                    u_src = (qx + 0.5) * (Wc / float(q_side))
                    v_src = (qy + 0.5) * (Hc / float(q_side))

                    # source camera matrices
                    Ks = intr_arr[int(vp)]
                    if Ks.shape[0] == 4:
                        Ks = Ks[:3, :3]
                    w2c_s = extr_arr[int(vp)]

                    # 각 key view 별로 epipolar 라인 그리기
                    for ki in range(K):
                        tgt_view_idx = int(k_idx[ki]) if ki < len(k_idx) else int(k_idx[0])
                        Kt = intr_arr[tgt_view_idx]
                        if Kt.shape[0] == 4:
                            Kt = Kt[:3, :3]
                        w2c_t = extr_arr[tgt_view_idx]

                        Fmat = self._fundamental_from_w2c(Ks, Kt, w2c_s, w2c_t)

                        # PIL로 변환해서 라인 그림
                        tgt_img_chw = blended[ki]
                        tgt_img = (tgt_img_chw.clamp(0,1).mul(255).byte()).permute(1,2,0).cpu().numpy()
                        tgt_pil = _PILImage.fromarray(tgt_img)

                        line_width = max(1, min(Hc, Wc) // 200)
                        tgt_pil = self._draw_epiline_pil(tgt_pil, (u_src, v_src), Fmat, color=(0,255,0), width=line_width)

                        blended[ki] = torch.from_numpy(np.array(tgt_pil)).permute(2,0,1).to(device=dev, dtype=viz_dtype) / 255.0

                    # make K→grid
                    grid_rows = []
                    for s in range(0, K, ncols):
                        row = blended[s:s + ncols]
                        if row.shape[0] < ncols:
                            pad = torch.zeros((ncols - row.shape[0], 3, Hc, Wc), device=dev, dtype=blended.dtype)
                            row = torch.cat([row, pad], dim=0)
                        row = row.permute(1, 2, 0, 3).reshape(3, Hc, ncols * Wc)
                        grid_rows.append(row)
                    grid = torch.cat(grid_rows, dim=1)  # (3, H*rows, W*ncols)

                    # query panel (left)
                    if q_panel_use_pred and pred_views_t is not None and vp >= cond_num and vp < pred_views_t.shape[0]:
                        qsrc = pred_views_t[vp]
                    else:
                        qsrc = gt_views_t[vp]
                    qsrc = _resize_like(qsrc.unsqueeze(0), Hc, Wc).squeeze(0)

                    # ▶ 쿼리 포인트: 파란 점(흰테두리)
                    if draw_query_points:
                        qy, qx = divmod(int(q_local), q_side)
                        cx = int(round((qx + 0.5) * (Wc / float(q_side))))
                        cy = int(round((qy + 0.5) * (Hc / float(q_side))))
                        r_q = max(3, min(Hc, Wc) // 90)
                        qsrc = _draw_dot_chw(qsrc, cx, cy, r_q,
                                                fill_rgb=(0, 120, 255), outline_rgb=(255, 255, 255))  # 🔵+흰

                    # concat query panel and grid horizontally
                    panel = torch.cat([qsrc, grid], dim=2)
                    rows.append(panel)
            return rows

        # backgrounds
        bg_unet = _make_key_backgrounds(use_pred_for_targets=replace_targets_in_unet)
        bg_gt   = _make_key_backgrounds(use_pred_for_targets=False)

        # ★ Pred는 config 모드, GT는 항상 per_view
        pred_rows = _rows_from_prob(pm, True,  bg_unet, norm_mode=pred_norm_mode, draw_query_points=draw_points, draw_max_points=draw_points)
        gt_rows   = _rows_from_prob(gm, False, bg_gt,   norm_mode=gt_norm_mode,  draw_query_points=draw_points, draw_max_points=draw_points)

        # stack rows vertically
        pred_mat = torch.cat(pred_rows, dim=1) if len(pred_rows) else torch.zeros((3, 1, 1), device=dev, dtype=viz_dtype)
        gt_mat   = torch.cat(gt_rows,   dim=1) if len(gt_rows)   else torch.zeros((3, 1, 1), device=dev, dtype=viz_dtype)

        # width pad to equal
        Wmax = max(pred_mat.shape[2], gt_mat.shape[2])
        def _pad_w(x, W):
            if x.shape[2] >= W: return x
            pad = torch.zeros((x.shape[0], x.shape[1], W - x.shape[2]), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=2)
        pred_mat = _pad_w(pred_mat, Wmax)
        gt_mat   = _pad_w(gt_mat, Wmax)

        # separator
        sep_h = max(8, int(min(pred_mat.shape[1], pred_mat.shape[2]) * 0.05))
        sep = torch.zeros((3, sep_h, Wmax), device=dev, dtype=viz_dtype)

        # final canvas (top: UNet/pred, bottom: VGGT/gt)
        canvas = torch.cat([pred_mat, sep, gt_mat], dim=1)

        # to PIL
        with torch.inference_mode():
            canvas = canvas.clamp(0, 1)
            canvas_uint8 = (canvas.mul(255).byte()).permute(1, 2, 0).cpu().numpy()
        final_canvas = _PILImage.fromarray(canvas_uint8)

        # save (optional)
        save_dir = self.visualize_config.get('viz_save_dir', None)
        q_abs_list = rec.get('q_abs_list', None)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if q_abs_list is None:
                q_abs_list = []
                for vp in q_idx:
                    for q_local in q_in:
                        q_abs_list.append(int(vp) * per_q + int(q_local))
            q_suffix = "-".join([str(int(x)) for x in q_abs_list])
            out_path = os.path.join(save_dir, f"attn_step{step_index}_{layer_key}_q{q_suffix}.png")
            final_canvas.save(out_path)

        # cache for wandb logging
        if not hasattr(self, 'attention_images_list'):
            self.attention_images_list = []
        if q_abs_list is None:
            q_abs_list = []
            for vp in q_idx:
                for q_local in q_in:
                    q_abs_list.append(int(vp) * per_q + int(q_local))
        # background source info for logging
        bg_source = 'final'
        if self._use_tweedie_bg and int(step_index) in self._tweedie_views_by_step:
            bg_source = 'tweedie'
        key = f"attn/{seq}/step{step_index}/{layer_key}/q{'-'.join([str(int(x)) for x in q_abs_list])}"
        caption = f"{seq} | {layer_key} | step {step_index} | bg:{bg_source} | qs:{'-'.join([str(int(x)) for x in q_abs_list])}"
        self.attention_images_list.append({
            'key': key,
            'image': final_canvas,
            'caption': caption,
        })




        
    def finalize_visualizations(self, pred_images=None, cond_num=None):
        if not self._pending_viz:
            return
        # pred_images → [F,H,W,C] float(0~1) → uint8 HWC 리스트로 변환
        pred_views_final = None
        if pred_images is not None:
            arr = np.array(pred_images)
            pred_views_final = [ (arr[f]*255).clip(0,255).astype('uint8') for f in range(arr.shape[0]) ]

        if cond_num is None:
            cond_num = self.cond_num

        for rec in self._pending_viz:
            step_idx = int(rec.get('step_index', -1))
            if self._use_tweedie_bg and step_idx in self._tweedie_views_by_step:
                print(f"[INFO] finalize_visualizations: using Tweedie bg for step {step_idx}")
                self._render_viz_record(
                    rec, pred_views=self._tweedie_views_by_step[step_idx],
                    cond_num=cond_num, replace_targets_in_unet=True
                )
            else:
                reason = self._tweedie_store_reasons.get(step_idx, 'no_reason_recorded')
                print(f"[INFO] finalize_visualizations: using final pred bg for step {step_idx} (tweedie_reason={reason})")
                self._render_viz_record(
                    rec, pred_views=pred_views_final,
                    cond_num=cond_num, replace_targets_in_unet=True
                )
        self._pending_viz.clear()
    
    def _prepare_vggt_cache(self):
        """VGGT attention cache를 미리 계산"""
        with torch.no_grad():
            image = self.batch['image'].to(self.device)  # [B,F,3,H,W]
            # Ensure VGGT has been configured to cache the numeric vggt layers requested in visualize_config['pairs']
            try:
                desired_layers = []
                for p in self.visualize_config.get('pairs', []):
                    if isinstance(p, dict):
                        vt = p.get('vggt_layer', None)
                    else:
                        try:
                            _, vt = p
                        except Exception:
                            vt = None
                    try:
                        if vt is not None:
                            vtid = int(vt)
                            if vtid not in desired_layers:
                                desired_layers.append(vtid)
                    except Exception:
                        # non-numeric vggt layer (e.g., 'track_head') -> skip
                        pass
                if desired_layers and hasattr(self.vggt_model, 'set_attn_cache'):
                    # (re)configure VGGT to cache these layers (idempotent)
                    try:
                        self.vggt_model.set_attn_cache(attn_layer_ids=desired_layers)
                    except Exception:
                        pass
                # run forward and capture attn_cache
                vggt_pred = self.vggt_model(image)
                self.batch['vggt_attn_cache'] = vggt_pred.get('attn_cache', {})
                print(f"[DEBUG] VGGT reported cache_attn_layer_ids: {getattr(self.vggt_model, 'cache_attn_layer_ids', None)}")
                print(f"[DEBUG] vggt_pred['attn_cache'] keys: {list(self.batch['vggt_attn_cache'].keys()) if self.batch.get('vggt_attn_cache') else None}")
            except Exception as e:
                print(f"[WARN] failed to prepare vggt cache: {e}")
                self.batch['vggt_attn_cache'] = {}

    def _logits_to_probs(self, logits: torch.Tensor, num_key_views: int, mode: Optional[str]) -> torch.Tensor:
        """Convert logits to probabilities according to `mode`.

        This is a class-level helper (accessible via self) — pair-level callers must
        provide an explicit `mode` (no global fallbacks).
        """
        if logits is None:
            return logits
        if mode is None:
            raise ValueError("_logits_to_probs requires explicit 'mode' (pair-level configuration only)")

        # preserve dtype/device
        device = logits.device
        dtype = logits.dtype

        # quick check: if values along K already sum to ~1, assume probs and return
        try:
            sums = logits.sum(dim=-1)
            if torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
                return logits
        except Exception:
            pass

        if mode == 'global':
            return logits.softmax(dim=-1)

        # expect K divisible by num_key_views
        K = int(logits.shape[-1])
        num_k = int(num_key_views)
        if num_k <= 0 or K % num_k != 0:
            raise ValueError(f"softmax per-view requires num_key_views that divides K: num_key_views={num_k}, K={K}")
        per = K // num_k

        if mode == 'per_view':
            # reshape last dim to (..., num_k, per) and softmax over per
            K = int(logits.shape[-1])
            if num_k <= 0 or K % num_k != 0:
                # Provide a clearer error message to help debugging malformed shapes
                raise ValueError(f"per_view softmax requires num_key_views that divides K: num_key_views={num_k}, K={K}")
            per = K // num_k
            leading = logits.shape[:-1]
            # defensive reshape: ensure the computed shape matches element count
            try:
                view_shaped = logits.view(*leading, num_k, per)
            except Exception as e:
                raise RuntimeError(f"Failed to reshape logits for per_view softmax: attempted shape {*leading, num_k, per} from logits.shape={logits.shape}: {e}")
            probs = view_shaped.softmax(dim=-1)
            return probs.view(*leading, K)

        raise ValueError(f"Unsupported softmax mode: {mode}")

    def _alpha_sigma_from_scheduler(self, scheduler, timestep):
        # DDIM: alphas_cumprod[t] = \bar{\alpha}_t
        t_idx = int(timestep.item() if torch.is_tensor(timestep) else timestep)
        if isinstance(scheduler.alphas_cumprod, torch.Tensor):
            alpha_bar_t = float(scheduler.alphas_cumprod[t_idx].item())
        else:
            alpha_bar_t = float(scheduler.alphas_cumprod[t_idx])
        alpha_t = math.sqrt(alpha_bar_t)
        sigma_t = math.sqrt(max(0.0, 1.0 - alpha_bar_t))
        return alpha_t, sigma_t

    @torch.no_grad()
    def _maybe_store_tweedie_bg(self, pipeline, step_index, timestep, callback_kwargs):
        if not self._use_tweedie_bg:
            return
        if self.vae is None:
            raise RuntimeError(f"Tweedie background requested but callback has no VAE (step {step_index})")

        x_t = callback_kwargs.get('latent_model_input', None)
        eps_or_v = callback_kwargs.get('noise_pred', None)
        if x_t is None or eps_or_v is None:
            missing = []
            if x_t is None:
                missing.append('latent_model_input')
            if eps_or_v is None:
                missing.append('noise_pred')
            raise RuntimeError(f"Tweedie background requested but missing tensors for step {step_index}: {','.join(missing)}")

        # Normalize batch/frame shapes between latent_model_input and noise_pred
        # Determine base batch (B) and number of frames (F) from provided batch images if available
        try:
            B = int(self.batch['image'].shape[0])
            F = int(self.batch['image'].shape[1])
        except Exception:
            B = None
            F = None

        epsN = eps_or_v.shape[0]
        is_eps_per_frame = (B is not None and F is not None and epsN == B * F)

        if is_eps_per_frame:
            # ensure x_t becomes per-frame with shape (B*F, ...)
            if x_t.shape[0] == 2 * B * F:
                x_t = x_t[B * F :]
            elif x_t.shape[0] == 2 * B:
                # CFG doubled per-sample: take second half (B,) then expand to per-frame
                x_t = x_t[B:]
                # expand per-sample to per-frame
                x_t = x_t.unsqueeze(1).repeat(1, F, *([1] * (x_t.dim() - 1)))
                x_t = x_t.view(B * F, *x_t.shape[2:])
            elif x_t.shape[0] == B:
                # expand per-sample to per-frame
                x_t = x_t.unsqueeze(1).repeat(1, F, *([1] * (x_t.dim() - 1)))
                x_t = x_t.view(B * F, *x_t.shape[2:])
            elif x_t.shape[0] == B * F:
                pass
            else:
                # diagnostic
                try:
                    print(f"[ERROR] Tweedie batch/frame mismatch at step {step_index}: x_t.shape={tuple(x_t.shape)}, eps.shape={tuple(eps_or_v.shape)}, inferred B,F={B},{F}")
                except Exception:
                    pass
                raise RuntimeError(f"Tweedie background requested but could not align latent_model_input to per-frame shape at step {step_index}")

        else:
            # treat eps as per-sample (epsN == B or unknown). Align x_t per-sample
            base = epsN
            if x_t.shape[0] == 2 * base:
                x_t = x_t[base:]
            elif x_t.shape[0] != base:
                # diagnostic
                lat = callback_kwargs.get('latents', None)
                try:
                    print(f"[ERROR] Tweedie batch mismatch at step {step_index}: x_t.shape={tuple(x_t.shape) if hasattr(x_t,'shape') else type(x_t)}, noise_pred_firstdim={epsN}")
                    print(f"[ERROR] latents shape: {tuple(lat.shape) if hasattr(lat,'shape') else None}")
                except Exception:
                    pass
                raise RuntimeError(f"Tweedie background requested but batch size mismatch at step {step_index}: x_t={x_t.shape[0]}, noise_pred={epsN}")

        sched = pipeline.scheduler
        try:
            alpha_t, sigma_t = self._alpha_sigma_from_scheduler(sched, timestep)
        except Exception as e:
            raise RuntimeError(f"Failed to compute alpha/sigma for step {step_index}: {e}")

        pred_type = str(getattr(sched.config, 'prediction_type', 'epsilon') or 'epsilon').lower()

        if pred_type in ('epsilon', 'eps'):
            x0_latents = (x_t - sigma_t * eps_or_v) / max(alpha_t, 1e-12)
        elif pred_type in ('v', 'v_prediction', 'v-prediction'):
            x0_latents = alpha_t * x_t - sigma_t * eps_or_v
        elif pred_type in ('sample', 'x0'):
            x0_latents = eps_or_v
        else:
            raise RuntimeError(f"Unsupported scheduler prediction_type for Tweedie restore: {pred_type}")

        if x0_latents.dim() == 5:
            x0_latents = x0_latents[0]

        scaling = float(getattr(self.vae.config, 'scaling_factor', 0.18215))
        x0_latents = x0_latents.to(device=self.device, dtype=torch.float32)
        imgs = self.vae.decode(x0_latents / scaling).sample
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0
        imgs = (imgs * 255.0).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        self._tweedie_views_by_step[int(step_index)] = [imgs[i] for i in range(imgs.shape[0])]
        try:
            print(f"[DEBUG] stored tweedie bg for step {step_index}: {len(self._tweedie_views_by_step[int(step_index)])} views")
        except Exception:
            pass
    


    def _resolve_heads(self, pair: Dict[str, Any], F: int):
        """
        Return a (unet_head, vggt_head) tuple by instantiating from per-pair
        overrides or falling back to the global `visualize_config` settings.
        """
        # Always use the unified head class `softmax` for both UNet and VGGT.
        # Allow per-pair or global kwargs to configure temperatures and per_view.
        logit_head_cls = Softmax
        viz_head_cls = Softmax_HeadMean
        
        if logit_head_cls is None:
            raise RuntimeError("Unified logit head class 'softmax' not found")
        if viz_head_cls is None:
            raise RuntimeError("Unified viz head class 'softmax_headmean' not found")
        
        ## view 수
        
        loss_keys = self.visualize_config.get('loss_key', None) ## target, reference, all
        if loss_keys == "target":
            loss_num_view = 1
        elif loss_keys == "reference":
            loss_num_view = F - 1
        elif loss_keys == "all":
            loss_num_view = F
        else:
            raise ValueError(f"visualize_config['loss_key'] {loss_keys} not implemented")
        
        vis_keys = self.visualize_config.get('viz_key', None) ## target, reference, all
        if vis_keys == "target":
            viz_num_view = 1
        elif vis_keys == "reference":
            viz_num_view = F - 1
        elif vis_keys == "all":
            viz_num_view = F
        else:
            raise ValueError(f"visualize_config['viz_key'] {vis_keys} not implemented")
        

        # kwargs resolution: prefer per-pair kwargs, then global visualize_config
        unet_kwargs = pair.get('unet_logit_head_kwargs', None) if isinstance(pair, dict) else None
        vggt_kwargs = pair.get('vggt_logit_head_kwargs', None) if isinstance(pair, dict) else None

        mode_loss = str(pair.get('loss_softmax_mode', 'global')).lower()
        mode_viz = str(pair.get('viz_softmax_mode', 'global')).lower()
        # dict + dict 연산 오류 수정: dict.update() 또는 {**dict1, **dict2} 사용
        unet_loss_kwargs = {**(unet_kwargs or {}), 'per_view': mode_loss == 'per_view', 'num_view_for_per_view': loss_num_view}
        vggt_loss_kwargs = {**(vggt_kwargs or {}), 'per_view': mode_loss == 'per_view', 'num_view_for_per_view': loss_num_view}
        unet_viz_kwargs = {**(unet_kwargs or {}), 'per_view': mode_viz == 'per_view', 'num_view_for_per_view': viz_num_view}
        # vggt_viz_kwargs = {**(vggt_kwargs or {}), 'per_view': mode_viz == 'per_view', 'num_view_for_per_view': viz_num_view}
        vggt_viz_kwargs = {**(vggt_kwargs or {}), 'per_view': True, 'num_view_for_per_view': viz_num_view}

        
        unet_loss_head = logit_head_cls(**unet_loss_kwargs)
        vggt_loss_head = logit_head_cls(**vggt_loss_kwargs)
        unet_viz_head = viz_head_cls(**unet_viz_kwargs)
        vggt_viz_head = viz_head_cls(**vggt_viz_kwargs)
        return unet_loss_head, vggt_loss_head, unet_viz_head, vggt_viz_head


    def _resize_token(self, tok: torch.Tensor, target_size: int, F: int) -> torch.Tensor:
        """토큰을 target_size로 리사이즈"""
        # Support inputs with and without explicit Head dimension.
        # Acceptable input shapes:
        # - (B, Head, FHW, C)
        # - (B, FHW, C)  -> treated as Head=1
        if tok.dim() == 4:
            B, Head, FHW, C = tok.shape
            has_head = True
        elif tok.dim() == 3:
            B, FHW, C = tok.shape
            Head = 1
            has_head = False
            # add head dim at dim=1 for consistent processing
            tok = tok.unsqueeze(1)  # (B,1,FHW,C)
        else:
            raise ValueError(f"Unexpected token tensor shape for _resize_token: {tok.shape}")

        HW = FHW // F
        H = W = int(math.sqrt(HW))
        tok = einops.rearrange(tok, 'B Head (F H W) C -> (B Head F) C H W', 
                              B=B, Head=Head, F=F, H=H, W=W, C=C)
        tok = torch.nn.functional.interpolate(tok, size=(target_size, target_size), mode='bilinear')
        tok = einops.rearrange(tok, '(B Head F) C H W -> B Head (F H W) C', 
                              B=B, Head=Head, F=F, H=target_size, W=target_size, C=C)

        # If original input had no head dim, keep Head dim=1 (caller should handle)
        return tok
    
    def _slice_attention_map(self, attnmap: torch.Tensor, query_idx: List[int], 
                           key_idx: List[int], F: int) -> torch.Tensor:
        """attention map을 query/key index에 따라 슬라이싱"""
        # import pdb; pdb.set_trace()
        B, Head, Q, K = attnmap.shape
        HW = Q // F
        attnmap = einops.rearrange(attnmap, 'B Head (F1 HW1) (F2 HW2) -> B Head F1 HW1 F2 HW2', 
                                  B=B, Head=Head, F1=F, HW1=HW, F2=F, HW2=HW)
        attnmap = attnmap[:, :, query_idx][:, :, :, :, key_idx]
        attnmap = einops.rearrange(attnmap, 'B Head f1 HW1 f2 HW2 -> B Head (f1 HW1) (f2 HW2)', 
                                  B=B, Head=Head, f1=len(query_idx), f2=len(key_idx), HW1=HW, HW2=HW)
        return attnmap
    
    def _extract_view_indices(self, F: int, mode: str) -> Tuple[List[int], List[int]]:
        """Loss 계산용 query와 key의 view index 추출"""
        # Loss Query index 추출
        if self.visualize_config[f'{mode}_query'] == "target":
            query_idx = list(range(self.cond_num, F))
        elif self.visualize_config[f'{mode}_query'] == "all":
            query_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['loss_query'] {self.visualize_config['loss_query']} not implemented")
        
        # Loss Key index 추출
        if self.visualize_config[f'{mode}_key'] == "reference":
            key_idx = list(range(0, self.cond_num))
        elif self.visualize_config[f'{mode}_key'] == "all":
            key_idx = list(range(0, F))
        else:
            raise NotImplementedError(f"visualize_config['loss_key'] {self.visualize_config['loss_key']} not implemented")
        
        return query_idx, key_idx

    
    ##### VISUALIZATION UTILS #####
    def _attn_gray_to_rgb(self, gray: np.ndarray) -> np.ndarray:
        """Map normalized [0,1] gray heatmap to RGB uint8 (simple HSV-like colormap).

        Accepts either a numpy array or a torch tensor. Preserves dtype/device for
        torch inputs by converting to CPU numpy for processing and returning a
        uint8 numpy array suitable for PIL.
        """
        # Accept torch tensors as input (convert to numpy)
        is_torch = False
        try:
            if isinstance(gray, torch.Tensor):
                is_torch = True
                gray_arr = gray.detach().cpu().numpy()
            else:
                gray_arr = np.array(gray)
        except Exception:
            gray_arr = np.array(gray)

        # Ensure float in [0,1]
        gray_arr = gray_arr.astype(np.float32)
        # clamp defensively
        gray_arr = np.clip(gray_arr, 0.0, 1.0)

        h = (1.0 - gray_arr) * (2.0 / 3.0)
        s = np.ones_like(gray_arr)
        v = np.ones_like(gray_arr)
        hp = (h * 6.0)
        i = (np.floor(hp).astype(np.int32)) % 6
        f = hp - np.floor(hp)
        p = np.zeros_like(gray_arr)
        q = 1.0 - f
        t = f
        r = np.empty_like(gray_arr)
        g = np.empty_like(gray_arr)
        b = np.empty_like(gray_arr)
        mask = (i == 0)
        r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
        mask = (i == 1)
        r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
        mask = (i == 2)
        r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
        mask = (i == 3)
        r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
        mask = (i == 4)
        r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
        mask = (i == 5)
        r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgb

    def _draw_query_point(self, img: _PILImage.Image, qx: int, qy: int, q_side: int) -> _PILImage.Image:
        d = _PILDraw.Draw(img)
        w, h = img.size
        sx = w / float(q_side)
        sy = h / float(q_side)
        cx = int((qx + 0.5) * sx)
        cy = int((qy + 0.5) * sy)
        r_in = max(2, min(w, h) // 120)
        # outline + fill
        d.ellipse((cx - r_in*2, cy - r_in*2, cx + r_in*2, cy + r_in*2), fill=(255, 255, 255))
        d.ellipse((cx - r_in, cy - r_in, cx + r_in, cy + r_in), fill=(0, 128, 255))
        return img

    def _draw_max_attention_point(self, img: _PILImage.Image, attention_tile: np.ndarray, tile_side: int) -> _PILImage.Image:
        """각 view별로 가장 확률이 높은 부분에 파란 점을 그림"""
        d = _PILDraw.Draw(img)
        w, h = img.size
        # compute argmax index
        flat = attention_tile.flatten()
        max_idx = int(np.argmax(flat))
        max_y, max_x = np.unravel_index(max_idx, attention_tile.shape)

        # compute softargmax (expectation) on the tile for consistency with debug
        # NOTE: softargmax display disabled per user request; we still compute hard argmax
        # for fallback but do not draw the soft-argmax marker.
        xs = np.arange(tile_side)
        total = attention_tile.sum()
        if total > 0:
            pred_sx = (attention_tile * xs.reshape(1, tile_side)).sum(axis=(0, 1)) / (total + 1e-12)
            pred_sy = (attention_tile * xs.reshape(tile_side, 1)).sum(axis=(0, 1)) / (total + 1e-12)
        else:
            pred_sx = float(max_x)
            pred_sy = float(max_y)

        # tile 좌표를 이미지 좌표로 변환
        sx = w / float(tile_side)
        sy = h / float(tile_side)
        cx_hard = int((max_x + 0.5) * sx)
        cy_hard = int((max_y + 0.5) * sy)
        cx_soft = int((pred_sx + 0.5) * sx)
        cy_soft = int((pred_sy + 0.5) * sy)

        # dot sizes
        r_in = max(3, min(w, h) // 80)

        # draw argmax (red) with white border
        r_small = max(2, r_in // 2)
        d.ellipse((cx_hard - (r_small+2), cy_hard - (r_small+2), cx_hard + (r_small+2), cy_hard + (r_small+2)), fill=(255,255,255))
        d.ellipse((cx_hard - r_small, cy_hard - r_small, cx_hard + r_small, cy_hard + r_small), fill=(255,0,0))

        return img

    def _coords_argmax(self, tile: np.ndarray) -> Tuple[int, int]:
        """Return (x,y) integer coords of argmax within tile."""
        idx = int(np.argmax(tile.flatten()))
        y, x = np.unravel_index(idx, tile.shape)
        return int(x), int(y)

    def _coords_softargmax(self, tile: np.ndarray) -> Tuple[float, float]:
        """Return (x,y) soft-argmax (expectation) within tile as floats."""
        side = tile.shape[0]
        xs = np.arange(side)
        total = float(tile.sum())
        if total <= 0:
            # fallback to argmax if tile is all zeros
            ax, ay = self._coords_argmax(tile)
            return float(ax), float(ay)
        # pred_sx sums over columns for x coordinate, pred_sy sums over rows for y
        pred_sx = (tile * xs.reshape(1, side)).sum(axis=(0, 1)) / (total + 1e-12)
        pred_sy = (tile * xs.reshape(side, 1)).sum(axis=(0, 1)) / (total + 1e-12)
        return float(pred_sx), float(pred_sy)

    def _maybe_save_attn_overlay(
            self,
            *,
            step_index: int,
            layer_key: str,
            pred_logits: torch.Tensor,  # [B, Head, Q, K]
            gt_logits: torch.Tensor,    # [B, Head, Q, K]
            F: int,
            query_idx_list: List[int],
            key_idx_list: List[int],
    ) -> None:
        """Save attention overlay image. Pred (위), GT (아래), 내부는 키-뷰 오버레이 그리드(기본 3x3)."""

        # ---------- helpers ----------
        def preprocess_image(img: torch.Tensor) -> np.ndarray:
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = torch.clamp(img, 0, 1).detach().cpu()
            return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        def _select_query_indices(qxy_cfg, qindex_cfg, num_queries_cfg, per_q, q_side=32):
            """우선순위: 좌표 > 인덱스 > 개수 > 중앙토큰. 좌표는 canonical 32-grid 기준으로 스케일."""
            q_in_views = []
            base_q_side = 32
            try:
                scale = float(q_side) / float(base_q_side)
            except Exception:
                scale = 1.0

            def _scale_and_clamp_coord(raw_x, raw_y):
                try:
                    rx = float(raw_x); ry = float(raw_y)
                except Exception:
                    rx = int(raw_x); ry = int(raw_y)
                sx = int(round(rx * scale)); sy = int(round(ry * scale))
                sx = max(0, min(q_side - 1, sx))
                sy = max(0, min(q_side - 1, sy))
                return sx, sy

            if qxy_cfg is not None:
                # (x, y) or [(x,y), ...]
                if (isinstance(qxy_cfg, (list, tuple))
                    and len(qxy_cfg) > 0
                    and not isinstance(qxy_cfg[0], (int, float))):
                    for pair in qxy_cfg:
                        qx_raw, qy_raw = pair[0], pair[1]
                        qx, qy = _scale_and_clamp_coord(qx_raw, qy_raw)
                        q_in_views.append(qy * q_side + qx)
                else:
                    qx_raw, qy_raw = qxy_cfg[0], qxy_cfg[1]
                    qx, qy = _scale_and_clamp_coord(qx_raw, qy_raw)
                    q_in_views.append(qy * q_side + qx)
            elif qindex_cfg is not None:
                if isinstance(qindex_cfg, (list, tuple)):
                    for qi in qindex_cfg:
                        q_in_views.append(int(qi) % per_q)
                else:
                    q_in_views.append(int(qindex_cfg) % per_q)
            elif num_queries_cfg is not None:
                try:
                    num_queries = int(num_queries_cfg)
                except Exception:
                    num_queries = 1
                num_queries = max(1, num_queries)
                max_indices = per_q
                if num_queries >= max_indices:
                    q_in_views = list(range(max_indices))
                else:
                    rand_indices = torch.randperm(max_indices, device=torch.device('cpu'))[:num_queries].tolist()
                    q_in_views = [int(idx) for idx in rand_indices]
            else:
                q_in_views = [per_q // 2]
            return q_in_views

        # def tiles_from_vec(vec: torch.Tensor, num_views: int, side: int) -> List[np.ndarray]:
        #     """1D 벡터 -> per-view 정규화된 side×side 타일 리스트."""
        #     tiles: List[np.ndarray] = []
        #     vec_np = vec.detach().cpu().numpy()
        #     per = side * side
        #     for v in range(num_views):
        #         seg = vec_np[v * per:(v + 1) * per]
        #         m = seg.min(); M = seg.max()
        #         if M > m:
        #             seg = (seg - m) / (M - m)
        #         else:
        #             seg = np.zeros_like(seg)
        #         tiles.append(seg.reshape(side, side))
        #     return tiles

        # def build_grid(tiles: List[np.ndarray], backgrounds: List[np.ndarray], ncols: int = 3) -> _PILImage.Image:
        #     """타일들을 배경 위에 블렌딩해 ncols 격자로 배치. 부족분은 패딩."""
        #     assert len(tiles) == len(backgrounds), f"#tiles({len(tiles)}) != #bgs({len(backgrounds)})"
        #     blended_imgs = []
        #     for tile, bg in zip(tiles, backgrounds):
        #         Hc, Wc = bg.shape[:2]
        #         color = self._attn_gray_to_rgb(tile)
        #         color_img = _PILImage.fromarray(color).resize((Wc, Hc), _PILImage.NEAREST)
        #         alpha = float(self.visualize_config.get('viz_alpha', 0.6))
        #         blended = (bg.astype(np.float32) * (1 - alpha) + np.array(color_img).astype(np.float32) * alpha)
        #         blended = blended.clip(0, 255).astype(np.uint8)
        #         blended_pil = _PILImage.fromarray(blended)
        #         blended_pil = self._draw_max_attention_point(blended_pil, tile, k_side)
        #         blended_imgs.append(np.array(blended_pil))

        #     n = len(blended_imgs)
        #     ncols = max(1, int(self.visualize_config.get('viz_grid_cols', ncols)))  # 기본 3열, 설정 가능
        #     nrows = int(math.ceil(n / ncols))
        #     H0, W0 = blended_imgs[0].shape[:2]
        #     pad = np.zeros((H0, W0, 3), dtype=np.uint8)
        #     rows = []
        #     for r in range(nrows):
        #         row_imgs = blended_imgs[r*ncols:(r+1)*ncols]
        #         while len(row_imgs) < ncols:
        #             row_imgs.append(pad.copy())
        #         rows.append(np.concatenate(row_imgs, axis=1))
        #     grid = np.concatenate(rows, axis=0)
        #     return _PILImage.fromarray(grid)

        # ---------- I/O ----------
        save_dir = self.visualize_config.get('viz_save_dir', None)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # ---------- tensors to CPU float ----------
        _target_dtype = torch.float16 if self.visualize_config.get('viz_dtype','fp16') == 'fp16' else torch.float32
        pred_orig = pred_logits.detach().to(dtype=_target_dtype)   # device 유지
        gt_orig   = gt_logits.detach().to(dtype=_target_dtype)



        
        num_k_views = max(1, len(key_idx_list))
        num_q_views = max(1, len(query_idx_list))

        # ---------- reshape & sanity checks ----------
        if pred_orig.dim() == 5:  # [B, H, num_q_views*per_q, num_k_views, per_k]
            Bp, Hp, Qp, num_k_from_shape_p, per_k_p = pred_orig.shape
            if num_k_from_shape_p != num_k_views:
                raise ValueError(f"UNet num_k_from_shape ({num_k_from_shape_p}) != num_k_views ({num_k_views})")
            per_q_p = Qp // num_q_views
            if Qp % num_q_views != 0:
                raise ValueError(f"UNet per_q ({per_q_p}) != num_q_views ({num_q_views}) * per_q ({per_q_p})")
            pred = pred_orig.view(Bp, Hp, num_q_views, per_q_p, num_k_views, per_k_p)

        elif pred_orig.dim() == 4:  # [B, H, num_q_views*per_q, num_k_views*per_k]
            Bp, Hp, Qp, Kp = pred_orig.shape
            
            per_q_p = Qp // num_q_views
            per_k_p = Kp // num_k_views
            
            if Kp % num_k_views != 0:
                raise ValueError(f"UNet K ({Kp}) not divisible by num_k_views ({num_k_views})")
            if Qp % num_q_views != 0:
                raise ValueError(f"UNet per_q ({per_q_p}) != num_q_views ({num_q_views}) * per_k ({per_k_p})")
            pred = pred_orig.view(Bp, Hp, num_q_views, per_q_p, num_k_views, per_k_p)
        else:
            raise ValueError(f"UNet unsupported pred_logits shape: {pred_orig.shape}")

        if gt_orig.dim() == 5:  # [B, H, num_q_views*per_q, num_k_views, per_k]
            Bg, Hg, Qg, num_k_from_shape_g, per_k_g = gt_orig.shape
            if num_k_from_shape_g != num_k_views:
                raise ValueError(f"VGGT num_k_from_shape ({num_k_from_shape_g}) != num_k_views ({num_k_views})")
            per_q_g = Qg // num_q_views
            if Qg % num_q_views != 0:
                raise ValueError(f"VGGT per_q ({per_q_g}) != num_q_views ({num_q_views}) * per_k ({per_k_g})")
            gt = gt_orig.view(Bg, Hg, num_q_views, per_q_g, num_k_views, per_k_g)

        elif gt_orig.dim() == 4:   # [B, H, num_q_views*per_q, num_k_views*per_k]
            Bg, Hg, Qg, Kg = gt_orig.shape
            if Kg % num_k_views != 0:
                raise ValueError(f"VGGT K ({Kg}) not divisible by num_k_views ({num_k_views})")
            per_k_g = Kg // num_k_views
            per_q_g = Qg // num_q_views
            if Qg % num_q_views != 0:
                raise ValueError(f"VGGT per_q ({per_q_g}) != num_q_views ({num_q_views}) * per_k ({per_k_g})")
            gt = gt_orig.view(Bg, Hg, num_q_views, per_q_g, num_k_views, per_k_g)
        else:
            raise ValueError(f"VGGT unsupported gt_logits shape: {gt_orig.shape}")

        # per_q/per_k 확정
        if per_q_g != per_q_p:
            raise ValueError(f"per_q_g ({per_q_g}) != per_q_p ({per_q_p})")
        if per_k_g != per_k_p:
            raise ValueError(f"per_k_g ({per_k_g}) != per_k_p ({per_k_p})")
        per_q = int(per_q_g)
        per_k = int(per_k_g)

        # 토큰 geometry
        q_side = int(math.sqrt(per_q))
        if q_side * q_side != per_q:
            raise ValueError(f"Q per-view must be square, got {per_q}")
        k_side = int(math.sqrt(per_k))
        if k_side * k_side != per_k:
            raise ValueError(f"K per-view must be square, got {per_k}")

        # ---------- config ----------
        qxy_cfg        = self.visualize_config.get('viz_query_xy', None)
        qindex_cfg     = self.visualize_config.get('viz_query_index', None)
        num_queries_cfg= self.visualize_config.get('viz_num_queries', None)

        # ---------- average over heads ----------
        pred_mean = pred.mean(dim=1)  # (B, num_q_views, per_q, num_k_views, per_k)
        gt_mean   = gt.mean(dim=1)

        # ---------- backgrounds ----------
        # bg_keys, bg_queries = [], []
        # with torch.no_grad():
        #     imgs = self.batch['image'][0]  # [F,3,H,W]
        #     for vidx in key_idx_list:
        #         bg_keys.append(preprocess_image(imgs[vidx]))
        #     for vidx in query_idx_list:
        #         bg_queries.append(preprocess_image(imgs[vidx]))

        # if len(bg_keys) == 0:
        #     # 키-뷰가 없으면 시각화 불가
        #     return

        # ---------- query selection ----------
        q_in_views = _select_query_indices(qxy_cfg, qindex_cfg, num_queries_cfg, per_q, q_side)

        # prepare flattened probability buffers for visualization
        Bp_, num_qv_p, per_q_chk_p, num_kv_p, per_k_p2 = pred_mean.shape
        pred_flat = pred_mean.reshape(Bp_, num_qv_p * per_q_chk_p, num_kv_p * per_k_p2)
        pred_prob = pred_flat[0]

        Bg_, num_qv_g2, per_q_chk_g, num_kv_g2, per_k_g2 = gt_mean.shape
        gt_flat = gt_mean.reshape(Bg_, num_qv_g2 * per_q_chk_g, num_kv_g2 * per_k_g2)
        gt_prob = gt_flat[0]
        
        grid_cols = int(self.visualize_config.get('viz_grid_cols', num_k_views))

        # --- 시퀀스 이름 ---
        seq = self.visualize_config.get('viz_seq_name', None)

        # pred_prob, gt_prob 방금 만든 직후!
        rec = self._pack_viz_record(
            step_index=step_index, layer_key=layer_key,
            pred_mean=pred_mean, gt_mean=gt_mean,
            F=F, per_q=per_q, per_k=per_k, q_side=q_side, k_side=k_side,
            query_idx_list=query_idx_list, key_idx_list=key_idx_list,
            q_in_views=q_in_views, grid_cols=grid_cols, seq=seq
        )

        defer = bool(self.visualize_config.get('viz_defer_render', True))
        if defer:
            # ✅ 보류: 배경 준비(이미지→numpy) 절대 하지 않음
            self._pending_viz.append(rec)
            return
        else:
            # ⬇️ 즉시 렌더일 때만 배경 준비가 필요하면 여기서 하거나,
            #     더 깔끔하게는 _render_viz_record가 알아서 배경을 만들게 두면 됩니다.
            # (권장) 배경은 _render_viz_record에서 만들도록 두고, 여기선 바로 호출만:
            self._render_viz_record(
                rec,
                pred_views=None,             # 최종 pred 없음
                cond_num=self.cond_num,
                replace_targets_in_unet=False
            )
            return

     


    def _roll_gt_map(self, gt_tensor: torch.Tensor) -> torch.Tensor:
        """Circularly roll the GT map along the last dimension if configured.

        Minimal, dtype/device-preserving implementation. If `visualize_config['roll_gt_map']`
        is missing or zero, returns the input unchanged.
        """
        shift = self.visualize_config.get('roll_gt_map', 0)
        if not shift:
            return gt_tensor

        try:
            shift_int = int(shift)
        except Exception:
            print(f"Warning: invalid roll_gt_map value: {shift}; skipping roll")
            return gt_tensor

        try:
            print(f"Rolling GT map by {shift_int} positions")
            return torch.roll(gt_tensor, shifts=shift_int, dims=-1)
        except Exception as e:
            print(f"Warning: roll_gt_map failed: {e}")
            return gt_tensor

    def _is_step_enabled(self, step_index):
        # Support separate viz_steps and loss_steps.
        # Semantics: `None` (not provided) => all steps; empty list [] => no steps;
        # otherwise only steps in the list are enabled.
        viz_enabled = False
        loss_enabled = False
        viz_steps = self.visualize_config.get('viz_steps', None)
        loss_steps = self.visualize_config.get('loss_steps', None)
        if viz_steps is None:
            viz_enabled = True
        elif isinstance(viz_steps, (list, tuple)) and len(viz_steps) == 0:
            viz_enabled = False
        else:
            viz_enabled = step_index in viz_steps
        if loss_steps is None:
            loss_enabled = True
        elif isinstance(loss_steps, (list, tuple)) and len(loss_steps) == 0:
            loss_enabled = False
        else:
            loss_enabled = step_index in loss_steps
        return viz_enabled, loss_enabled
    
    def _process_layer_pair(self):
        # 각 layer pair에 대해 loss 계산
        visualize_pairs = list(self.visualize_config['pairs'])
        
        # sort pairs so same unet_layer are consecutive -> avoids popping UNet cache early
        def _get_unet_layer(p):
            if isinstance(p, dict):
                return p.get('unet_layer')
            try:
                return p[0]
            except Exception:
                return None
            
        visualize_pairs = [p for p in visualize_pairs if _get_unet_layer(p) is not None]
        visualize_pairs.sort(key=_get_unet_layer)
        print(f"[DEBUG] Processing {len(visualize_pairs)} layer pairs (sorted): {visualize_pairs}")
        return visualize_pairs
    
    def _get_pred_attn_logit(
        self,
        unet_layer: int,
        current_unet: Optional[int],
        unet_attn_cache: Dict[str, Any],
        unet_qk_cache: Optional[Dict[str, Any]],
    ):
        if current_unet is not None and current_unet != unet_layer:
            unet_attn_cache.pop(str(current_unet), None)
            if unet_qk_cache is not None:
                unet_qk_cache.pop(str(current_unet), None)
            torch.cuda.empty_cache()

        current_unet = unet_layer
        key = str(unet_layer)
        if key not in unet_attn_cache:
            raise RuntimeError(f"UNet attention cache에 레이어 {unet_layer}가 존재하지 않습니다.")

        pred_attn_logit = unet_attn_cache[key]
        if pred_attn_logit.dim() == 4 and pred_attn_logit.shape[1] != 1:
            pred_attn_logit = pred_attn_logit[-1].unsqueeze(0)

        qk_entry = None
        if unet_qk_cache is not None and key in unet_qk_cache:
            qk_entry = unet_qk_cache[key]
            if not isinstance(qk_entry, (tuple, list)) or len(qk_entry) != 2:
                raise RuntimeError(f"레이어 {unet_layer}의 Q/K 캐시가 (query, key) 튜플 형태가 아닙니다.")

        return pred_attn_logit, qk_entry, current_unet
                    
    def _get_gt_tokens(self, vggt_layer: str):
         # Obtain GT tokens for the chosen vggt_layer.
                # - For 'point_map' the batch must include a point_map tensor shaped
                #   (B, V, 3, H, W) which we convert to (B, 1, VHW, C).
                # - For other layers we require `vggt_attn_cache` to contain the
                #   requested layer key and both 'query' and 'key' tensors.
        if vggt_layer == "point_map":
            if 'point_map' not in self.batch:
                raise RuntimeError("[DEBUG] point_map requested for visualization but missing from batch")
            pointmap = self.batch['point_map']
            Bp, Vp, Cp, Hp, Wp = pointmap.shape
            return pointmap.shape, pointmap.permute(0, 1, 3, 4, 2).reshape(Bp, 1, -1, Cp), pointmap.permute(0, 1, 3, 4, 2).reshape(Bp, 1, -1, Cp)
        else:
            if 'vggt_attn_cache' not in self.batch or str(vggt_layer) not in self.batch['vggt_attn_cache']:
                raise RuntimeError(f"[DEBUG] vggt_attn_cache missing layer {vggt_layer} in batch")
            layer_cache = self.batch['vggt_attn_cache'][str(vggt_layer)]
            if 'query' not in layer_cache or 'key' not in layer_cache:
                raise RuntimeError(f"[DEBUG] vggt_attn_cache[{vggt_layer}] missing 'query'/'key')")
            gt_query = layer_cache['query'].detach()
            gt_key = layer_cache['key'].detach()
            print(f"[DEBUG] gt_query.shape: {gt_query.shape}")
            print(f"[DEBUG] gt_key.shape: {gt_key.shape}")
            # 만약 레이어가 숫자면 1번 차원 (head) mean (keep dim)
            if str(vggt_layer).isdigit():
                gt_query = gt_query.mean(dim=1, keepdim=True)
                gt_key = gt_key.mean(dim=1, keepdim=True)
                print(f"[DEBUG] Applied head mean for numeric layer {vggt_layer}")
                print(f"[DEBUG] gt_query.shape after head mean: {gt_query.shape}")
                print(f"[DEBUG] gt_key.shape after head mean: {gt_key.shape}")
            return gt_query.shape, gt_query, gt_key
    
    def _get_gt_costmap(self, gt_query_resized, gt_key_resized, pair_metric: str, num_head: int):
        metric_fn = COST_METRIC_FN.get(pair_metric, None)
        print(f"[DEBUG] Using costmap metric: {pair_metric}")
        if metric_fn is None:
            raise ValueError(f"Unknown costmap metric {pair_metric}, falling back to neg_log_l2")
        
        
        pred_head = int(num_head)

        # Compute costmap once on the provided tensors (head dimension == 1 expected) to reduce peak GPU memory.
        # Then replicate the resulting per-head map across heads if needed.
        gt_attn = metric_fn(gt_query_resized, gt_key_resized)
        # metric_fn may return shape (B,1,Q,K) or (B,1,Q,K,1); normalize
        if gt_attn.dim() == 5:
            gt_attn = gt_attn.squeeze(-1)
        if gt_attn.dim() == 4 and gt_attn.shape[1] == 1:
            gt_attn_logit = gt_attn.squeeze(1)
        else:
            gt_attn_logit = gt_attn

        if pred_head > 1:
            # ensure shape is (B, Q, K) before expanding
            if gt_attn_logit.dim() == 3:
                gt_attn_logit = gt_attn_logit.unsqueeze(1)
            gt_attn_logit = gt_attn_logit.repeat(1, pred_head, 1, 1)

        return gt_attn_logit
    
    
    def _get_loss_fn(self, pair_loss_fn: str):
        # determine chosen loss name (pair override or global)
        if pair_loss_fn is None:
            raise ValueError(f"Pair must provide 'loss_fn' entry for loss calculation: pair={pair}")
        # support passing dict: {'fn': 'cross_entropy', 'head_mean': 'pre'}
        if isinstance(pair_loss_fn, dict):
            fn_name = pair_loss_fn.get('fn') or pair_loss_fn.get('name')
            head_mean = pair_loss_fn.get('head_mean', None)
            if fn_name is None:
                raise ValueError(f"loss_fn dict must contain 'fn'/'name' key: {pair_loss_fn}")
            key = str(fn_name).lower()
            if head_mean is not None:
                hm = str(head_mean).lower()
                if hm in ('none', 'null', 'no'):
                    # treat as no head-mean
                    key = f"{key}_pre"
                    is_head_mean = True
                elif hm in ('pre', 'post'):
                    key = f"{key}_{hm}"
                    is_head_mean = True
                else:
                    raise ValueError(f"Unsupported head_mean value: {head_mean}")
            else:
                key = f"{key}"
                is_head_mean = False
        else:
            # default; pre-headmean
            is_head_mean = True
            key = f"{key}_pre"
                
        loss_fn = self.ATTN_LOSS_FN.get(key, None)
        
        if loss_fn is None:
            raise ValueError(f"Unknown/unsupported loss key requested: {key}. Available: {list(self.ATTN_LOSS_FN.keys())}")
        print(f"[DEBUG] Using loss_fn: {key}")    
        return loss_fn, key, is_head_mean
    
    def _validate_matching_precision_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(cfg, dict):
            raise ValueError("matching_precision 설정은 dict 형태여야 합니다.")
        required_keys = [
            'grid_size',
            'knn_k',
            'max_matches',
            'pixel_thresholds',
            'query_view_indices',
            'key_view_indices',
        ]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"matching_precision 설정에서 다음 키가 누락되었습니다: {missing}")

        grid_size = int(cfg['grid_size'])
        if grid_size <= 0:
            raise ValueError(f"grid_size는 양수여야 합니다. 현재 값: {grid_size}")

        knn_k = int(cfg['knn_k'])
        if knn_k < 2:
            raise ValueError(f"knn_k는 2 이상이어야 합니다. 현재 값: {knn_k}")

        max_matches = int(cfg['max_matches'])
        if max_matches <= 0:
            raise ValueError(f"max_matches는 양수여야 합니다. 현재 값: {max_matches}")

        pixel_thresholds = [float(v) for v in cfg['pixel_thresholds']]
        if len(pixel_thresholds) == 0:
            raise ValueError("pixel_thresholds에는 하나 이상의 값이 필요합니다.")
        if any(v <= 0.0 for v in pixel_thresholds):
            raise ValueError(f"pixel_thresholds 항목은 모두 양수여야 합니다. 현재 값: {pixel_thresholds}")
        if pixel_thresholds != sorted(pixel_thresholds):
            raise ValueError(f"pixel_thresholds는 오름차순으로 정렬되어야 합니다. 현재 값: {pixel_thresholds}")

        query_view_indices = [int(v) for v in cfg['query_view_indices']]
        key_view_indices = [int(v) for v in cfg['key_view_indices']]
        if len(query_view_indices) == 0 or len(key_view_indices) == 0:
            raise ValueError("query_view_indices와 key_view_indices는 비어있을 수 없습니다.")
        if len(set(query_view_indices)) != len(query_view_indices):
            raise ValueError(f"query_view_indices에 중복이 있습니다: {query_view_indices}")
        if len(set(key_view_indices)) != len(key_view_indices):
            raise ValueError(f"key_view_indices에 중복이 있습니다: {key_view_indices}")

        return {
            'grid_size': grid_size,
            'knn_k': knn_k,
            'max_matches': max_matches,
            'pixel_thresholds': pixel_thresholds,
            'query_view_indices': query_view_indices,
            'key_view_indices': key_view_indices,
        }

    @staticmethod
    def _align_cfg_batch(tensor: torch.Tensor, expected_batch: int) -> torch.Tensor:
        if tensor.shape[0] == expected_batch:
            return tensor
        if tensor.shape[0] == 2 * expected_batch:
            return tensor[expected_batch:]
        raise ValueError(
            f"UNet Q/K 캐시의 batch 크기가 예상과 다릅니다. expected={expected_batch}, actual={tensor.shape[0]}"
        )

    @staticmethod
    def _reshape_qk_for_matching(tensor: torch.Tensor, F: int, grid_size: int) -> torch.Tensor:
        if tensor.dim() != 4:
            raise ValueError(f"Q/K 텐서는 4차원이어야 합니다. 현재 shape={tensor.shape}")
        B, n_heads, seq_len, head_dim = tensor.shape
        if seq_len % F != 0:
            raise ValueError(f"시퀀스 길이 {seq_len}은 뷰 수 {F}로 나누어 떨어져야 합니다.")
        tokens_per_view = seq_len // F
        side = int(math.sqrt(tokens_per_view))
        if side * side != tokens_per_view:
            raise ValueError(
                f"뷰당 토큰 수 {tokens_per_view}는 정사각형 격자(side^2) 형태여야 합니다."
            )
        feat = tensor.view(B, n_heads, F, side, side, head_dim)
        feat = feat.permute(0, 2, 1, 5, 3, 4).reshape(B, F, n_heads * head_dim, side, side)
        feat = feat.to(dtype=torch.float32)
        if side == grid_size:
            return feat
        feat_flat = feat.view(B * F, feat.shape[2], side, side)
        upsampled = FNN.interpolate(
            feat_flat, size=(grid_size, grid_size), mode='bilinear', align_corners=False
        )
        return upsampled.view(B, F, feat.shape[2], grid_size, grid_size)

    @staticmethod
    def _resize_point_map_for_matching(point_map: torch.Tensor, grid_size: int) -> torch.Tensor:
        if point_map is None:
            raise RuntimeError("point_map이 batch에 포함되어야 matching_precision을 계산할 수 있습니다.")
        if point_map.dim() != 5:
            raise ValueError(f"point_map 텐서는 5차원이어야 합니다. 현재 shape={point_map.shape}")
        B, F, C, H, W = point_map.shape
        if C != 3:
            raise ValueError(f"point_map 채널 수는 3이어야 합니다. 현재 값: {C}")
        point_map_cpu = point_map.detach().to(device='cpu', dtype=torch.float32)
        resized_batch = []
        for b in range(B):
            resized_views = []
            for v in range(F):
                view = point_map_cpu[b, v].unsqueeze(0)  # (1,3,H,W)
                resized = FNN.interpolate(
                    view, size=(grid_size, grid_size), mode='bilinear', align_corners=False
                )
                resized_views.append(resized.squeeze(0))
            resized_batch.append(torch.stack(resized_views, dim=0))
        return torch.stack(resized_batch, dim=0)  # (B,F,3,grid,grid)

    @staticmethod
    def _project_world_to_pixel_np(
        world_point: np.ndarray,
        w2c: np.ndarray,
        intrinsic: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        if world_point.shape[-1] != 3:
            raise ValueError(f"world_point는 길이 3이어야 합니다. 현재 shape={world_point.shape}")
        if w2c.shape[0] >= 4:
            w2c_use = w2c[:3, :]
        elif w2c.shape == (3, 4):
            w2c_use = w2c
        else:
            raise ValueError(f"extrinsic(w2c)의 shape이 3x4 또는 4x4 여야 합니다. 현재 shape={w2c.shape}")
        R = w2c_use[:, :3]
        t = w2c_use[:, 3]
        cam = R @ world_point.astype(np.float32) + t.astype(np.float32)
        z = cam[2]
        if abs(float(z)) < 1e-6:
            return None
        if intrinsic.shape[0] >= 3 and intrinsic.shape[1] >= 3:
            K = intrinsic[:3, :3]
        else:
            raise ValueError(f"intrinsic의 shape이 3x3 이상이어야 합니다. 현재 shape={intrinsic.shape}")
        uv_h = K @ cam
        denom = uv_h[2]
        if abs(float(denom)) < 1e-6:
            return None
        return float(uv_h[0] / denom), float(uv_h[1] / denom)

    @staticmethod
    def _format_precision_threshold(threshold: float) -> str:
        if abs(threshold - round(threshold)) < 1e-6:
            return str(int(round(threshold)))
        return f"{threshold:.3f}".rstrip('0').rstrip('.')

    def _compute_matching_precision(
        self,
        *,
        match_cfg: Dict[str, Any],
        qk_pair: Tuple[torch.Tensor, torch.Tensor],
        F: int,
        image_tensor: torch.Tensor,
        point_map: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        validated = self._validate_matching_precision_config(match_cfg)
        if qk_pair is None or not isinstance(qk_pair, (tuple, list)) or len(qk_pair) != 2:
            raise RuntimeError("matching_precision을 위해서는 UNet Q/K 캐시(tuple)가 필요합니다.")

        q_tensor = qk_pair[0].detach()
        k_tensor = qk_pair[1].detach()
        expected_batch = int(image_tensor.shape[0])
        q_tensor = self._align_cfg_batch(q_tensor, expected_batch)
        k_tensor = self._align_cfg_batch(k_tensor, expected_batch)

        grid_size = validated['grid_size']
        q_feat = self._reshape_qk_for_matching(q_tensor, F, grid_size)
        k_feat = self._reshape_qk_for_matching(k_tensor, F, grid_size)
        world_grid = self._resize_point_map_for_matching(point_map, grid_size)

        if intrinsics.shape[0] != expected_batch or extrinsics.shape[0] != expected_batch:
            raise ValueError("intrinsic/extrinsic의 batch 차원이 입력 이미지 batch와 일치해야 합니다.")
        intr_np = intrinsics.detach().to(device='cpu', dtype=torch.float32).numpy()
        extr_np = extrinsics.detach().to(device='cpu', dtype=torch.float32).numpy()

        query_views = validated['query_view_indices']
        key_views = validated['key_view_indices']
        for vid in query_views + key_views:
            if vid < 0 or vid >= F:
                raise ValueError(f"뷰 인덱스 {vid}가 범위를 벗어났습니다. 총 뷰 수: {F}")

        B = q_feat.shape[0]
        pixel_thresholds = validated['pixel_thresholds']
        max_matches = validated['max_matches']
        knn_k = validated['knn_k']
        if knn_k < 2:
            raise ValueError(f"knn_k는 2 이상이어야 합니다. 현재 값: {knn_k}")

        grid_axis = np.arange(grid_size, dtype=np.int32)
        xs, ys = np.meshgrid(grid_axis, grid_axis, indexing='xy')
        xs_flat = xs.reshape(-1)
        ys_flat = ys.reshape(-1)
        num_positions = xs_flat.shape[0]

        results = []
        for b in range(B):
            sample_height = int(image_tensor.shape[-2])
            sample_width = int(image_tensor.shape[-1])
            scale_x = float(sample_width) / float(grid_size)
            scale_y = float(sample_height) / float(grid_size)

            q_sample = q_feat[b]
            k_sample = k_feat[b]
            world_sample_np = world_grid[b].numpy()  # (F,3,grid,grid)
            world_flat = world_sample_np.reshape(F, 3, num_positions).transpose(0, 2, 1)

            query_features_list: List[torch.Tensor] = []
            query_meta_list: List[np.ndarray] = []
            for view_idx in query_views:
                feat = q_sample[view_idx]  # (C,grid,grid)
                feat_flat = feat.reshape(feat.shape[0], -1).transpose(0, 1).contiguous()
                query_features_list.append(feat_flat)
                view_ids = np.full(num_positions, view_idx, dtype=np.int32)
                query_meta_list.append(np.stack([view_ids, xs_flat, ys_flat], axis=1))
            query_features = torch.cat(query_features_list, dim=0)  # (Nq, C)
            query_meta_np = np.concatenate(query_meta_list, axis=0)

            key_features_list: List[torch.Tensor] = []
            key_meta_list: List[np.ndarray] = []
            for view_idx in key_views:
                feat = k_sample[view_idx]
                feat_flat = feat.reshape(feat.shape[0], -1).transpose(0, 1).contiguous()
                key_features_list.append(feat_flat)
                view_ids = np.full(num_positions, view_idx, dtype=np.int32)
                key_meta_list.append(np.stack([view_ids, xs_flat, ys_flat], axis=1))
            key_features = torch.cat(key_features_list, dim=0)  # (Nk, C)
            key_meta_np = np.concatenate(key_meta_list, axis=0)

            if key_features.shape[0] < 2:
                raise RuntimeError("matching_precision 계산을 위해서는 최소 두 개 이상의 key 피쳐가 필요합니다.")

            key_norm = (key_features ** 2).sum(dim=1)  # (Nk,)
            key_features_t = key_features.transpose(0, 1).contiguous()

            num_queries = query_features.shape[0]
            if num_queries == 0:
                raise RuntimeError("matching_precision 계산을 위한 query 피쳐가 비어 있습니다.")
            chunk = min(1024, num_queries)

            dist0_list: List[torch.Tensor] = []
            dist1_list: List[torch.Tensor] = []
            idx0_list: List[torch.Tensor] = []

            start_idx = 0
            while start_idx < num_queries:
                end_idx = min(start_idx + chunk, num_queries)
                q_chunk = query_features[start_idx:end_idx]  # (chunk, C)
                q_norm = (q_chunk ** 2).sum(dim=1)  # (chunk,)

                logits = torch.matmul(q_chunk, key_features_t)  # (chunk, Nk)
                dist = q_norm.unsqueeze(1) + key_norm.unsqueeze(0) - 2.0 * logits
                dist = torch.clamp(dist, min=0.0)

                top2 = torch.topk(dist, k=2, dim=1, largest=False, sorted=True)
                dist0_list.append(top2.values[:, 0])
                dist1_list.append(top2.values[:, 1])
                idx0_list.append(top2.indices[:, 0])

                start_idx = end_idx

            dist0 = torch.cat(dist0_list, dim=0)
            dist1 = torch.cat(dist1_list, dim=0)
            idx0 = torch.cat(idx0_list, dim=0)

            ratio = dist0 / (dist1 + 1e-12)
            ratio_np = ratio.detach().cpu().numpy()
            idx0_np = idx0.detach().cpu().numpy()

            order = np.argsort(ratio_np)
            selected = order[:max_matches]
            selected_key_indices = idx0_np[selected]

            selected_query_meta = query_meta_np[selected]
            selected_key_meta = key_meta_np[selected_key_indices]

            pred_x = (selected_key_meta[:, 1].astype(np.float32) + 0.5) * scale_x
            pred_y = (selected_key_meta[:, 2].astype(np.float32) + 0.5) * scale_y

            q_view_ids = selected_query_meta[:, 0].astype(np.int64)
            q_x_idx = selected_query_meta[:, 1].astype(np.int64)
            q_y_idx = selected_query_meta[:, 2].astype(np.int64)
            flat_indices = q_y_idx * grid_size + q_x_idx
            world_points = world_flat[q_view_ids, flat_indices]  # (N,3)

            errors = np.full(selected.shape[0], np.inf, dtype=np.float32)
            extr_sample = extr_np[b]
            intr_sample = intr_np[b]

            for i in range(selected.shape[0]):
                key_view = int(selected_key_meta[i, 0])
                if key_view < 0 or key_view >= F:
                    continue
                projected = self._project_world_to_pixel_np(
                    world_points[i],
                    extr_sample[key_view],
                    intr_sample[key_view],
                )
                if projected is None:
                    continue
                gt_x, gt_y = projected
                dx = float(pred_x[i]) - gt_x
                dy = float(pred_y[i]) - gt_y
                errors[i] = math.sqrt(dx * dx + dy * dy)

            total_matches = int(errors.shape[0])
            if total_matches == 0:
                raise RuntimeError("선택된 매칭이 없습니다.")

            precisions = {}
            finite_errors = errors
            for thr in pixel_thresholds:
                thr_val = float(thr)
                success = float(np.sum(finite_errors <= thr_val))
                precisions[thr_val] = success / float(total_matches)

            results.append({
                'precisions': precisions,
                'match_count': total_matches,
            })

        return results

    def _record_matching_precision(
        self,
        *,
        layer_key: str,
        step_index: int,
        results: List[Dict[str, Any]],
    ) -> None:
        for sample_idx, sample in enumerate(results):
            precisions = sample['precisions']
            match_count = sample['match_count']
            prefix = f"val/step{step_index}/{layer_key}/matching_precision/sample{sample_idx}"
            for thr_val, value in precisions.items():
                thr_key = self._format_precision_threshold(thr_val)
                metric_key = f"{prefix}/px{thr_key}"
                self.visualize_loss_dict[metric_key] = float(value)
                layer_entry = self.match_precision_records.setdefault(layer_key, {}).setdefault(thr_key, [])
                layer_entry.append(float(value))
            self.visualize_loss_dict[f"{prefix}/num_matches"] = float(match_count)

    def _execute_matching_precision(
        self,
        *,
        match_cfg: Dict[str, Any],
        qk_pair: Tuple[torch.Tensor, torch.Tensor],
        F: int,
        step_index: int,
        layer_key: str,
    ) -> None:
        if 'point_map' not in self.batch:
            raise RuntimeError("matching_precision을 계산하려면 batch에 'point_map'이 포함되어야 합니다.")
        if 'intrinsic' not in self.batch or 'extrinsic' not in self.batch:
            raise RuntimeError("matching_precision을 계산하려면 batch에 'intrinsic'과 'extrinsic'이 필요합니다.")
        if qk_pair is None:
            raise RuntimeError(
                "matching_precision을 위해 UNet Q/K 캐시가 필요합니다. validate.py에서 set_qk_cache가 활성화되어 있는지 확인하세요."
            )
        results = self._compute_matching_precision(
            match_cfg=match_cfg,
            qk_pair=qk_pair,
            F=F,
            image_tensor=self.batch['image'],
            point_map=self.batch['point_map'],
            intrinsics=self.batch['intrinsic'],
            extrinsics=self.batch['extrinsic'],
        )
        self._record_matching_precision(
            layer_key=layer_key,
            step_index=step_index,
            results=results,
        )

    def __call__(
        self,
        pipeline,
        step_index: int,
        timestep: int,
        callback_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        매 denoising step마다 호출되는 callback 함수
        UNet attention을 캐시에서 가져와 VGGT attention과 비교하여 loss 계산
        """
        viz_enabled, loss_enabled = self._is_step_enabled(step_index) 
          
        if not (viz_enabled or loss_enabled):
            print(f"[DEBUG] Skipping attention visualization callback for step {step_index}")
            clear_attn_cache(pipeline.unet)
            return callback_kwargs
        
        print(f"[DEBUG] AttentionVisualizationCallback.__call__ invoked: step_index={step_index}, timestep={timestep}")
        print(f"[DEBUG] do_attn_visualize: {self.do_attn_visualize}")
        
        ### 주 Logic ###
        try:
            # UNet attention cache 가져오기
            unet_attn_cache = pop_cached_attn(pipeline.unet)
            unet_qk_cache = pop_cached_qk(pipeline.unet)
            print(f"[DEBUG] UNet attention cache keys: {list(unet_attn_cache.keys()) if unet_attn_cache else 'None'}")
            print(f"[DEBUG] UNet QK cache keys: {list(unet_qk_cache.keys()) if unet_qk_cache else 'None'}")
            
            if not unet_attn_cache:
                # attention cache가 비어있으면 스킵
                print("[DEBUG] UNet attention cache is empty, skipping")
                return callback_kwargs
            
            # 이미지 shape 정보
            image = self.batch['image']
            B, F, _, H, W = image.shape
            
            step_loss_dict = {}
            current_unet = None
            pred_attn_logit = None
            
            visualize_pairs = self._process_layer_pair()

            for pair in visualize_pairs:
                
                if isinstance(pair, dict):
                    unet_layer = pair.get('unet_layer')
                    vggt_layer = pair.get('vggt_layer')
                    pair_metric = pair.get('costmap_metric', None)
                    pair_loss_fn = pair.get('loss_fn', None)
                else:
                    raise ValueError(f"invalid pair entry: {pair}")

                pred_attn_logit, qk_entry, current_unet = self._get_pred_attn_logit(
                    unet_layer, current_unet, unet_attn_cache, unet_qk_cache
                )

                print(f"[DEBUG] Processing layer pair: {unet_layer}, {vggt_layer}")
                print(f"[DEBUG] UNet attention cache keys: {list(unet_attn_cache.keys()) if unet_attn_cache else 'None'}")
                print(f"[DEBUG] VGGT attention cache keys: {list(self.batch.get('vggt_attn_cache', {}).keys()) if self.batch.get('vggt_attn_cache') else 'None'}")

                gt_query_shape, gt_query, gt_key = self._get_gt_tokens(vggt_layer)

                # Ensure pred exists 
                if pred_attn_logit is None:
                    raise RuntimeError(f"[DEBUG] missing UNet attention logits for unet layer {unet_layer}")
                Q, K = pred_attn_logit.shape[-2], pred_attn_logit.shape[-1]
                if Q != K:
                    raise ValueError(f"pred attn must have equal Q and K dims, got {pred_attn_logit.shape}")
                if Q % F != 0 or K % F != 0:
                    raise ValueError(f"pred attnmap must be divisible by F; got {pred_attn_logit.shape} and F={F}")

                #  move GT to the same dtype/device as pred
                target_dtype = pred_attn_logit.dtype
                target_device = pred_attn_logit.device
                gt_query = gt_query.to(dtype=target_dtype, device=target_device)
                gt_key = gt_key.to(dtype=target_dtype, device=target_device)
            
                pred_query_size = int(math.sqrt(Q // F))
                pred_key_size = int(math.sqrt(K // F))
                gt_query_resized = self._resize_token(gt_query, pred_query_size, F)
                gt_key_resized = self._resize_token(gt_key, pred_key_size, F)
                
                # Compute GT attention logits once (used by both loss and viz branches).
                # This prevents UnboundLocalError when viz is enabled but loss is not.
                gt_attn_logit = self._get_gt_costmap(gt_query_resized, gt_key_resized, pair_metric, num_head=pred_attn_logit.shape[1])
                ### 차원 정리
                # gt_query_resized: (B, 1, pred_query_size, pred_query_size, C)
                # gt_key_resized: (B, 1, pred_key_size, pred_key_size, C)
                # pred_attn_logit: (B, Head, pred_query_size, pred_key_size)
                
                unet_softmax_head, vggt_softmax_head, unet_viz_head, vggt_viz_head = self._resolve_heads(pair, F)
                if unet_softmax_head is None or vggt_softmax_head is None or unet_viz_head is None or vggt_viz_head is None:
                    raise RuntimeError(f"Missing unet/vggt logit head for pair: {pair}")

                # Define layer_key once for both loss and visualization branches to
                # avoid UnboundLocalError when one branch is disabled.
                layer_key = f"unet{unet_layer}_vggt{vggt_layer}"

                match_cfg = pair.get('matching_precision', None) if isinstance(pair, dict) else None
                if match_cfg is not None:
                    self._execute_matching_precision(
                        match_cfg=match_cfg,
                        qk_pair=qk_entry,
                        F=F,
                        step_index=step_index,
                        layer_key=layer_key,
                    )

                if loss_enabled:
                    print(f"Calculating loss for step {step_index}, layer {layer_key}")
                    loss_query_idx, loss_key_idx = self._extract_view_indices(F, mode="loss")
                    
                    # pred_attn_logit_sliced: (B, Head, pred_query_size, pred_key_size) / (2,10,3072,3072)
                    pred_attn_logit_sliced = self._slice_attention_map(pred_attn_logit, loss_query_idx, loss_key_idx, F)
                    
                    # gt_attn_logit_sliced: (B, Head, pred_query_size, pred_key_size)
                    gt_attn_logit_sliced = self._slice_attention_map(gt_attn_logit, loss_query_idx, loss_key_idx, F)

                    print(f"[DEBUG] pred_attn_logit_sliced.shape: {pred_attn_logit_sliced.shape}")
                    print(f"[DEBUG] gt_attn_logit_sliced.shape: {gt_attn_logit_sliced.shape}")
                    
                    # Instantiate logit-heads for UNet and VGGT outputs.
                    # Preference order: 1) per-pair explicit head class, 2) global visualize_config
                    # if per-view => (View, Head, HW, View, HW)
                    # else global => (view, Head, HW, View*HW)
                    pred_processed_viz = unet_softmax_head(pred_attn_logit_sliced)
                    gt_processed_viz = vggt_softmax_head(gt_attn_logit_sliced)
                    
                    # aggregated loss 계산
                    loss_fn, chosen_fn_str, is_head_mean = self._get_loss_fn(pair_loss_fn)
                    loss_value = loss_fn(pred_processed_viz, gt_processed_viz)

                    # include chosen loss function name in the step-level key
                    chosen_fn_str = str(chosen_fn_str).replace('/', '_')
                    
                    # compute per-head tensor and reshape to (B, H) when possible
                    per_head_tensor = loss_value.detach().cpu().view(-1)
                    B = int(pred_attn_logit_sliced.shape[0])
                    H = int(pred_attn_logit_sliced.shape[1])
                    if per_head_tensor.numel() == B * H:
                        per_head_2d = per_head_tensor.view(B, H)
                    else:
                        # fallback: keep flattened as single-batch
                        per_head_2d = per_head_tensor.view(1, -1)

                    per_head_list = per_head_2d.view(-1).tolist()
                    loss_scalar = float(sum(per_head_list) / len(per_head_list)) if per_head_list else 0.0

                    # Use pair+loss_fn(+head) keys for metrics (avoid exploding metric names with denoise step)
                    step_loss_dict[f"val/{layer_key}/{chosen_fn_str}"] = loss_scalar

                    # aggregate per-head across batch for metric display
                    for hid in range(per_head_2d.shape[1]):
                        head_vals = [float(per_head_2d[b, hid].item()) for b in range(per_head_2d.shape[0])]
                        step_loss_dict[f"val/{layer_key}/{chosen_fn_str}/head{hid}"] = sum(head_vals) / len(head_vals)

                    # collect detailed raw rows for export (one row per sample x head)
                    seq = self.visualize_config.get('viz_seq_name', None)
                    sample_idx = self.visualize_config.get('viz_wandb_step_base', None)
                    if not hasattr(self, '_detailed_rows'):
                        self._detailed_rows = []
                    for b in range(per_head_2d.shape[0]):
                        for hid in range(per_head_2d.shape[1]):
                            self._detailed_rows.append({
                                "sample_idx": sample_idx,
                                "sequence_name": seq,
                                "denoise_step_index": int(step_index),
                                "timestep": int(timestep),
                                "pair_id": layer_key,
                                "loss_fn": chosen_fn_str,
                                "head": int(hid),
                                "value": float(per_head_2d[b, hid].item())
                            })

                    print(f"[DEBUG] Calculated loss for {layer_key} (fn={chosen_fn_str}): avg={loss_scalar}, per_head={per_head_list}")

                    if layer_key not in self.layer_losses:
                        self.layer_losses[layer_key] = []
                    # store the per-head list (not scalar) for later analysis
                    self.layer_losses[layer_key].append(per_head_list)
                    
                if viz_enabled:
                    viz_query_idx, viz_key_idx = self._extract_view_indices(F, mode="viz")
                    
                    pred_attn_logit_viz = self._slice_attention_map(pred_attn_logit, viz_query_idx, viz_key_idx, F)
                    gt_attn_logit_viz = self._slice_attention_map(gt_attn_logit, viz_query_idx, viz_key_idx, F)

                    # For visualization, always use Softmax_HeadMean and only consider
                    # viz_mode = pair.get('viz_softmax_mode', None)
                    # determine number of key-views for viz (prefer pair, then global config)
                    # viz_num_k = pair.get('viz_num_key_views', None)
                    # if viz_num_k is None:
                    #     viz_num_k = int(self.visualize_config.get('viz_num_key_views', self.visualize_config.get('loss_num_key_views', 1)))
                    # per_view_flag = str(viz_mode).lower() == 'per_view'
                    # num_view_for_per_view = int(viz_num_k) if per_view_flag else None
                    
                    # import pdb; pdb.set_trace()

                    pred_processed_viz = unet_viz_head(pred_attn_logit_viz)
                    gt_processed_viz = vggt_viz_head(gt_attn_logit_viz)
                    
                    print(f"[DEBUG] pred_processed_viz.shape: {pred_processed_viz.shape}")
                    print(f"[DEBUG] gt_processed_viz.shape: {gt_processed_viz.shape}")
                    
                    # if len(pred_processed_viz.shape)== 5:
                    #     pred_processed_viz = pred_processed_viz.mean(dim=-2)
                    # if len(gt_processed_viz.shape)== 5:
                    #     gt_processed_viz = gt_processed_viz.mean(dim=-2)
                        
                    # print(f"[DEBUG] pred_processed_viz.shape: {pred_processed_viz.shape}")
                    # print(f"[DEBUG] gt_processed_viz.shape: {gt_processed_viz.shape}")

                    self._maybe_save_attn_overlay(
                        step_index=step_index,
                        layer_key=layer_key,
                        pred_logits=pred_processed_viz,
                        gt_logits=gt_processed_viz,
                        F=F,
                        query_idx_list=viz_query_idx,
                        key_idx_list=viz_key_idx,
                    )

                # cleanup per-pair intermediates
                try:
                    del gt_query, gt_key, gt_query_resized, gt_key_resized, gt_attn_logit, gt_attn_logit_sliced
                except Exception:
                    pass
                try:
                    del pred_attn_logit_viz, gt_attn_logit_viz, pred_attn_logit_sliced
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # after finishing all pairs for all unet layers, free any remaining pred cache entries
            # Now aggregate step-level losses collected in step_loss_dict
            if step_loss_dict:
                try:
                    step_avg_loss = sum(step_loss_dict.values()) / len(step_loss_dict)
                except Exception:
                    # convert tensors to floats if necessary
                    vals = [v.item() if hasattr(v, 'item') else float(v) for v in step_loss_dict.values()]
                    step_avg_loss = sum(vals) / len(vals)
                step_loss_dict[f"val/step{step_index}/avg_loss"] = step_avg_loss

                # step별 layer별 loss 저장
                self.step_layer_losses[step_index] = {}
                for key, loss_value in step_loss_dict.items():
                    if key.startswith(f"val/step{step_index}/") and key != f"val/step{step_index}/avg_loss":
                        # Preserve the rest of the key after val/step{step_index}/ as-is
                        layer_key = key.replace(f"val/step{step_index}/", "")
                        # store numeric value for metric
                        self.step_layer_losses[step_index][layer_key] = loss_value.item() if hasattr(loss_value, 'item') else float(loss_value)

                self.step_losses.append(step_avg_loss.item() if hasattr(step_avg_loss, 'item') else float(step_avg_loss))
                print(f"[DEBUG] Step {step_index} avg loss: {float(step_avg_loss):.6f}")

                # 전체 loss dict에 추가
                self.visualize_loss_dict.update(step_loss_dict)
                
                
        except Exception as e:
            print(f"Error in attention visualization callback at step {step_index}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Avoid clearing processor-level attention caches here because cached
            # tensors are already popped by `pop_cached_attn`. Only free global
            # GPU memory to avoid surprising removal of cached data across calls.
            # If tweedie is enabled, let errors propagate (no fallback)
            if self._use_tweedie_bg:
                self._maybe_store_tweedie_bg(pipeline, step_index, timestep, callback_kwargs)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if step_index % 10 == 0:
                print(f"Emptied CUDA cache after step {step_index}")
        
        return callback_kwargs
    
    def get_final_loss(self) -> torch.Tensor:
        """모든 step의 평균 loss 반환"""
        if not self.step_losses:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        return torch.tensor(sum(self.step_losses) / len(self.step_losses), 
                          device=self.device, dtype=torch.float32)
    
    def get_loss_dict(self) -> Dict[str, torch.Tensor]:
        """전체 loss dictionary 반환"""
        # 최종 평균 loss도 포함
        final_dict = self.visualize_loss_dict.copy()
        final_dict["val/visualize/final_avg_loss"] = self.get_final_loss()
        return final_dict
    
    def get_structured_losses(self) -> Dict[str, Any]:
        """구조화된 loss 정보 반환 (wandb 로깅용)"""
        structured_losses = {
            'step_losses': self.step_losses,
            'step_layer_losses': self.step_layer_losses,
            'layer_summary': {},
            'overall_summary': {},
            'matching_precision': {},
        }
        
        # Layer별 통계 계산
        for layer_key, losses in self.layer_losses.items():
            if losses:
                # losses may contain per-step per-head lists; normalize to per-step scalars
                scalar_vals = []
                for item in losses:
                    if isinstance(item, (list, tuple)):
                        # average per-head values to a scalar for summary
                        try:
                            scalar_vals.append(float(sum(item) / len(item)))
                        except Exception:
                            # fallback: coerce first element
                            scalar_vals.append(float(item[0]))
                    else:
                        scalar_vals.append(float(item))

                mean_val = sum(scalar_vals) / len(scalar_vals)
                min_val = min(scalar_vals)
                max_val = max(scalar_vals)
                std_val = (sum((x - mean_val) ** 2 for x in scalar_vals) / len(scalar_vals)) ** 0.5 if len(scalar_vals) > 1 else 0.0

                structured_losses['layer_summary'][layer_key] = {
                    'mean': mean_val,
                    'min': min_val,
                    'max': max_val,
                    'std': std_val,
                    'count': len(losses)
                }
        
        # 전체 통계
        if self.step_losses:
            structured_losses['overall_summary'] = {
                'mean': sum(self.step_losses) / len(self.step_losses),
                'min': min(self.step_losses),
                'max': max(self.step_losses),
                'std': (sum((x - sum(self.step_losses)/len(self.step_losses))**2 for x in self.step_losses) / len(self.step_losses))**0.5 if len(self.step_losses) > 1 else 0.0,
                'total_steps': len(self.step_losses)
            }
        
        if self.match_precision_records:
            mp_summary = {}
            for layer_key, threshold_map in self.match_precision_records.items():
                layer_summary = {}
                for thr_key, values in threshold_map.items():
                    if not values:
                        continue
                    mean_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 0.0
                    layer_summary[thr_key] = {
                        'mean': mean_val,
                        'min': min_val,
                        'max': max_val,
                        'std': std_val,
                        'count': len(values),
                    }
                if layer_summary:
                    mp_summary[layer_key] = layer_summary
            structured_losses['matching_precision'] = mp_summary
        
        return structured_losses
    
    def get_attention_images(self) -> Dict[str, Dict[str, Any]]:
        """저장된 attention 이미지들 반환 (wandb 로깅용)"""
        # return list-form images for downstream logging
        if hasattr(self, 'attention_images_list'):
            return list(self.attention_images_list)
        return []
    
    def clear_vggt_cache(self):
        """VGGT attention cache 정리 (pipeline 완료 후 호출)"""
        if 'vggt_attn_cache' in self.batch:
            # VGGT cache 메모리 정리
            for layer_key in list(self.batch['vggt_attn_cache'].keys()):
                if 'query' in self.batch['vggt_attn_cache'][layer_key]:
                    del self.batch['vggt_attn_cache'][layer_key]['query']
                if 'key' in self.batch['vggt_attn_cache'][layer_key]:
                    del self.batch['vggt_attn_cache'][layer_key]['key']
                del self.batch['vggt_attn_cache'][layer_key]
            del self.batch['vggt_attn_cache']
            print("Cleared VGGT attention cache")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
    
    def reset(self):
        """callback 상태 초기화"""
        # flush detailed rows to disk/artifact before clearing
        try:
            self.flush_detailed_rows()
        except Exception:
            # do not raise from reset; best-effort flush
            pass

        self.visualize_loss_dict.clear()
        self.step_losses.clear()
        self.step_layer_losses.clear()
        self.layer_losses.clear()
        self.match_precision_records.clear()
        if hasattr(self, 'attention_images'):
            self.attention_images.clear()
        # clear collected detailed rows
        if hasattr(self, '_detailed_rows'):
            self._detailed_rows.clear()

    def get_detailed_rows(self) -> List[Dict[str, Any]]:
        """Return a copy of collected detailed rows (one row per sample x head)."""
        if not hasattr(self, '_detailed_rows'):
            return []
        return list(self._detailed_rows)

    def flush_detailed_rows(self, filename: Optional[str] = None, upload_wandb: bool = True) -> Optional[str]:
        """
        Flush collected detailed rows to a parquet file. If `upload_wandb` is True and
        wandb is available and `viz_log_wandb` is set, an artifact is uploaded.

        Returns the written filename or None if nothing was written.
        """
        if not hasattr(self, '_detailed_rows') or len(self._detailed_rows) == 0:
            return None

        out_dir = self.visualize_config.get('viz_save_dir', None)
        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            ts = int(time.time())
            seq = self.visualize_config.get('viz_seq_name', 'sample')
            filename = os.path.join(out_dir, f"attn_details_{seq}_{ts}.parquet")

        try:
            df = pd.DataFrame(self._detailed_rows)
            # prefer parquet (snappy) but fallback to csv if pyarrow not installed
            try:
                df.to_parquet(filename, compression='snappy', index=False)
            except Exception:
                csv_fn = filename.replace('.parquet', '.csv')
                df.to_csv(csv_fn, index=False)
                filename = csv_fn

            # optionally upload as wandb artifact
            # always attempt wandb upload when wandb is importable and viz_log_wandb is True
            if self.visualize_config.get('viz_log_wandb', False):
                try:
                    import wandb
                    art_name = f"attn-details-{self.visualize_config.get('viz_seq_name', 'sample')}-{ts}"
                    artifact = wandb.Artifact(art_name, type="attn-details")
                    artifact.add_file(filename)
                    wandb.log_artifact(artifact)
                except Exception:
                    # non-fatal: ignore upload errors
                    pass

            return filename
        finally:
            # do not clear rows here; caller may want to keep in-memory until reset
            pass

