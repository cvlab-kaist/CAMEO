import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
'''
x = [B, Head, Q, K]
'''
class ArgmaxOneHot(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        """
        x : B Head Q K(num_view HW)
        Returns: one-hot encoding of argmin along dimension dim
        """
        idx = torch.argmax(x, dim=-1, keepdim=True)
        onehot = torch.zeros_like(x).scatter_(-1, idx, 1.0)
        return onehot

def build_mlp(in_dim, out_dim, mlp_ratio=4.0, mlp_depth=1, mid_dim = None):
    
    mlp = []
    if mlp_depth <= 0:
        mlp += [nn.Linear(in_dim, out_dim)]
    else:
        if mid_dim is not None:
            assert mlp_ratio is None
        else:
            mid_dim = int(in_dim * mlp_ratio)
        mlp += [nn.Linear(in_dim, mid_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            mlp += [nn.Linear(mid_dim, mid_dim), nn.GELU()]
        mlp += [nn.Linear(mid_dim, out_dim)]
    return nn.Sequential(*mlp)

class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x, **kwargs):
        return x
    
class Softmax(nn.Module):
    def __init__(self, 
                 per_view = False,
                 softmax_temp=1.0, learnable_temp = False, **kwargs):
        super(Softmax, self).__init__()
        self.per_view = per_view
        self.softmax_temp = softmax_temp
        if learnable_temp:
            self.softmax_temp = nn.Parameter(torch.tensor(softmax_temp))

    def forward(self, x, num_view=None, **kwargs):
        if self.per_view:
            K = x.shape[-1]
            HW = K // num_view
            x = x.reshape(*x.shape[:-1], num_view, HW)
        x = x / self.softmax_temp
        x = x.softmax(dim=-1)
        if self.per_view:
            x = x.reshape(*x.shape[:-2], num_view*HW)
        return x
 
class Softmax_HeadMean(Softmax):
    def forward(self, x, num_view=None, **kwargs):
        ''' x : B Head Q K(num_view HW)
        '''
        x = super().forward(x, num_view, **kwargs)
        return x.mean(dim=1, keepdim=True)

class Softmax_HeadMlp(Softmax):
    def __init__(self, 
            in_head_num = 24, out_head_num = 1,
            mlp_ratio = 4.0, mlp_depth = 1,
            final_activation = None,
            **kwargs
        ):
        super(Softmax_HeadMlp, self).__init__(**kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x, num_view= None, **kwargs):
        x = super().forward(x, num_view, **kwargs)
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        x = self.final_activation(x)
        return x

class HeadMlp_Softmax(Softmax):
    def __init__(self, 
            in_head_num = 24, out_head_num = 1,
            mlp_ratio = 4.0, mlp_depth = 1,
            **kwargs
        ):
        super(HeadMlp_Softmax, self).__init__(**kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)

    def forward(self, x, num_view=None,**kwargs):
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        x = super().forward(x, num_view, **kwargs)
        return x

class TimeAdaTemp_HeadMlp_Softmax(Softmax):
    def __init__(self, 
            adamlp_mid_dim = 64,
            adamlp_depth = 1, 
            # mlp
            in_head_num = 24, out_head_num = 1,
            mlp_ratio = 4.0, mlp_depth = 1,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.mlp = build_mlp(in_head_num, out_head_num, mlp_ratio, mlp_depth)
        self.ada_scale = build_mlp(1, 1, mlp_ratio=None, mlp_depth=adamlp_depth, mid_dim=adamlp_mid_dim)
        nn.init.zeros_(self.ada_scale[-1].weight)
        nn.init.zeros_(self.ada_scale[-1].bias)
    
    def forward(self, x, num_view, timesteps, **kwargs):
        # 1) logit head mlp
        x = x.permute(0,2,3,1) # B Q K Head
        x = self.mlp(x).permute(0,3,1,2) # B Out_head Q K
        # 2) ada temp
        scale = self.ada_scale(timesteps[:, None]) # B 1
        scale = (scale + 1).view(-1,1,1,1) # B 1 1 1
        x = x * scale
        # softmax
        x = super().forward(x, num_view, **kwargs)
        return x

class LinearTime2Temp_Softmax_HeadMean(Softmax_HeadMean):
    def timesteps_to_temp(self, timesteps):
        min_temp = 0.0001
        max_temp = 0.1
        # linear temp
        temp = min_temp + (max_temp - min_temp) * (timesteps / 1000.0)
        return temp
    def forward(self, x, num_view, timesteps, **kwargs):
        ''' x : B Head Q K(num_view HW)
        '''
        assert self.softmax_temp == 1.0, "LinearTime2Temp_Softmax_Headmean requires softmax_temp=1.0"
        temp = self.timesteps_to_temp(timesteps)
        x = x / temp[:, None, None, None]
        return super().forward(x, num_view, **kwargs)



def cycle_consistency_checker(costmap, pixel_threshold=None):
    ''' costmap : [B, HW, (V-1)*HW]
        cross cost/attn only
    '''
    # Get dimensions from the input tensor
    B, HW, _ = costmap.shape
    device = costmap.device

    H = W = int(math.sqrt(HW))

    # Step 2: Find the best initial matches (already vectorized)
    max_idx = torch.argmax(costmap, dim=-1)  # (B, HW)
    transpose_costmap = costmap.transpose(1, 2)  # (B, 2HW, HW)
    b_idx = torch.arange(B, device=device)[:, None] # (B, 1)
    reverse_map = transpose_costmap[b_idx, max_idx] # (B, HW, HW)
    _, final_indices = torch.max(reverse_map, dim=-1) 
    # Create a tensor representing the original indices (0, 1, 2, ..., HW-1) for each batch item
    original_indices = torch.arange(HW, device=device).expand(B, -1) # (B, HW)

    # Convert 1D indices to 2D coordinates for both original and matched points
    x1 = original_indices // W
    y1 = original_indices % H
    x2 = final_indices // W
    y2 = final_indices % H

    if pixel_threshold is None:
        pixel_threshold = round(H / 10.0)

    # Calculate Euclidean distance and check if it's within the threshold
    distance = torch.sqrt(((x1 - x2)**2 + (y1 - y2)**2).float())
    is_close = (distance < pixel_threshold).float() # .float() converts boolean (True/False) to (1.0/0.0)

    # Step 6: Reshape final output tensor
    final_distance_tensor = is_close.view(B, HW, 1)

    return final_distance_tensor



LOGIT_HEAD_CLS = {
    "identity": Identity,
    "softmax": Softmax,
    "softmax_headmean": Softmax_HeadMean,
    "softmax_headmlp": Softmax_HeadMlp,
    "headmlp_softmax": HeadMlp_Softmax,
    "lineartime2temp_softmax_headmean": LinearTime2Temp_Softmax_HeadMean,
    "timeadatemp_headmlp_softmax": TimeAdaTemp_HeadMlp_Softmax,
    "argmaxonehot": ArgmaxOneHot,
    # "headmlp": HeadMlp,
    # "headmlp_softargmax": HeadMlp_SoftArgmax,
    # "headmean_softargmax": HeadMean_SoftArgmax,
    # "interpolate4d_headconv4d_softmax": Interpolate4D_HeadConv4D_Softmax,
}

# def per_view_softargmax2d(prob, num_view=1):
#     ''' leffa: https://arxiv.org/pdf/2412.08486v2
#         args
#             - prob: [..., vhw]
#         return
#             - indices: [..., v, 2]
#     '''
#     K = prob.shape[-1]
#     hw = K // num_view
#     prob = prob.reshape(*prob.shape[:-1], num_view, hw)
#     h = w = int(hw**0.5)

#     indices_c, indices_r = np.meshgrid(
#         np.linspace(-1, 1, w),
#         np.linspace(-1, 1, h),
#         indexing='xy'
#     )

#     indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(prob.device, dtype=prob.dtype)
#     indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(prob.device, dtype=prob.dtype)

#     result_r = torch.sum(prob * indices_r, dim=-1)
#     result_c = torch.sum(prob * indices_c, dim=-1)

#     result = torch.stack([result_r, result_c], dim=-1)

#     return result

# class HeadMean_SoftArgmax(Softmax_HeadMean):
#     ''' Softmax -> HeadMean -> Prob * idx
#     '''
#     def __init__(self, 
#         compute_entropy= False,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         assert self.per_view == True, "SoftArgmax requires per_view computation"
#         self.compute_entropy = compute_entropy
#         self.entropy_val =  None
#     def forward(self, x, num_view=None, **kwargs):
#         assert num_view is not None, "softargmax requires num_view"
#         x = super().forward(x, num_view) # per_view prob: B 1 Q K(VHW)
#         if self.compute_entropy:
#             K = x.shape[-1]
#             HW = K // num_view
#             per_view_prob = x.reshape(*x.shape[-1], num_view, HW)
#             eps = 1e-8
#             self.entropy_val = - (per_view_prob * (per_view_prob + eps).log()).sum(dim=-1) # B 1 Q V
#         x = per_view_softargmax2d(x, num_view) # B 1 Q V 2
#         return x.flatten(-2,-1) # B 1 Q V*2


# class HeadMlp_SoftArgmax(HeadMlp_Softmax):
#     def __init__(self,
#         compute_entropy= False,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         assert self.per_view == True, "SoftArgmax requires per_view computation"
#         self.compute_entropy = compute_entropy
#         self.entropy_val =  None

#     def forward(self, x, num_view=None, **kwargs):
#         ''' x: B Head Q K
#             return: B Head Q 1 2 or  B Head Q key_num_view 2 (per view)
#         '''
#         assert num_view is not None, "softargmax must provide num_view"
#         x = super().forward(x, num_view, **kwargs) # per_view prob: B 1 Q K(VHW)
#         if self.compute_entropy:
#             K = x.shape[-1]
#             HW = K // num_view
#             per_view_prob = x.reshape(*x.shape[-1], num_view, HW)
#             eps = 1e-8
#             self.entropy_val = - (per_view_prob * (per_view_prob + eps).log()).sum(dim=-1) # B 1 Q V
#         x = per_view_softargmax2d(x, num_view) # B 1 Q V 2
#         return x.flatten(-2,-1) # B 1 Q V*2
    
# class Conv4D(nn.Module):
#     ''' https://github.com/cvlab-kaist/AM-Adapter/blob/main/src/models/conv4d.py
#     '''
#     def __init__(self,
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         **kwargs
#     ):
#         super(Conv4D, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

#     def forward(self, x, num_view, num_view_query, **kwargs):
#         ''' x: B Head Q K = B Head VHW VHW
#         '''
#         B, Head, Q, K = x.shape
#         v_q, h_q, w_q  = num_view_query, int((Q // num_view_query) ** 0.5), int((Q // num_view_query) ** 0.5) 
#         v_k, h_k, w_k  = num_view, int((K // num_view) ** 0.5), int((K // num_view) ** 0.5)
#         assert h_q == h_k and w_q == w_k, "4D conv requires same spatial size between query and key"
#         x_query, x_key = x.clone(), x.clone()
#         x_query = rearrange(x_query, "B Head (v_q h_q w_q) (v_k h_k w_k) -> (B v_q v_k h_k w_k) Head h_q w_q", v_q=v_q, v_k=v_k, h_q=h_q, w_q=w_q, h_k=h_k, w_k=w_k) # (B v1 v2 h_k w_k) Head h_q w_q
#         x_query = self.query_conv(x_query) 
#         x_query = rearrange(x_query, "(B v_q v_k h_k w_k) Head h_q w_q -> B Head (v_q h_q w_q) (v_k h_k w_k)", B=B, v_q=v_q, v_k=v_k, h_k=h_k, w_k=w_k, h_q=h_q, w_q=w_q) # B Head Q K

#         x_key = rearrange(x_key, "B Head (v_q h_q w_q) (v_k h_k w_k) -> (B v_q v_k h_q w_q) Head h_k w_k", v_q=v_q, v_k=v_k, h_q=h_q, w_q=w_q, h_k=h_k, w_k=w_k) # (B v1 v2 h_q w_q) Head h_k w_k
#         x_key = self.key_conv(x_key) 
#         x_key = rearrange(x_key, "(B v_q v_k h_q w_q) Head h_k w_k -> B Head (v_q h_q w_q) (v_k h_k w_k)", B=B, v_q=v_q, v_k=v_k, h_k=h_k, w_k=w_k, h_q=h_q, w_q=w_q) # B Head Q K

#         x = x_query + x_key
#         return x

# class Interpolate4D_HeadConv4D_Softmax(Softmax):
#     def __init__(self, 
#         in_head_num, 
#         conv_dim_ratio, # mid_dim / in_dim
#         group_num, 
#         depth, 
#         out_head_num = 1,
#         target_size = None,
#         interpolate_mode = 'nearest',
#         kernel_size=3, 
#         stride=1, 
#         padding=1, 
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.target_size = target_size
#         self.interpolate_mode = interpolate_mode
#         net = []
#         if depth <= 0:
#             net += [Conv4D(in_head_num, out_head_num, kernel_size=kernel_size, stride=stride, padding=padding)]
#         else:
#             mid_dim = int(in_head_num * conv_dim_ratio)
#             net += [Conv4D(in_head_num, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#                         nn.GroupNorm(group_num, mid_dim),
#                         nn.GELU()]
#             for _ in range(depth - 1):
#                 net += [Conv4D(mid_dim, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#                         nn.GroupNorm(group_num, mid_dim),
#                         nn.GELU()]
#             net += [Conv4D(mid_dim, out_head_num, kernel_size=kernel_size, stride=stride, padding=padding)]
#         self.net = nn.Sequential(*net)

#     def forward(self, x, num_view, num_view_query, **kwargs):
#         ''' x: B Head Q K = B Head VHW VHW
#         '''
#         # interpolate 4D
#         if self.target_size is not None:
#             B, Head, Q, K = x.shape
#             v_q, h_q, w_q  = num_view_query, int((Q // num_view_query) ** 0.5), int((Q // num_view_query) ** 0.5) 
#             v_k, h_k, w_k  = num_view, int((K // num_view) ** 0.5), int((K // num_view) ** 0.5)
#             x = rearrange(x, "B Head (v_q h_q w_q) (v_k h_k w_k) -> (B v_q v_k h_k w_k) Head h_q w_q", v_q=v_q, v_k=v_k, h_q=h_q, w_q=w_q, h_k=h_k, w_k=w_k) # (B v1 v2 h_k w_k) Head h_q w_q
#             x = nn.functional.interpolate(x, size=self.target_size, mode=self.interpolate_mode)
#             x = rearrange(x, "(B v_q v_k h_k w_k) Head h_q w_q -> (B v_q v_k h_q w_q) Head h_k w_k", B=B, v_q=v_q, v_k=v_k, h_k=h_k, w_k=w_k, h_q=self.target_size, w_q=self.target_size) # (B v1 v2 h_q w_q) Head h_k w_k
#             x = nn.functional.interpolate(x, size=self.target_size, mode=self.interpolate_mode)
#             x = rearrange(x, "(B v_q v_k h_q w_q) Head h_k w_k -> B Head (v_q h_q w_q) (v_k h_k w_k)", B=B, v_q=v_q, v_k=v_k, h_k=self.target_size, w_k=self.target_size, h_q=self.target_size, w_q=self.target_size) # B Head Q K
#         # conv4d
#         for layer in self.net:
#             if isinstance(layer, Conv4D):
#                 x = layer(x, num_view=num_view, num_view_query=num_view_query)
#             else:
#                 x = layer(x)
#         # softmax
#         x = super().forward(x, num_view)
#         return x
