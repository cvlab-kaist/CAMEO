import torch.nn as nn
from einops import rearrange
'''
    x: (b, head, fhw, c)
'''

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

class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_ratio, mlp_depth, **kwargs):
        super(Mlp, self).__init__()
        self.net = build_mlp(in_dim, out_dim, mlp_ratio=mlp_ratio, mlp_depth=mlp_depth)

    def forward(self, x, **kwargs):
        # x: (b, head, fhw, c)
        b, head, fhw, c = x.shape
        x = rearrange(x, 'b head fhw c -> (b head fhw) c', )  # (b*head*fhw, c)
        x = self.net(x)  # (b*head*fhw, out_ch)
        x = rearrange(x, '(b head fhw) c2 -> b head fhw c2', b=b, head=head, fhw=fhw)  # (b, head, fhw, out_ch)
        return x

class DeepConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim, depth, kernel_size=3, stride=1, padding=1, use_batchnorm = False, last_activation = None,**kwargs):
        super(DeepConv2d, self).__init__()
        net = []
        if depth <= 0:
            net += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            net += [nn.Conv2d(in_dim, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding), nn.GELU()] if not use_batchnorm else [nn.Conv2d(in_dim, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(mid_dim), nn.GELU()]
            for _ in range(depth - 1):
                net += [nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding), nn.GELU()] if not use_batchnorm else [nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(mid_dim), nn.GELU()]
            net += [nn.Conv2d(mid_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)]
        if last_activation is not None:
            if last_activation.lower() == 'gelu':
                net += [nn.GELU()]
            elif last_activation.lower() == 'relu':
                net += [nn.ReLU()]
            else:
                raise ValueError(f"Unsupported last_activation: {last_activation}")
        self.use_batchnorm = use_batchnorm
        self.net = nn.Sequential(*net)

    def forward(self, x, num_view, **kwargs):
        # x: (b, head, fhw, c)
        b, head, fhw, c = x.shape
        hw = fhw // num_view
        h = w = int(hw ** 0.5)
        x = rearrange(x, 'b head (f h w) c -> (b head f) c h w', h=h, w=w, f=num_view)  # (b*head*v, c, h, w)
        x = self.net(x)  # (b*head*v, out_ch, h', w')
        x = rearrange(x, '(b head f) c2 h w -> b head (f h w) c2', b=b, head=head, f=num_view)  # (b, head, fhw', out_ch)
        return x
    
class Interpolate2d_DeepConv2d(DeepConv2d):
    def __init__(self, upsample_size, interpolate_mode='nearest' , **kwargs):
        super(Interpolate2d_DeepConv2d, self).__init__(**kwargs)
        if isinstance(upsample_size, int):
            upsample_size = (upsample_size, upsample_size)
        elif isinstance(upsample_size, tuple):
            pass
        else:
            raise ValueError(f"upsample_size should be int or tuple, but got {type(upsample_size)}")
        self.upsample_size = upsample_size
        self.interpolate_mode = interpolate_mode

    def forward(self, x, num_view, **kwargs):
        # x: (b, head, fhw, c)
        b, head, fhw, c = x.shape
        hw = fhw // num_view
        h = w = int(hw ** 0.5)
        x = rearrange(x, 'b head (f h w) c -> (b head f) c h w', h=h, w=w, f=num_view)  # (b*head*v, c, h, w)
        x = nn.functional.interpolate(x, size=self.upsample_size, mode=self.interpolate_mode)  # (b*head*v, c, h*2, w*2)
        x = self.net(x)  # (b*head*v, out_ch, h', w')
        x = rearrange(x, '(b head f) c2 h w -> b head (f h w) c2', b=b, head=head, f=num_view)  # (b, head, fhw', out_ch)
        return x
    
class N_Interpolate2d_DeepConv2d(nn.Module):
    def __init__(self, in_dim, block_num, block_kwargs_list,**kwargs):
        super(N_Interpolate2d_DeepConv2d, self).__init__()
        self.block_num = block_num
        self.block_kwargs_list = block_kwargs_list
        blocks = []
        for i in range(block_num):
            block_kwargs = block_kwargs_list[i]
            blocks.append(Interpolate2d_DeepConv2d(in_dim=in_dim, **block_kwargs_list[i]))
            in_dim = block_kwargs['out_dim']
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, num_view, **kwargs):
        for i in range(self.block_num):
            x = self.blocks[i](x, num_view=num_view)
        return x

FEAT_HEAD_CLS = {
    "identity": Identity,
    "mlp": Mlp,
    "deepconv2d": DeepConv2d,
    'interpolate2d_deepconv2d': Interpolate2d_DeepConv2d,
    'n_interpolate2d_deepconv2d': N_Interpolate2d_DeepConv2d,
}