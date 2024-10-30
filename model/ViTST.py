import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViTST(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, bias = False, device = None, dtype = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.expand = 2
        self.nframes = num_frames    
        self.dim = dim
        self.d_inner = int(self.expand * self.dim)
        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias, **factory_kwargs)

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.space_transformer = Transformer(self.d_inner * 2, depth, heads, dim_head, dim*scale_dim, dropout)
        self.temporal_transformer = Transformer(self.d_inner * 2, depth, heads, dim_head, dim*scale_dim, dropout)

        self.out_proj = nn.Linear(self.d_inner*2, self.dim, bias=bias, **factory_kwargs)

    def forward(self, x):
        b, n, dim = x.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b l d",
            l=n,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        out = self.temporal_transformer(xz)
        out_b = self.temporal_transformer(xz.flip([-1]))

        xz_s = xz.chunk(self.nframes, dim=-1)
        xz_s = torch.stack(xz_s,dim=-1)
        xz_s = xz_s.flatten(-2)

        out_s = self.space_transformer(xz_s)

        out = F.linear((out + out_b.flip([-1]) + out_s) / 3, self.out_proj.weight, self.out_proj.bias)
        return out