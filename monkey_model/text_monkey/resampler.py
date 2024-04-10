from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
import torch.nn as nn
import torch
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
#Resample model
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
from monkey_model.text_monkey.merge import *


class FeedForward(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU()
        
        self.fc2 =  nn.Linear(hidden_features, out_features, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x) 
        x = self.fc2(x)
        x = self.scale*x
        return x




class Block(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)


    def forward(self, x):
        x = self.fc_1(x)
        x = self.norm(x)
        return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        in_dim=1024, 
        out_dim=4096,
        depth=1,
        dim_head=128,
        heads=8,
        visual_tokens_num=512,
        ff_mult=4,
    ):
        super().__init__()

        self.downsample = nn.Linear(out_dim,in_dim,bias=False)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=in_dim, dim_head=dim_head, heads=heads),
                        FeedForward(in_features=in_dim, hidden_features=in_dim,out_features=out_dim),
                    ]
                )
            )

    def forward(self, x,r=0):
        B,L,C = x.shape

        merge = self_soft_matching(x, r)  # Replace with your features and r value
        latents = merge(x)        
        down_x = self.downsample(x)
        down_latent = self.downsample(latents)
        for attn, ff in self.layers:
            down_latent = attn(down_x, down_latent) 
            latents = ff(down_latent) + latents
        return latents
    
