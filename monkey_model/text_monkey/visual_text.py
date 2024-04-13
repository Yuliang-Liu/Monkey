# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List
import numpy as np
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from monkey_model.text_monkey.window import CrossWindowAttention,PatchMerging
from monkey_model.text_monkey.resampler import *
import random
import ipdb
import numpy as np
import matplotlib.pyplot as plt


def reconstruct_matrix(windows):
    temp =[]
    for col in windows:
        temp.append(torch.cat((col),dim=3))
    all_img = torch.cat(temp,dim=2)
    return all_img


def sliding_window(matrix, window_size, stride):
    b,c,height, width = matrix.shape
    window_rows = math.ceil((height - window_size[0]) / stride) + 1
    window_cols = math.ceil((width - window_size[1]) / stride) + 1
    #windows = np.zeros((window_rows, window_cols, window_size[0], window_size[1]))
    windows = []
    for i in range(window_rows):
        windows_col = []
        for j in range(window_cols):
            window = matrix[:,:, i*stride:i*stride+window_size[0],  j*stride:j*stride+window_size[1]]
            windows_col.append(window)
        windows.append(windows_col)
    return windows
def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)



class Lora_Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 out_feat=None,
                 r=16,
                 dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.out_feat = out_feat
        self.r = r

        self.lora_scale = nn.Parameter(torch.ones(1))


        self.lora_a = nn.Linear(self.d_model, self.r,bias=False)
        self.lora_b = nn.Linear(self.r, self.out_feat,bias=False)

        self.lora_dropout =  nn.Dropout(p=dropout)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

    def forward(self, x ):
        #residual = x if residual is None else residual

        x = self.lora_dropout(x)
        down = self.lora_a(x)
        up = self.lora_b(down)

        up = up * self.lora_scale
        output = up

        return output


class VisualAttention(nn.Module):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads,
                 bias=True, kdim=None, vdim=None,lora_repeat_num=4):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, 'Only Support SelfAttention Currently'
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.in_proj_lora = []
        for _ in range(lora_repeat_num):
            self.in_proj_lora.append(Lora_Adapter(d_model=embed_dim,out_feat=3 * embed_dim))
        self.in_proj_lora = nn.ModuleList(self.in_proj_lora)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj_lora = []
        for _ in range(lora_repeat_num):
            self.out_proj_lora.append(Lora_Adapter(d_model=embed_dim,out_feat=embed_dim))
        self.out_proj_lora = nn.ModuleList(self.out_proj_lora)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query, key, value, attn_mask = None,lora_idx = None):
        # query/key/value: [sq, b, h]
        sq, b, _ = query.size()

        assert query is key, 'Only Support Self-Attention Currently'
        sk = sq
        mixed_x_layer = self.in_proj(query)
        if lora_idx == None:
            pass
        else:
            lora_res = self.in_proj_lora[lora_idx](query)
            mixed_x_layer += lora_res

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            self.hidden_size_per_attention_head, dim=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(sq,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(sk,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            attention_probs = torch.baddbmm(attn_mask, q_scaled, key_layer.transpose(-2, -1))
        else:
            attention_probs = torch.bmm(q_scaled, key_layer.transpose(-2, -1))
        attention_probs = attention_probs.softmax(dim=-1)

        value_layer = value_layer.view(sk,
            b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(b,
            self.num_attention_heads_per_partition,
            sq, self.hidden_size_per_attention_head)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.out_proj(context_layer)
        if lora_idx == None:
            pass
        else:
            lora_res = self.out_proj_lora[lora_idx](context_layer)
            output += lora_res

        return output


class VisualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            is_cross_attention: bool = False,
            lora_repeat_num = 4,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head,lora_repeat_num = lora_repeat_num)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.mlp_lora = []
        for _ in range(lora_repeat_num):
            self.mlp_lora.append(Lora_Adapter(d_model=d_model,out_feat=d_model,r=32))
        self.mlp_lora = nn.ModuleList(self.mlp_lora)


    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            lora_idx = None
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask,lora_idx=lora_idx)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            lora_idx = None
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask,lora_idx=lora_idx)
        residual = x 
        x = x + self.mlp(self.ln_2(x))

        
        if lora_idx == None:
            pass
        else:
            x += self.mlp_lora[lora_idx](residual)
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            lora_repeat_num=4,
            add_window=False,
            window_all=False,
            image_size=(896,896)
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.add_window = add_window
        self.window_all = window_all

        self.window_pos = [2,6,24,46]
        self.window_dim = [128,256,512,1024]
        self.window_head = [4,8,16,32]
        if isinstance(image_size, tuple) or isinstance(image_size, list):
            image_size = tuple(size // 14 for size in image_size)
        else:
            image_size = image_size//14

        if self.add_window:
            self.window_attention = []
            for idx in range(len(self.window_pos)):
                self.window_attention.append(CrossWindowAttention(image_size=image_size,dim=1664,hidden_dim=self.window_dim[idx],head=self.window_head[idx]))
            self.window_attention = nn.ModuleList(self.window_attention)
   
        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer,lora_repeat_num=lora_repeat_num)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self, x,attn_mask: Optional[torch.Tensor] = None,lora_idx=None,image_size=(64,64)):
        
        if isinstance(x,List):
            window_idx = 0
            for r_idx,r in enumerate(self.resblocks):
                if self.add_window:
                    if r_idx in self.window_pos:
                        for i in range(len(x)):
                            for j in range(len(x[i])):
                                x[i][j] = x[i][j].permute(1, 0, 2) # LND -> NLD
                                x[i][j] = x[i][j].permute(0, 2, 1)  # shape = [*, width, grid ** 2,]
                                x[i][j] = x[i][j].reshape(x[i][j].shape[0], x[i][j].shape[1], 32,32)
                        whole_image = reconstruct_matrix(x) #shape = [*,width,grid,grid]
                        whole_image = self.window_attention[window_idx](whole_image,image_size)
                        x = sliding_window(whole_image,(32,32),32)
                        for i in range(len(x)):
                            for j in range(len(x[i])):
                                x[i][j] = x[i][j].reshape(x[i][j].shape[0], x[i][j].shape[1], -1)
                                x[i][j] = x[i][j].permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                                x[i][j] = x[i][j].permute(1, 0, 2) # NLD -> LND
                        window_idx  += 1
                            
                if lora_idx is None or lora_idx == 0 :      
                    for i in range(len(x)):
                        for j in range(len(x[i])):
                            x[i][j] = r(x[i][j],attn_mask=attn_mask,lora_idx=lora_idx)
                else:
                    temp_lora_idx = 0
                    for i in range(len(x)):
                        for j in range(len(x[i])):
                            x[i][j] = r(x[i][j],attn_mask=attn_mask,lora_idx=temp_lora_idx)
                            temp_lora_idx += 1       
            return x
        else:
            for r in self.resblocks:
                x = r(x, attn_mask=attn_mask)
            return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            image_size,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            n_queries: int = 256,
            output_dim: int = 512,
            lora_repeat_num: int = 0,
            add_window: bool = False,
            use_global:bool =False,
            resampler=False,
            r=512,
            **kwargs
    ):
        super().__init__()
        if isinstance(image_size, tuple) or isinstance(image_size, list):
            image_height, image_width = self.image_size = image_size
        else:
            image_height, image_width = self.image_size = (image_size,image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim
        self.add_window = add_window
        self.use_global = use_global
        self.resampler = resampler
        self.r = r
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        self.image_transform = transforms.Compose([
            transforms.Resize(
               self.image_size,
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.lora_repeat_num = lora_repeat_num
        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(
            width,
            layers,
            heads,
            mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            lora_repeat_num=lora_repeat_num,
            add_window=add_window,
            image_size=image_size
        )

        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(256)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
        )

        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter((output_dim** -0.5) * torch.randn(output_dim, output_dim))

        if self.resampler:
            self.downresampler = PerceiverResampler()

    def forward(self, x: torch.Tensor,lora_idx=None,add_window=False):

        x = x.to(
            dtype=self.transformer.get_cast_dtype(),
            device=self.transformer.get_cast_device(),
        )
        
        # to patches
        x = self.conv1(x)  # shape = [b, width, grid, grid]
        b,c,h,w = x.shape
        if add_window:
            x = sliding_window(x,(32,32),32)
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] = x[i][j].reshape(x[i][j].shape[0], x[i][j].shape[1], -1)
                    x[i][j] = x[i][j].permute(0, 2, 1)  # shape = [*, grid ** 2, width]

                    x[i][j] = x[i][j] + get_abs_pos(self.positional_embedding,x[i][j].size(1))

                    x[i][j] = self.ln_pre(x[i][j])
                    x[i][j] = x[i][j].permute(1, 0, 2) # NLD -> LND
        else:
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = x + get_abs_pos(self.positional_embedding, x.size(1))
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
        
        x = self.transformer(x,lora_idx=lora_idx,image_size=(h,w))
        
        if add_window:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] = x[i][j].permute(1, 0, 2)  # LND -> NLD
                    x[i][j] = self.attn_pool(x[i][j])
                    x[i][j] = self.ln_post(x[i][j])
                    x[i][j] = x[i][j] @ self.proj
            temp =[]
            for col in x:
                temp.append(torch.cat((col),dim=1))
            x = torch.cat(temp,dim=1)     
        else:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.attn_pool(x)
            x = self.ln_post(x)
            x = x @ self.proj
        return x


    def encode(self, image_paths: List[str],lora_idx=None,input_image=None):
        if input_image is None:
            images = []
            for image_path in image_paths:
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    image = Image.open(image_path)
                image = image.convert("RGB")
                ## to imitate transmission loss in the real world.
                if self.training:
                    output = BytesIO()
                    qual =  random.randint(20, 100)
                    image.save(output, format='JPEG', quality=qual)
                    image_data = output.getvalue()
                    image =Image.open(BytesIO(image_data))
            
                images.append(self.image_transform(image))
            images = torch.stack(images, dim=0)
        else:
            images = input_image
        images_448 = F.interpolate(images, size=(448,448), mode='bicubic')
        
        if lora_idx == 1:
            local_feat = self(images,0,add_window=True)
        else:
            local_feat = self(images,lora_idx,add_window=True)
        
        if self.resampler:
            local_feat = self.downresampler(local_feat,r = self.r)

        if self.use_global:
            global_feat = self(images_448,lora_idx=None,add_window=False)
            return torch.cat([local_feat,global_feat],dim=1)
        else:
            return local_feat




