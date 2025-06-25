# models/nn.py
# 模型所需基础组件：attention、normalization、conv_nd、embedding等

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 支持不同维度（1D/2D/3D）的卷积
def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

# 零初始化模块
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# 时间步嵌入（Sinusoidal Embedding）
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(0, half, dtype=th.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding

# 标准归一化层
def normalization(channels):
    return nn.GroupNorm(32, channels)

def layer_norm(shape):
    return nn.LayerNorm(shape)

# models/fp16_util.py
# float16 模型转换工具

def convert_module_to_f16(l):
    if hasattr(l, 'to'): l.to(dtype=th.float16)

def convert_module_to_f32(l):
    if hasattr(l, 'to'): l.to(dtype=th.float32)

def avg_pool_nd(dims, kernel_size, stride=None):
    if dims == 1:
        return nn.AvgPool1d(kernel_size, stride=stride)
    elif dims == 2:
        return nn.AvgPool2d(kernel_size, stride=stride)
    elif dims == 3:
        return nn.AvgPool3d(kernel_size, stride=stride)
    else:
        raise NotImplementedError(f"avg_pool_nd not implemented for dims={dims}")
    
class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)