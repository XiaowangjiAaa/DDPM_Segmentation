import torch as th
from torch import nn
import torch.nn.functional as F

from models.nn import (
    conv_nd, linear, normalization, zero_module, timestep_embedding
)
from models.blocks import ResBlock, AttentionBlock, Upsample, Downsample, TimestepEmbedSequential
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .generic_unet import Generic_UNet

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if hasattr(layer, 'forward') and 'emb' in layer.forward.__code__.co_varnames:
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class UNetModel_newpreview(nn.Module):
    def __init__(
        self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
        attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2,
        num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1,
        num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False,
        use_new_attention_order=False, high_way=True
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.model_channels = model_channels
        self.in_channels = in_channels
        self.dtype = th.float16 if use_fp16 else th.float32
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block = ResBlock(
                    ch, time_embed_dim, dropout, out_channels=mult * model_channels,
                    dims=dims, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm
                )
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(block))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                down = Downsample(ch, conv_resample, dims)
                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims),
            AttentionBlock(ch, num_heads, num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, dims=dims),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=model_channels * mult, dims=dims
                    )
                ]
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    up = Upsample(ch, conv_resample, dims)
                    layers.append(up)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # anchor 分支
        if high_way:
            self.hwm = Generic_UNet(
                self.in_channels - 1, 32, 1, 5, anchor_out=True, upscale_logits=True
            )
            # 32 + 32 + 64 = 128 → 映射回 model_channels
            self.anchor_adapter = conv_nd(2, 128, model_channels, kernel_size=1)

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if hasattr(self, 'label_emb') and y is not None:
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        c = h[:, :-1, ...]
        anch, cal = self.hwm(c)

        hs = []
        for ind, module in enumerate(self.input_blocks):
            if ind == 0:
                h = module(h, emb)

                # 上采样 + 映射通道后加到主干
                anch1_up = F.interpolate(anch[1], size=anch[0].shape[2:], mode="bilinear", align_corners=False)
                anch_fused = th.cat((anch[0], anch[0], anch1_up), dim=1)
                anch_fused = self.anchor_adapter(anch_fused)
                h = h + anch_fused.detach()
            else:
                h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        out = self.out(h)
        return out, cal
