from __future__ import annotations

import os
from functools import partial

import torch

from .vary_encoder import ImageEncoderViT


def build_vary(checkpoint: str | None = None) -> ImageEncoderViT:
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: list[int],
    checkpoint: str | None = None,
) -> ImageEncoderViT:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    if checkpoint is None:
        return image_encoder

    checkpoint_path = (
        os.path.join(checkpoint, "pytorch_model.bin")
        if os.path.isdir(checkpoint)
        else checkpoint
    )
    state_dict = torch.load(checkpoint_path, weights_only=True)

    def get_w(weights, keyword):
        return {
            k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k
        }

    image_encoder.load_state_dict(get_w(state_dict, "vision_tower_high"), strict=True)
    return image_encoder
