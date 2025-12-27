from __future__ import annotations

from .eva_vit import EVAViT


def build_eva_vit(model_name=None, image_size=224, window_attn=True):
    if window_attn:
        window_block_indexes = (
            list(range(0, 2))
            + list(range(3, 5))
            + list(range(6, 8))
            + list(range(9, 11))
            + list(range(12, 14))
            + list(range(15, 17))
            + list(range(18, 20))
            + list(range(21, 23))
        )
    else:
        window_block_indexes = ()

    model = EVAViT(
        img_size=image_size,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        window_block_indexes=window_block_indexes,
        qkv_bias=True,
        drop_path_rate=0.0,
        xattn=False,
    )

    image_size = 224  # HARDCODE
    eva_config = dict(
        image_size=image_size,
        patch_size=16,
        window_size=16,
        hidden_dim=1024,
        depth=24,
        num_heads=16,
        window_block_indexes=window_block_indexes,
        num_patches=image_size**2 // 16**2,
        pretrained_from=model_name,
    )

    return model, eva_config
