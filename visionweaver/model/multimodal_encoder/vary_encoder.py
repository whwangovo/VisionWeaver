import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPImageProcessor, SamModel, SamProcessor, SamVisionConfig

from .vision_models.vary import build_vary


def forward_vision_encoder(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    b, h, w, c = x.shape
    x = x.reshape(b, -1, c)
    
    if self.pos_embed is not None:
        pos_embed = resample_pos_embed(self.pos_embed.flatten(1, 2), x.shape[1], num_prefix_tokens=0)
        x = x + pos_embed
    x = x.reshape(b, h, w, c)

    for blk in self.blocks:
        x = blk(x)

    # print("blk", x.shape)
    x = self.neck(x.permute(0, 3, 1, 2))
    # print("neck", x.shape)
    x = self.net_2(x)
    # print("net_2", x.shape)
    x = self.net_3(x)
    # print("net_3", x.shape)
    
    return x

def resample_pos_embed(
        posemb,
        new_size: int,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    new_size = [int(math.sqrt(new_size - num_prefix_tokens)), int(math.sqrt(new_size - num_prefix_tokens))]
    num_pos_tokens = posemb.shape[1] - num_prefix_tokens
    old_size = int(math.sqrt(num_pos_tokens))
    bs = posemb.shape[0]

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(bs, old_size, old_size, -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(bs, -1, embed_dim)
    posemb = posemb.to(dtype=orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], 1)

    if not torch.jit.is_scripting() and verbose:
        print(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb


class VaryVisionTower(nn.Module):
    def __init__(self, vision_tower, args):
        super().__init__()

        self.is_loaded = False

        self.args = args
        self.vision_tower_name = vision_tower
        self.freeze_vision = args.freeze_vision_tower
        
        self.input_image_size = args.input_image_size

        self.load_model()

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        image_processor_name = getattr(self.args, "vision_image_processor", None)
        if not image_processor_name:
            raise ValueError("vision_image_processor must be set in the config.")
        self.image_processor = CLIPImageProcessor.from_pretrained(image_processor_name)
        self.vision_tower = build_vary(self.vision_tower_name)

        cls_ = self.vision_tower
        bound_method = forward_vision_encoder.__get__(cls_, cls_.__class__)
        setattr(cls_, 'forward', bound_method)
        
        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.interpolate_pos_encoding(image_features)

        return image_features
    
    def interpolate_pos_encoding(self, image_features):
        if len(image_features.shape) == 3:
            b, n, c = image_features.shape
            w = h = int(n ** 0.5)
            image_features = image_features.transpose(1, 2).reshape(b, c, h, w)
        else:
            b, c, h, w = image_features.shape

        # if w != self.input_image_tokens:
        #     image_features = F.interpolate(image_features.float(), size=(self.num_grids, self.num_grids), mode='bilinear', align_corners=True)
            
        return image_features.flatten(2, 3).transpose(1, 2)
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        raise NotImplementedError

    @property
    def hidden_size(self):
        hidden_size = getattr(self.args, "vary_hidden_size", None)
        if hidden_size is None:
            raise ValueError("vary_hidden_size must be set in the config.")
        return hidden_size

    @property
    def num_patches(self):
        return self.config.num_patches
