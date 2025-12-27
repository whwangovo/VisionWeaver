import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPImageProcessor, SamModel, SamProcessor, SamVisionConfig


def forward_patch_embeddings(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # sam does not have positional coding and can change the input resolution at will
        # if height != self.image_size[0] or width != self.image_size[1]:
            # raise ValueError(
            #     f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            # )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings
    
def forward_vision_encoder(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden_states = self.patch_embed(pixel_values)
        
        b, h, w, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, -1, c)
        if self.pos_embed is not None:
            pos_embed = resample_pos_embed(self.pos_embed.flatten(1, 2), hidden_states.shape[1], num_prefix_tokens=0)
            hidden_states = hidden_states + pos_embed

        hidden_states = hidden_states.reshape(b, h, w, c)
        # print("input_hidden_states:", hidden_states.shape)
        
        for i, layer_module in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]
            
        # print("output_hidden_states:", hidden_states.shape)
        
        # hidden_states = self.neck(hidden_states)
        
        # print("neck_hidden_states:", hidden_states.shape)

        return hidden_states

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


class SAMVisionTower(nn.Module):
    def __init__(self, vision_tower, args):
        super().__init__()

        self.is_loaded = False

        self.args = args
        self.vision_tower_name = vision_tower
        self.freeze_vision = args.freeze_vision_tower
        
        self.input_image_size = args.input_image_size
        self.pixel_shuffle = args.add_pixel_shuffle

        self.load_model()

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        # self.image_processor = CLIPImageProcessor(
        #     crop_size={
        #         "height": self.input_image_size, 
        #         "width": self.input_image_size
        #         },
        #     size={
        #         'shortest_edge': self.input_image_size
        #         },
        #     image_mean=[0.485, 0.456, 0.406],
        #     image_std=[0.229, 0.224, 0.225])
        # self.image_processor = SamProcessor.from_pretrained(self.vision_tower_name)
        # self.image_processor.preprocess = self.image_processor.__call__
        # self.image_processor.image_mean = [0.485, 0.456, 0.406]
        # self.image_processor.pad_size = {
        #     "height": self.input_image_size,
        #     "width": self.input_image_size,
        # }
        # self.image_processor.size = {
        #     "longest_edge": self.input_image_size,
        # }
        
        # print("input_image_size", input_image_size)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.vision_tower = SamModel.from_pretrained(self.vision_tower_name).vision_encoder
        # sam_model.neck = ShortSamVisionNeck(sam_model.config)
        self.sam_model_config = self.vision_tower.config
        
        cls_ = self.vision_tower.patch_embed
        bound_method = forward_patch_embeddings.__get__(cls_, cls_.__class__)
        setattr(cls_, 'forward', bound_method)
        
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
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).flatten(1, 2)
        
        # if self.pixel_shuffle:
        #     b, n, c = image_features.shape
        #     h = w = int(n ** 0.5)
        #     image_features = image_features.transpose(1,2).reshape(b, c, h, w) 
        #     image_features = nn.functional.pixel_unshuffle(image_features, 2)

        return image_features
    
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
        return self.sam_model_config

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return self.config.num_patches
