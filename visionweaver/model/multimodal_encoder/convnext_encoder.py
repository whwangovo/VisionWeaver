import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import BaseVisionTower
from .vision_models.convnext import convnext_xxlarge
from .utils import load_clip_image_processor, log_already_loaded, require_config_value


class ConvNextVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.select_layer = args.mm_vision_select_layer

        self.input_image_size = args.input_image_size

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return
        
        self.image_processor = load_clip_image_processor(self.args)
        self.vision_tower = convnext_xxlarge(self.vision_tower_name)

        self._freeze_if_needed()

        for s in self.vision_tower.stages:
            s.grad_checkpointing = True

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs[self.select_layer]
        
        return image_features

    def forward_features(self, x):
        x = self.vision_tower.stem(x)
        image_forward_out = []
        for blk in self.vision_tower.stages:
            x = blk(x)
            b, c, h, w = x.shape
            image_forward_out.append(x.view(b, c, -1).transpose(1, 2))
            
        return image_forward_out

    def forward(self, images):

        image_forward_outs = self.forward_features(images.to(device=self.device, dtype=self.dtype))
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def config(self):
        raise NotImplementedError("ConvNeXt config property not implemented")

    @property
    def num_attention_heads(self):
        return require_config_value(self.args, "convnext_num_attention_heads")
    
    @property
    def num_layers(self):
        return require_config_value(self.args, "convnext_num_layers")
    
    @property
    def hidden_size(self):
        hidden_size_map = getattr(self.args, "convnext_hidden_size_map", None)
        if not hidden_size_map:
            raise ValueError("convnext_hidden_size_map must be set in the config.")
        if self.select_layer in hidden_size_map:
            return hidden_size_map[self.select_layer]
        key = str(self.select_layer)
        if key in hidden_size_map:
            return hidden_size_map[key]
        raise ValueError(f"Missing hidden size for select_layer={self.select_layer}.")

    @property
    def num_patches(self):
        raise NotImplementedError("ConvNeXt num_patches not implemented")
