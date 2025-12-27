import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_models.convnext import convnext_xxlarge
from .utils import load_clip_image_processor, log_already_loaded, require_config_value


class ConvNextVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args = args
        
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.freeze_vision = args.freeze_vision_tower

        self.input_image_size = args.input_image_size

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return
        
        # self.image_processor = CLIPImageProcessor(
        #     crop_size={
        #         "height": self.input_image_size, 
        #         "width": self.input_image_size
        #         },
        #     size={
        #         'shortest_edge': self.input_image_size
        #         },
        #     image_mean=[0.48145466, 0.4578275, 0.40821073],
        #     image_std=[0.26862954, 0.26130258, 0.27577711],)
        self.image_processor = load_clip_image_processor(self.args)
        self.vision_tower = convnext_xxlarge(self.vision_tower_name)
        # self.vision_tower  = timm.create_model(self.vision_tower_name, pretrained=True)

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

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

        # print("image_features: ", image_features.shape)
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
        assert  NotImplementedError
        pass

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
        return (cfg['image_size'] // self.patch_embed.patch_size[0]) ** 2
