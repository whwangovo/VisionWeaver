import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from .vision_models.convnext import convnext_xxlarge


class ConvNextVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.freeze_vision = getattr(args, 'freeze_vision_tower', True)

        self.input_image_size = getattr(args, 'input_image_size', 336)

        if not delay_load:
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
        #     image_mean=[0.48145466, 0.4578275, 0.40821073],
        #     image_std=[0.26862954, 0.26130258, 0.27577711],)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
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
        # as constant
        return 16
    
    @property
    def num_layers(self):
        # as constant
        return 4
    
    @property
    def hidden_size(self):
        if self.select_layer == -2:
            return 1536
        else:
            # -1 
            return 3072

    @property
    def num_patches(self):
        return (cfg['image_size'] // self.patch_embed.patch_size[0]) ** 2
