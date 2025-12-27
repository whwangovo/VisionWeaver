import os
import warnings
from collections import OrderedDict

import torch
from torch import nn
from visionweaver.model.multimodal_encoder.hf_utils import resolve_checkpoint_path
from visionweaver.model.multimodal_encoder.utils import (
    load_clip_image_processor,
    log_already_loaded,
)
from visionweaver.model.multimodal_encoder.vision_models.eva_vit_builder import (
    build_eva_vit,
)


class EVAVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.args = args
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.freeze_vision = args.freeze_vision_tower

        self.input_image_size = args.input_image_size
        self.input_image_tokens = (
            self.input_image_size // args.patch_size
        ) ** 2
        # self.vision_tower.config = vision_tower_config

        self.load_model()


    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return

        # # hardcode
        # self.image_processor = CLIPImageProcessor(crop_size=
        #                                           {"height": self.input_image_size, 
        #                                            "width": self.input_image_size
        #                                            },
        #                                     size={'shortest_edge': self.input_image_size},
        #                                     image_mean=[0.48145466, 0.4578275, 0.40821073],
        #                                     image_std=[0.26862954, 0.26130258, 0.27577711])

        # load weights
        self.image_processor = load_clip_image_processor(self.args)
        self.vision_tower, self.config = build_eva_vit(model_name=self.vision_tower_name, image_size=self.input_image_size)
        self.load_vision_checkpoint()
        # self.vision_tower.config = vision_tower_config

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def load_vision_checkpoint(self):
        checkpoint_filename = getattr(self.args, "checkpoint_filename", None)
        checkpoint_path = resolve_checkpoint_path(
            self.vision_tower_name, filename=checkpoint_filename
        )
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            warnings.warn(
                "The vision tower weights for EVA-02 do not exist; training from "
                "scratch will likely fail."
            )
            self.is_loaded = True
            return 
        
        pretrained_params = torch.load(checkpoint_path, weights_only=True)
        if 'ema_state' in pretrained_params:
            pretrained_params = pretrained_params['ema_state']
        elif 'module' in pretrained_params:
            pretrained_params = pretrained_params['module']

        new_params = OrderedDict()
        
        kw = ""
        if "det" in self.vision_tower_name.lower():
            kw = "backbone.net."
        elif "clip" in self.vision_tower_name.lower():
            kw = "visual."

        for k, v in pretrained_params.items():
            if len(kw) > 0:
                if kw in k and ("rope" not in k):
                    new_params[k.replace(kw, "")] = v
            else:
                if "rope" not in k:
                    new_params[k] = v

        incompatiblekeys = self.vision_tower.load_state_dict(new_params, strict=False)    
        for k in incompatiblekeys[0]:
            if "rope" not in k:
                warnings.warn(f"Find incompatible keys {k} in state dict.")

    # @torch.no_grad()
    def forward(self, images):
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

    # @property
    # def config(self):
    #     return self.vision_tower.config

    @property
    def hidden_size(self) -> int:
        #return self.config.hidden_size
        return self.config['hidden_dim'] 

    @property
    def num_patches(self):
        # return (self.config.image_size // self.config.patch_size) ** 2
        return self.config['num_patches']
