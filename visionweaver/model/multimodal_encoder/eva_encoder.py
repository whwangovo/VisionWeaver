import os
import warnings
from collections import OrderedDict

import torch
from visionweaver.model.multimodal_encoder.hf_utils import resolve_checkpoint_path
from visionweaver.model.multimodal_encoder.utils import (
    interpolate_pos_encoding,
    load_clip_image_processor,
    log_already_loaded,
)
from visionweaver.model.multimodal_encoder.vision_models.eva_vit_builder import (
    build_eva_vit,
)
from .base_encoder import BaseVisionTower


class EVAVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.select_layer = args.mm_vision_select_layer

        self.input_image_size = args.input_image_size
        self.input_image_tokens = (
            self.input_image_size // args.patch_size
        ) ** 2
        self.load_model()


    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return

        # load weights
        self.image_processor = load_clip_image_processor(self.args)
        self.vision_tower, self.config = build_eva_vit(model_name=self.vision_tower_name, image_size=self.input_image_size)
        self.load_vision_checkpoint()

        self._freeze_if_needed()

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

    def forward(self, images):
        image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
        image_features = interpolate_pos_encoding(image_features)

        return image_features

    @property
    def hidden_size(self) -> int:
        return self.config['hidden_dim']

    @property
    def num_patches(self):
        return self.config['num_patches']
