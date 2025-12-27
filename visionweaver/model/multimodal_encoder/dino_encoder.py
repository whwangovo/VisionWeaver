import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, Dinov2Model

from .utils import load_clip_image_processor, log_already_loaded, require_config_value


class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()


        self.is_loaded = False
        self.args = args

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = args.mm_vision_select_feature
        self.freeze_vision = args.freeze_vision_tower

        self.input_image_size = args.input_image_size

        #self.load_model()

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return
        
        self.image_processor = load_clip_image_processor(self.args)
        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # image_forward_outs_1 = self.vision_tower_1.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return self.config.hidden_size

    @property
    def num_patches(self):
        image_size = getattr(self.config, "image_size", None)
        patch_size = getattr(self.config, "patch_size", None)
        if image_size and patch_size:
            return (image_size // patch_size) ** 2
        return require_config_value(self.args, "dino_num_patches")
