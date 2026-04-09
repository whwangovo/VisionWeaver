from transformers import SamModel

from .base_encoder import BaseVisionTower
from .utils import (
    forward_patch_embeddings,
    forward_sam_vision_encoder,
    load_clip_image_processor,
    log_already_loaded,
    require_config_value,
)


class SAMVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args):
        super().__init__(vision_tower, args)

        self.input_image_size = args.input_image_size
        self.pixel_shuffle = args.add_pixel_shuffle

        self.load_model()

    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return

        self.image_processor = load_clip_image_processor(self.args)
        self.vision_tower = SamModel.from_pretrained(self.vision_tower_name).vision_encoder
        self.sam_model_config = self.vision_tower.config

        cls_ = self.vision_tower.patch_embed
        bound_method = forward_patch_embeddings.__get__(cls_, cls_.__class__)
        setattr(cls_, 'forward', bound_method)

        cls_ = self.vision_tower
        bound_method = forward_sam_vision_encoder.__get__(cls_, cls_.__class__)
        setattr(cls_, 'forward', bound_method)

        self._freeze_if_needed()

        self.is_loaded = True

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).flatten(1, 2)

        return image_features

    @property
    def config(self):
        return self.sam_model_config

    @property
    def hidden_size(self):
        hidden_size = getattr(self.sam_model_config, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size
        return require_config_value(self.args, "sam_hidden_size")

    @property
    def num_patches(self):
        return self.config.num_patches
