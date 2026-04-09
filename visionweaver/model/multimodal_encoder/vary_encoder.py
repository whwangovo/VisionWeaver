import os

from .base_encoder import BaseVisionTower
from .hf_utils import resolve_checkpoint_path
from .utils import (
    forward_vary_vision_encoder,
    interpolate_pos_encoding,
    load_clip_image_processor,
    log_already_loaded,
)
from .vision_models.vary import build_vary


class VaryVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args):
        super().__init__(vision_tower, args)

        self.input_image_size = args.input_image_size

        self.load_model()

    def load_model(self):
        if self.is_loaded:
            log_already_loaded(self.vision_tower_name)
            return

        self.image_processor = load_clip_image_processor(self.args)
        checkpoint_filename = getattr(self.args, "checkpoint_filename", None)
        checkpoint_path = resolve_checkpoint_path(
            self.vision_tower_name, filename=checkpoint_filename
        )
        if checkpoint_path and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Vary checkpoint not found: {checkpoint_path}"
            )
        self.vision_tower = build_vary(checkpoint_path)

        cls_ = self.vision_tower
        bound_method = forward_vary_vision_encoder.__get__(cls_, cls_.__class__)
        setattr(cls_, 'forward', bound_method)

        self._freeze_if_needed()

        self.is_loaded = True

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_feature = self.vision_tower(image.to(device=self.device).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = interpolate_pos_encoding(image_features)

        return image_features

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
