from abc import abstractmethod

import torch
import torch.nn as nn


class BaseVisionTower(nn.Module):
    """Base class for all vision tower encoders in VisionWeaver."""

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.args = args
        self.vision_tower_name = vision_tower
        self.freeze_vision = args.freeze_vision_tower

    @abstractmethod
    def load_model(self, **kwargs):
        ...

    @abstractmethod
    def forward(self, images, **kwargs):
        ...

    def _freeze_if_needed(self):
        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

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
    @abstractmethod
    def config(self):
        ...

    @property
    @abstractmethod
    def hidden_size(self):
        ...
