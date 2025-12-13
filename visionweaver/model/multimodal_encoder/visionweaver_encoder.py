import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.utils import logging

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.utils import rank0_print

from .visionweaver_experts import VisionExperts

logger = logging.get_logger(__name__)


class VisionTower(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_vision_tower = CLIPVisionTower(
            "openai/clip-vit-large-patch14-336", config
        )

        experts_config = copy.deepcopy(config)
        experts_config.mm_hidden_size = self.hidden_size
        self.vision_experts = VisionExperts(experts_config)
        self.vision_router = VisionRouter(experts_config)

    def forward(self, pixel_values, input_embeds=None):

        base_features = self.base_vision_tower(pixel_values)

        routing_weights, selected_experts = self.vision_router(base_features)

        expert_features = self.vision_experts(
            pixel_values, selected_experts, routing_weights
        )
        results = expert_features + base_features[:, 1:, :]

        return results

    @property
    def hidden_size(self):
        return self.base_vision_tower.hidden_size

    @property
    def image_processor(self):
        return self.base_vision_tower.image_processor

    @property
    def is_loaded(self):
        return self.base_vision_tower.is_loaded


class VisionRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = getattr(config, "router_top_k", config.num_experts)
        self.selector = nn.Linear(config.mm_hidden_size, config.num_experts)

    def forward(self, vision_features):
        router_logits = self.selector(vision_features[:, 0, :])

        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=1).to(router_logits.dtype)

        return routing_weights, selected_experts
