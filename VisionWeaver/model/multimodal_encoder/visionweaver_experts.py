from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext_encoder import ConvNextVisionTower
from .dino_encoder import DINOVisionTower
from .eva_encoder import EVAVisionTower
from .pix2struct_encoder import Pix2StructVisionTower
from .sam_encoder import SAMVisionTower
from .vary_encoder import VaryVisionTower


class VisionExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.is_loaded = False
        self.input_image_size = getattr(config, "image_size", 336)
        self.patch_size = getattr(config, "patch_size", 14)
        self.num_grids = self.input_image_size // self.patch_size
        self.num_tokens = self.num_grids**2
        self.hidden_size = getattr(config, "mm_hidden_size", 1024)
        self.freeze_vision = getattr(config, "freeze_vision_tower", True)

        vision_tower_name_list = getattr(
            config, "mm_vision_tower", getattr(config, "vision_tower", None)
        )

        if isinstance(vision_tower_name_list, str):
            vision_tower_name_list = vision_tower_name_list.split(";")
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower_name_list}")

        self.load_experts_vision_towers(vision_tower_name_list, config)

    def load_experts_vision_towers(self, vision_tower_name_list, config):
        self.vision_experts = nn.ModuleList()
        self.expert_projectors = nn.ModuleList()

        expert_registry = {
            "eva": {
                "cls": EVAVisionTower,
                "path": "checkpoints/EVA-02/eva02/det/eva02_L_coco_det_sys_o365.pth",
                "args": {},
            },
            "convnext": {
                "cls": ConvNextVisionTower,
                "path": "convnext_xxlarge.clip_laion2b_soup",
                "args": {},
            },
            "sam": {
                "cls": SAMVisionTower,
                "path": "facebook/sam-vit-large",
                "args": {},
            },
            "pix2struct": {
                "cls": Pix2StructVisionTower,
                "path": "google/pix2struct-large",
                "args": {"do_resize": True, "de_normalize": True},
            },
            "vary": {
                "cls": VaryVisionTower,
                "path": "checkpoints/Vary-toy",
                "args": {},
            },
            "dino": {
                "cls": DINOVisionTower,
                "path": "facebook/dinov2-large",
                "args": {"mm_vision_select_layer": -2},
            },
        }

        for name in vision_tower_name_list:
            if name not in expert_registry:
                raise NotImplementedError(f"Expert {name} is not implemented.")

            expert_info = expert_registry[name]
            expert_args = deepcopy(config)
            expert_args.input_image_size = self.input_image_size
            expert_args.freeze_vision = self.freeze_vision

            for k, v in expert_info["args"].items():
                setattr(expert_args, k, v)

            vision_tower = expert_info["cls"](expert_info["path"], expert_args)
            expert_projector = nn.Linear(vision_tower.hidden_size, self.hidden_size)

            self.vision_experts.append(vision_tower)
            self.expert_projectors.append(expert_projector)

        self.is_loaded = True

    def load_model(self):
        assert (
            self.is_loaded
        ), "All the vision encoders should be loaded during initialization!"

    def forward(self, x, selected_experts, routing_weights):
        x_b = x.shape[0]

        experts_feature = torch.zeros(
            (x_b, self.num_tokens * self.hidden_size),
            device=routing_weights.device,
            dtype=routing_weights.dtype,
        )

        for i, (vision_experts, expert_projectors) in enumerate(
            zip(self.vision_experts, self.expert_projectors)
        ):
            batch_idx, expert_idx = torch.where(selected_experts == i)

            if len(batch_idx) == 0:
                continue

            if vision_experts.input_image_size != self.input_image_size:
                resized_x = F.interpolate(
                    x.float(),
                    size=(
                        vision_experts.input_image_size,
                        vision_experts.input_image_size,
                    ),
                    mode="bilinear",
                    align_corners=True,
                ).to(x.dtype)
            else:
                resized_x = x

            # Select relevant images
            current_x = resized_x[batch_idx]
            feature = vision_experts(current_x)

            # Standardize feature shape to (b, c, h, w)
            if feature.dim() == 3:
                b, n, c = feature.shape
                w = h = int(n**0.5)
                feature = feature.transpose(1, 2).reshape(b, c, h, w)
            else:
                b, c, h, w = feature.shape

            # Interpolate if grid size doesn't match
            if w != self.num_grids:
                feature = F.interpolate(
                    feature.float(),
                    size=(self.num_grids, self.num_grids),
                    mode="bilinear",
                    align_corners=True,
                ).to(dtype=x.dtype)

            # Project features
            feature = expert_projectors(feature.flatten(2).transpose(1, 2))
            feature = feature.reshape(b, -1)

            # Accumulate features
            weight = routing_weights[batch_idx, expert_idx].unsqueeze(1)
            experts_feature[batch_idx] += weight * feature

        experts_feature = experts_feature.reshape(x_b, self.num_tokens, -1)
        return experts_feature

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
