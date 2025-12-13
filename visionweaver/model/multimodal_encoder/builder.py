from llava.utils import rank0_print

from .clip_encoder import CLIPVisionTower
from .convnext_encoder import ConvNextVisionTower
from .dino_encoder import DINOVisionTower
from .eva_encoder import EVAVisionTower
from .hr_clip_encoder import HRCLIPVisionTower
from .sam_encoder import SAMVisionTower
from .siglip_encoder import SiglipVisionTower
from .vary_encoder import VaryVisionTower
from .visionweaver_encoder import VisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", "clip"),
    )

    if ";" in vision_tower:
        vision_tower_cfg.num_experts = len(vision_tower.split(";"))
        return VisionTower(vision_tower_cfg)
    else:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
