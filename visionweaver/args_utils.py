from dataclasses import dataclass, field
from typing import Optional

import transformers
from omegaconf import MISSING


@dataclass
class ModelArguments:
    # base args
    model_name_or_path: Optional[str] = field(default=MISSING)
    vision_tower: Optional[str] = field(default=MISSING)
    mm_tunable_parts: Optional[str] = field(default=MISSING)
    freeze_backbone: bool = field(default=MISSING)
    freeze_vision_tower: bool = field(default=MISSING)
    tune_mm_mlp_adapter: bool = field(default=MISSING)
    tune_vision_adapter: bool = field(default=MISSING)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=MISSING)
    # experts args
    clip_hr: bool = field(default=MISSING)
    image_size: Optional[int] = field(default=MISSING)
    patch_size: Optional[int] = field(default=MISSING)
    num_experts: Optional[int] = field(default=MISSING)
    # visual args
    version: Optional[str] = field(default=MISSING)
    mm_vision_select_layer: Optional[int] = field(default=MISSING)
    mm_projector_type: Optional[str] = field(default=MISSING)
    mm_use_im_start_end: bool = field(default=MISSING)
    mm_use_im_patch_token: bool = field(default=MISSING)
    mm_patch_merge_type: Optional[str] = field(default=MISSING)
    mm_vision_select_feature: Optional[str] = field(default=MISSING)
    input_image_size: Optional[int] = field(default=MISSING)
    do_resize: bool = field(default=MISSING)
    de_normalize: bool = field(default=MISSING)
    pix2struct_max_tokens: Optional[int] = field(default=MISSING)
    pix2struct_grid_size: Optional[int] = field(default=MISSING)
    pix2struct_resize_size: Optional[list] = field(default=MISSING)
    unfreeze_mm_vision_tower: bool = field(default=MISSING)
    s2_scales: Optional[str] = field(default=MISSING)
    add_pixel_shuffle: bool = field(default=MISSING)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=MISSING, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = field(default=MISSING)
    is_multimodal: bool = field(default=MISSING)
    image_folder: Optional[str] = field(default=MISSING)
    image_aspect_ratio: str = field(default=MISSING)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=MISSING)
    optim: str = field(default=MISSING)
    remove_unused_columns: Optional[bool] = field(default=MISSING)
    freeze_mm_mlp_adapter: bool = field(default=MISSING)
    mpt_attn_impl: Optional[str] = field(default=MISSING)
    model_max_length: int = field(
        default=MISSING,
        metadata={"help": "Maximum sequence length. Sequences will be right padded."},
    )
    double_quant: bool = field(
        default=MISSING,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default=MISSING,
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=MISSING, metadata={"help": "How many bits to use."})
    lora_enable: bool = field(default=MISSING)
    lora_r: int = field(default=MISSING)
    lora_alpha: int = field(default=MISSING)
    lora_dropout: float = field(default=MISSING)
    lora_weight_path: str = field(default=MISSING)
    lora_bias: str = field(default=MISSING)
    mm_projector_lr: Optional[float] = field(default=MISSING)
    group_by_modality_length: bool = field(default=MISSING)
    auto_find_batch_size: bool = field(default=MISSING)
    gradient_checkpointing: bool = field(default=MISSING)
    verbose_logging: bool = field(default=MISSING)
    attn_implementation: str = field(
        default=MISSING,
        metadata={"help": "Use transformers attention implementation."},
    )
