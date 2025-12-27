from __future__ import annotations


def require_config_value(config, name: str):
    value = getattr(config, name, None)
    if value is None or value == "":
        raise ValueError(f"{name} must be set in the config.")
    return value


def load_clip_image_processor(config):
    from transformers import CLIPImageProcessor

    processor_name = require_config_value(config, "vision_image_processor")
    return CLIPImageProcessor.from_pretrained(processor_name)


def log_already_loaded(vision_tower_name: str) -> None:
    print(f"{vision_tower_name} is already loaded, `load_model` called again, skipping.")
