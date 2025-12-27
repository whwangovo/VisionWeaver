import torch
import transformers

from visionweaver.config import get_config

from visionweaver.model import VisionWeaverLlamaForCausalLM, VisionWeaverQwenForCausalLM


def _resolve_torch_dtype(training_args):
    if getattr(training_args, "bf16", False):
        return torch.bfloat16
    if getattr(training_args, "fp16", False):
        return torch.float16
    return None


def get_model(model_args, training_args, tokenizer_cfg=None):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError(
            "The 'sdpa' attention implementation requires torch version 2.1.2 or higher."
        )

    torch_dtype = _resolve_torch_dtype(training_args)

    if model_args.vision_tower is not None:
        if "qwen" in model_args.model_name_or_path.lower():
            model = VisionWeaverQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            model = VisionWeaverLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

    if tokenizer_cfg is None:
        tokenizer_cfg = get_config().tokenizer
    use_fast = getattr(tokenizer_cfg, "use_fast", False)
    padding_side = getattr(tokenizer_cfg, "padding_side", "right")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
    )

    return model, tokenizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = [
        "vision_tower",
        "vision_experts",
        "mm_projector",
        "vision_router",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            # print(f"continue, {name}")
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
