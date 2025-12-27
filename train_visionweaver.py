# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import logging
import os

import hydra
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_utils import get_last_checkpoint

from visionweaver import config as config_lib
from visionweaver import constants
from visionweaver import conversation as conversation_lib
from visionweaver.args_utils import DataArguments, ModelArguments, TrainingArguments
from visionweaver.data_utils import make_supervised_data_module
from visionweaver.model_utils import find_all_linear_names, get_model
from visionweaver.utils import print_trainable_params, rank0_print
from visionweaver.visionweaver_trainer import VisionWeaverTrainer


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    def save_adapter(keys_to_match):
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                ckpt_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(ckpt_folder, exist_ok=True)
                torch.save(
                    weight_to_save, os.path.join(ckpt_folder, f"{current_folder}.bin")
                )
            else:
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))

    check_only_save_mm_adapter_tunnable = (
        getattr(trainer.args, "mm_tunable_parts", "") == "pretrain"
    )

    trainer.accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")

    if check_only_save_mm_adapter_tunnable:
        save_adapter(["expert_projectors", "vision_router", "mm_projector"])
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def build_args(cfg: DictConfig):
    model_args = ModelArguments(**OmegaConf.to_container(cfg.model, resolve=True))
    data_args = DataArguments(**OmegaConf.to_container(cfg.data, resolve=True))
    training_args = TrainingArguments(
        **OmegaConf.to_container(cfg.training, resolve=True)
    )
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        training_args.local_rank = int(env_local_rank)
    return model_args, data_args, training_args


def maybe_enable_input_require_grads(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def maybe_apply_lora(model, training_args):
    if not training_args.lora_enable:
        return model

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    return get_peft_model(model, lora_config)


def set_default_conversation(version):
    if version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


def set_params_requires_grad_by_name(model, name_substrings, requires_grad=True):
    for name, param in model.named_parameters():
        if any(substring in name for substring in name_substrings):
            param.requires_grad_(requires_grad)


def configure_tunable_parts(model, vision_tower, model_args, training_args):
    stage = (model_args.mm_tunable_parts or "").strip()
    if stage not in {"pretrain", "finetune"}:
        raise ValueError(
            "mm_tunable_parts must be set to 'pretrain' or 'finetune'."
        )

    rank0_print(f"Using mm_tunable_parts: {stage}")
    model.config.mm_tunable_parts = training_args.mm_tunable_parts = stage

    if stage == "pretrain":
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(True)
        set_params_requires_grad_by_name(model, ["expert_projectors", "vision_router"])
    else:
        model.requires_grad_(True)
        vision_tower.requires_grad_(True)
        model.get_model().mm_projector.requires_grad_(True)


def configure_multimodal(model, tokenizer, model_args, data_args, training_args):
    if model_args.vision_tower is None:
        return

    model.get_model().initialize_vision_modules(
        model_args=model_args, fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    configure_tunable_parts(model, vision_tower, model_args, training_args)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
        model_args.mm_use_im_start_end
    )
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    config_lib.set_config(cfg)
    constants.reload_constants(cfg)

    model_args, data_args, training_args = build_args(cfg)

    model, tokenizer = get_model(
        model_args, training_args, tokenizer_cfg=cfg.tokenizer
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        maybe_enable_input_require_grads(model)

    model = maybe_apply_lora(model, training_args)

    tokenizer.pad_token = tokenizer.unk_token
    set_default_conversation(model_args.version)
    configure_multimodal(model, tokenizer, model_args, data_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        print_trainable_params(model)
        # print_config(model.config)

    trainer = VisionWeaverTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint and not training_args.overwrite_output_dir:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
