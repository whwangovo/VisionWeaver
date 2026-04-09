import logging
import logging.handlers
import os
import sys

import torch

from visionweaver import constants

handler = None


def rank0_print(*args, **kwargs):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def print_config(config):
    rank0_print(config)


def print_trainable_params(model):
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
