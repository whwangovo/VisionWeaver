from __future__ import annotations

from typing import Optional

from visionweaver.config import get_config

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
UNK_INDEX = 0
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

SERVER_ERROR_MSG = ""
MODERATION_MSG = ""


def _load_constants(cfg) -> None:
    global LOGDIR
    global CONTROLLER_HEART_BEAT_EXPIRATION
    global WORKER_HEART_BEAT_INTERVAL
    global UNK_INDEX
    global IGNORE_INDEX
    global IMAGE_TOKEN_INDEX
    global DEFAULT_IMAGE_TOKEN
    global DEFAULT_IMAGE_PATCH_TOKEN
    global DEFAULT_IM_START_TOKEN
    global DEFAULT_IM_END_TOKEN
    global IMAGE_PLACEHOLDER
    global SERVER_ERROR_MSG
    global MODERATION_MSG

    tokens = cfg.tokens
    heartbeat = cfg.heartbeat
    LOGDIR = cfg.logging.log_dir
    CONTROLLER_HEART_BEAT_EXPIRATION = heartbeat.controller_expiration
    WORKER_HEART_BEAT_INTERVAL = heartbeat.worker_interval
    UNK_INDEX = tokens.unk_index
    IGNORE_INDEX = tokens.ignore_index
    IMAGE_TOKEN_INDEX = tokens.image_token_index
    DEFAULT_IMAGE_TOKEN = tokens.default_image_token
    DEFAULT_IMAGE_PATCH_TOKEN = tokens.default_image_patch_token
    DEFAULT_IM_START_TOKEN = tokens.default_im_start_token
    DEFAULT_IM_END_TOKEN = tokens.default_im_end_token
    IMAGE_PLACEHOLDER = tokens.image_placeholder
    SERVER_ERROR_MSG = cfg.messages.server_error_msg
    MODERATION_MSG = cfg.messages.moderation_msg


def reload_constants(cfg: Optional[object] = None) -> None:
    _load_constants(cfg or get_config())


reload_constants()
