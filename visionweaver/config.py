from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

_CONFIG: Optional[DictConfig] = None


def _default_config_path() -> str:
    env_path = os.environ.get("VISIONWEAVER_CONFIG")
    if env_path:
        return env_path
    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = repo_root / "configs" / "train.yaml"
    if repo_candidate.exists():
        return str(repo_candidate)
    cwd_candidate = Path.cwd() / "configs" / "train.yaml"
    if cwd_candidate.exists():
        return str(cwd_candidate)
    raise FileNotFoundError("Could not locate configs/train.yaml")


def load_config(path: Optional[str] = None) -> DictConfig:
    global _CONFIG
    config_path = path or _default_config_path()
    _CONFIG = OmegaConf.load(config_path)
    return _CONFIG


def set_config(cfg: DictConfig) -> None:
    global _CONFIG
    _CONFIG = cfg


def get_config() -> DictConfig:
    if _CONFIG is None:
        return load_config()
    return _CONFIG
