from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
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
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        config_dir = str(Path(config_path).parent)
        config_name = Path(config_path).stem
        if GlobalHydra.instance().is_initialized():
            cfg = compose(config_name=config_name)
        else:
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name=config_name)
    _CONFIG = cfg
    return _CONFIG


def set_config(cfg: DictConfig) -> None:
    global _CONFIG
    _CONFIG = cfg


def get_config() -> DictConfig:
    if _CONFIG is None:
        return load_config()
    return _CONFIG
