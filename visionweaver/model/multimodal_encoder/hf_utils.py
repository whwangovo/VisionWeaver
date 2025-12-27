from __future__ import annotations

import os
from typing import Optional


def resolve_checkpoint_path(
    checkpoint: Optional[str], filename: Optional[str] = None
) -> Optional[str]:
    if not checkpoint:
        return checkpoint
    if os.path.exists(checkpoint):
        return checkpoint

    repo_id = checkpoint
    repo_filename = filename
    if repo_filename is None and checkpoint.endswith((".pth", ".pt", ".bin")):
        parts = checkpoint.split("/", 2)
        if len(parts) == 3:
            repo_id = f"{parts[0]}/{parts[1]}"
            repo_filename = parts[2]

    if repo_filename is None:
        return checkpoint

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download checkpoints from Hugging Face."
        ) from exc

    return hf_hub_download(repo_id=repo_id, filename=repo_filename)
