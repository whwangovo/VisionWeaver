# VisionWeaver: Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models

[![arXiv](https://img.shields.io/badge/Arxiv-2509.13836v1-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.13836v1)
[![hf_space](https://img.shields.io/badge/🤗-Dataset%20In%20HF-red.svg)](https://huggingface.co/datasets/KirenWH/VHBench_10)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP%202025-Findings-green.svg)](https://aclanthology.org/2025.findings-emnlp.936/)

Official implementation of **VisionWeaver** and the **VHBench-10** benchmark (EMNLP 2025 Findings).

VisionWeaver mitigates object hallucinations in Large Vision-Language Models (LVLMs) by dynamically aggregating features from multiple specialized visual encoders through a Context-Aware Routing Network.

## 📣 News

- `[2025.12.28]` 🔥 Training code released.
- `[2025.09.17]` Paper on [arXiv](https://arxiv.org/abs/2509.13836v1). VHBench-10 benchmark available.
- `[2025.08.21]` 🔥 Accepted by EMNLP 2025 Findings.

## 😮 Highlights

### VisionWeaver: Context-Aware Routing Network

Object hallucinations — describing objects or attributes absent from an image — critically undermine LVLM reliability. We hypothesize that different visual encoders possess distinct inductive biases, leading to varied hallucination patterns.

VisionWeaver addresses this by dynamically routing and fusing features from six specialized visual encoders:

| Encoder | Role |
|---------|------|
| [CLIP](https://github.com/openai/CLIP) | Primary encoder; provides `[CLS]` token as routing signal |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Self-supervised features for fine-grained recognition |
| [EVA-02](https://github.com/baaivision/EVA) | Enhanced ViT with improved training recipes |
| [SAM](https://github.com/facebookresearch/segment-anything) | Segmentation-aware spatial features |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | CNN-based multi-scale features |
| [Vary](https://github.com/Ucas-HaoranWei/Vary) | Document and OCR-oriented features |

The `[CLS]` token from the primary CLIP encoder generates routing weights that create a context-aware weighted fusion of expert features, reducing hallucinations across detection, segmentation, localization, and classification tasks.

### VHBench-10: Fine-Grained Hallucination Benchmark

**VHBench-10** evaluates LVLMs across 10 fine-grained hallucination categories grouped into four core visual competencies: Detection, Segmentation, Localization, and Classification.

- ~10,000 samples, each a ternary `(I, R, H)`: image, real caption, hallucinated caption
- Hallucinated captions generated via GPT-4o with category-specific prompts
- Available at [HuggingFace](https://huggingface.co/datasets/KirenWH/VHBench_10)

## 📁 Project Structure

```
VisionWeaver/
├── visionweaver/
│   ├── model/
│   │   ├── multimodal_encoder/     # Vision encoders (CLIP, DINO, EVA, SAM, ConvNeXt, Vary)
│   │   │   ├── base_encoder.py     # BaseVisionTower abstract base class
│   │   │   ├── utils.py            # Shared encoder utilities
│   │   │   └── *_encoder.py        # Individual encoder implementations
│   │   ├── multimodal_projector/   # Feature projection layers
│   │   └── visionweaver_arch.py    # Main model architecture
│   ├── conversation.py             # Conversation templates
│   ├── data_utils.py               # Dataset and data loading
│   └── utils.py                    # General utilities
├── configs/                        # Hydra configs (model, data, training, etc.)
├── scripts/                        # Training launch scripts
├── tests/                          # Pytest suite
├── train_visionweaver.py           # Training entrypoint
└── pyproject.toml
```

## ⚙️ Installation

### pip (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### pixi

```bash
pixi install
```

> Use a CUDA-matching PyTorch wheel for GPU training. Full pinned dependencies are in `pyproject.toml`.

## 🗝️ Training

1. Update dataset, model, and output paths in the config files under `configs/`.
2. Run a training script:

```bash
# Pretraining
scripts/pretrain_qwen_3b.sh
scripts/pretrain_llama_3b.sh

# Finetuning
scripts/finetune_qwen_3b.sh
scripts/finetune_llama_3b.sh
```

## 🧪 Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## 🐳 VHBench-10 Benchmark

VHBench-10 enables fine-grained diagnosis of visual perception failures in LVLMs.

| Competency | Categories |
|------------|------------|
| Detection | Object Existence, Attribute |
| Segmentation | Shape, Boundary |
| Localization | Relative Position, Counting |
| Classification | Color, Text Recognition, Scene, Action |

Download: [HuggingFace](https://huggingface.co/datasets/KirenWH/VHBench_10)

## 👍 Acknowledgement

- Built upon [LLaVA-1.5](https://github.com/haotian-liu/LLaVA).
- Visual encoder experts: [CLIP](https://github.com/openai/CLIP), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [DINOv2](https://github.com/facebookresearch/dinov2), [EVA-02](https://github.com/baaivision/EVA), [SAM](https://github.com/facebookresearch/segment-anything), [Vary](https://github.com/Ucas-HaoranWei/Vary).

## 🔒 License

This project is released under the [Apache 2.0 License](LICENSE).

## ✏️ Citation

```bibtex
@inproceedings{wang-etal-2025-diving,
    title = "Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models",
    author = "Wang, Weihang  and
      Li, Xinhao  and
      Wang, Ziyue  and
      Pang, Yan  and
      Zhang, Jielei  and
      Li, Peiyi  and
      Zhang, Qiang  and
      Gao, Longwen",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.936/",
    doi = "10.18653/v1/2025.findings-emnlp.936",
    pages = "17271--17289",
    ISBN = "979-8-89176-335-7",
}
```
