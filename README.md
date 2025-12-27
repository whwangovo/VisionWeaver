# VisionWeaver: Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models
[![arXiv](https://img.shields.io/badge/Arxiv-2509.13836v1-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.13836v1) 
[![hf_space](https://img.shields.io/badge/🤗-Dataset%20In%20HF-red.svg)](https://huggingface.co/datasets/KirenWH/VHBench_10)

This repository contains the official implementation of <b>VisionWeaver</b> and the <b>VHBench-10</b> benchmark. VisionWeaver is a novel architecture designed to mitigate object hallucinations in Large Vision-Language Models (LVLMs) by dynamically aggregating features from multiple specialized visual encoders.

## 📣 News
- `[2025.12.28]`  🔥 we released the training code.
- `[2025.09.17]`  we released our paper on arXiv.
- `[2025.09.17]`  the VHBench-10 benchmark became available at our [GitHub repository](https://github.com/whwangovo/VisionWeaver).
- `[2025.08.21]`  🔥 our paper was accepted by EMNLP Findings.


## 😮 Highlights

Object hallucinations, describing objects or attributes not present in an image, critically undermine the reliability of Large Vision-Language Models (LVLMs). We hypothesize that different visual encoders possess distinct inductive biases, leading to varied hallucination patterns. To address this, we introduce **Vision Weaver**, a Context-Aware Routing Network that intelligently leverages the strengths of multiple visual experts.

### VHBench-10: A Fine-Grained Hallucination Benchmark

To systematically analyze hallucination, we developed **VHBench-10**, a benchmark designed to evaluate LVLMs across 10 fine-grained hallucination categories. These are grouped into four core visual competencies: **Detection, Segmentation, Localization, and Classification**. 

### VisionWeaver: A Context-Aware Routing Network

Vision Weaver dynamically aggregates visual features from multiple specialized encoders, guided by the model's global visual understanding. It uses the `[CLS]` token from a primary CLIP encoder to generate routing signals, which create a weighted fusion of features from experts like DINOv2, SAM, and Vary, thereby reducing hallucinations and improving overall performance. 

## ⚙️ Requirements and Installation

Current environment (tested):

- Python 3.12
- torch 2.9.1 / torchvision 0.24.1
- transformers 4.57.3 / tokenizers 0.22.1
- hydra-core 1.3.2 / omegaconf 2.3.0
- accelerate 1.12.0 / deepspeed 0.15.4 / bitsandbytes 0.49.0
- huggingface_hub 0.36.0 / pillow 10.4.0

Install (editable):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

Notes:

- Use a CUDA-matching PyTorch wheel if you plan to train on GPU.
- Full dependency list (with pinned versions) lives in `pyproject.toml`.

## 🗝️ Training

Steps:

1) Update your dataset/model/output paths in the config files under `configs/`.
2) Run one of the training scripts:

```bash
scripts/pretrain_qwen_3b.sh
# or
scripts/pretrain_llama_3b.sh
# or
scripts/finetune_qwen_3b.sh
# or
scripts/finetune_llama_3b.sh
```

## 🐳 VHBench-10 Benchmark

VHBench-10 is a core contribution of our work, enabling a fine-grained diagnosis of visual perception failures in LVLMs. 

- **Structure**: The dataset contains approximately 10,00 samples. Each sample is a ternary `(I, R, H)`, where `I` is the image, `R` is the real, factually accurate caption, and `H` is a caption with a specific, deliberately injected hallucination. 
- **Categories**: The benchmark covers 10 distinct sub-categories, including Color, Shape, Counting, Text Recognition, Relative Position, and more. 
- **Generation**: Hallucinated captions were generated using GPT-4o, guided by specialized prompts to target each sub-category. 

The complete benchmark is available for download at our [HF repository](https://huggingface.co/datasets/KirenWH/VHBench_10).

## 👍 Acknowledgement

- Our work is built upon the [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) architecture.

- We thank the creators of the visual encoders used as experts in our model: [CLIP](https://github.com/openai/CLIP), [ConvNext](https://github.com/facebookresearch/ConvNeXt), [DINOv2](https://github.com/facebookresearch/dinov2), [EVA-02](https://github.com/baaivision/EVA), [SAM](https://github.com/facebookresearch/segment-anything), and [Vary](https://github.com/Ucas-HaoranWei/Vary).

## 🔒 License

- The majority of this project is released under the Apache 2.0 license as found in the `LICENSE` file.

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation ✏️.
```latex
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
