[![arXiv](https://img.shields.io/badge/Arxiv-2509.13836v1-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.13836v1) 
[![hf_space](https://img.shields.io/badge/🤗-Dataset%20In%20HF-red.svg)](https://huggingface.co/datasets/KirenWH/VHBench_10)

This repository contains the official implementation of <b>VisionWeaver</b> and the <b>VHBench-10</b> benchmark. Vision Weaver is a novel architecture designed to mitigate object hallucinations in Large Vision-Language Models (LVLMs) by dynamically aggregating features from multiple specialized visual encoders.

## 📣 News

- `[2025.09.17]`  🔥 We have released our paper on arXiv. 
- `[2025.09.17]`  🔥 The VHBench-10 benchmark are now available at our [GitHub repository](https://github.com/whwangovo/VisionWeaver). 

## 😮 Highlights

Object hallucinations, describing objects or attributes not present in an image, critically undermine the reliability of Large Vision-Language Models (LVLMs). We hypothesize that different visual encoders possess distinct inductive biases, leading to varied hallucination patterns. To address this, we introduce **Vision Weaver**, a Context-Aware Routing Network that intelligently leverages the strengths of multiple visual experts.

### VHBench-10: A Fine-Grained Hallucination Benchmark

To systematically analyze hallucination, we developed **VHBench-10**, a benchmark designed to evaluate LVLMs across 10 fine-grained hallucination categories. These are grouped into four core visual competencies: **Detection, Segmentation, Localization, and Classification**. 

### Vision Weaver: A Context-Aware Routing Network

Vision Weaver dynamically aggregates visual features from multiple specialized encoders, guided by the model's global visual understanding. It uses the `[CLS]` token from a primary CLIP encoder to generate routing signals, which create a weighted fusion of features from experts like DINOv2, SAM, and Vary, thereby reducing hallucinations and improving overall performance. 

## ⚙️ Requirements and Installation

To be released.

## 🗝️ Training & Inference

To be released.

## 🐳 VHBench-10 Benchmark

VHBench-10 is a core contribution of our work, enabling a fine-grained diagnosis of visual perception failures in LVLMs. 

- **Structure**: The dataset contains approximately 10,000 samples. Each sample is a ternary `(I, R, H)`, where `I` is the image, `R` is the real, factually accurate caption, and `H` is a caption with a specific, deliberately injected hallucination. 
- **Categories**: The benchmark covers 10 distinct sub-categories, including Color, Shape, Counting, Text Recognition, Relative Position, and more. 
- **Generation**: Hallucinated captions were generated using GPT-4O-mini, guided by specialized prompts to target each sub-category. 

The complete benchmark is available for download at our [HF repository](https://huggingface.co/datasets/KirenWH/VHBench_10).

## 👍 Acknowledgement

- Our work is built upon the [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) architecture.

- We thank the creators of the visual encoders used as experts in our model: [CLIP](https://github.com/openai/CLIP), [ConvNext](https://github.com/facebookresearch/ConvNeXt), [DINOv2](https://github.com/facebookresearch/dinov2), [EVA-02](https://github.com/baaivision/EVA), [SAM](https://github.com/facebookresearch/segment-anything), and [Vary](https://github.com/Ucas-HaoranWei/Vary).

## 🔒 License

- The majority of this project is released under the Apache 2.0 license as found in the `LICENSE` file.

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation ✏️.
```latex
@article{wang2025visionweaver,
  title={Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models},
  author={Wang, Weihang and Li, Xinhao and Wang, Ziyue and Pang, Yan and Zhang, Jielei and Li, Peiyi and Zhang, Qiang and Gao, Longwen},
  journal={arXiv preprint arXiv:2509.13836},
  year={2025}
}
```

