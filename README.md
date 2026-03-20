<div align="center">
  <div>
  <h1>CLTuner: A Generalized Framework for Continual Visual Instrucion Tuning</h1>
  </div>
</div>

[//]: # (<img src=".\resources\overview.png">)
[![GitHub Repo stars](https://img.shields.io/github/stars/zdyoung0519/cltuner?style=social)](https://github.com/zdyoung0519/cltuner/stargazers)
[![license](https://img.shields.io/github/license/zdyoung0519/cltuner.svg)](https://github.com/zdyoung0519/cltuner/blob/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/zdyoung0519/cltuner)](https://github.com/zdyoung0519/cltuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/zdyoung0519/cltuner)](https://github.com/zdyoung0519/cltuner/issues)

## 📖 Introduction
***Still working in progress. Please be patient...***
This repository is built to achieve Generalized Continual Learning of Multimodal Large Language Models.

**Main Features**
- The codebase is mainly based on [Xtuner](), which supports efficient fine-tuning techniques such as FlashAttention, Tritoon kernels, and DeepSpeed.
- Support different methods and algorithms for continual instruction tuning, including SeqFine-Tuning, EWC, LwF, L2P and etc.
- Support different benchmarks and evaluations for CIT of MLLMs.

## 🔥 News 
<!-- - We propose [MPO-LLaVA](). -->

## ✋ Features

Continual Instruction Tuning (CIT) Methods:
- [x] LoRA: Fine-Tuning with LoRA modules.
- [ ] [EWC](): LoRA Fine-tuning with EWC penalization.
- [ ] [LwF](): LoRA Fine-tuning with LwF penalization.
- [x] MoELoRA: Fine-Tuning with Mixture of LoRA modules.
- [ ] [HiDe-LLaVA](): Expand and match LoRA moduls at the top layer, while fuse 
- [ ] [TaDyRA](): Task-adaptive fusion and Dynamic-Rank Adaptation.
<!-- - [ ] [Replay](): Replay previous data. -->
<!-- - [ ] [L2P](): Construct a pool of learnable prompts, and select the prompt that is most relative to the input. -->
<!-- - [ ] [MR-LoRA](https://arxiv.org/abs/2506.05453): Train isolated LoRA modules for tasks and a Router LoRA to select LoRA at inference.
-->

Benchmarks:
- [x] [UCIT](): contains 6 tasks that have small overlap with the LLaVA pre-training data.
- [ ] [COIN](https://arxiv.org/abs/2403.08350): contains 8 different visual instruction tuning task, including QA, Grounding and e.t.c. 
- [ ] [COIN-Sampled](): a subset of COIN, provided by [Guo, et al.](https://github.com/Ghy0501/HiDe-LLaVA)
- [ ] [COIN-ASD](): 
- [ ] [MLLM-CL](https://arxiv.org/abs/2506.05453): includes 2 incremental tasks, one for Domain Continual Learning (DCL) and one for Ability Continual Learning (ACL).
- [ ] [LLaVA-665k]

## 🛠️ Introduciton
### 1.Installation

Clone the repository with git
```angular2html
git clone https://github.com/ZDYoung0519/cltuner.git
cd cltuner
```


It is recommended to build a Python-3.10 virtual environment using conda

```
conda create --name my-cltuner-env python=3.10 -y
conda activate my-cltuner-env
pip install -e '.[all]'
# Or install with tsinghua mirror
pip install -e '.[all]' -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 2.Preparation 

#### 2.1 Dataset Preparation
Please refer to ```docs/datasets```.

#### 2.2 Model Preparation
##### 2.2.1 LLaVA 
For ```LLaVA-v1.5``` model ```without``` pretraining and instruction-tuning, you need to download download `vicuna-7b-v1.5` and `clip-vit-large-patch14-336`:
```
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir /storage/huggingface/lmsys/vicuna-7b-v1.5
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir /storage/huggingface/openai/clip-vit-large-patch14-336
```
For ```LLaVA-v1.5``` model ```with pretrained MLP```  weights, you need to run the fowllowing cmd to download the pre-training weights and covert it into `xtuner` format:
```
# download llava-v1.5-mlp2x
huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5  \ 
--local-dir /storage/huggingface/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5

# convert it into xtuner format
python ./cltuner/tools/convert_projector_to_xtuner.py \
 --src_path /storage/huggingface/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin  \
 --dst_path /storage/huggingface/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector_xtuner.pt
```
If you want to use the `full weights` of `LLaVA-v1.5`, you need to run the fowllowing cmd to download and covert them into `xtuner` format:
```
# download llava-v1.5-mlp2x
huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5  --local-dir /storage/huggingface/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5

# convert it into xtuner format
python ./cltuner/tools/convert_llava_mlp_to_xtuner.py \
  --src_path /storage/huggingface/llava-v1.5-7b-xtuner/mm_projector.bin \
  --dst_path /storage/huggingface/llava-v1.5-7b-xtuner/mm_projector_xtuner.pt
```
NOTE: After downloading them, you need to modify the path in `./cltuner/configs/base/xxx.py`.



### 3. Train And Evaluate
We provide the training and evaluation scripts in ```scripts```. Take ```ucit_llava_v15pf_lora``` as an example, you can run the following command to train and evalute the model:
```
bash ./scripts/ucit_llava_v15pf_lora/run_all.sh
```
The naming convention for the experimental config files is as follows:
```
ucit_llava_v15pf_lora
{benchmark}_{architecture}_{pretrained_weights}_{method}
```
Here, "llava_v15pf" represents the LLaVA 1.5 model that has undergone both pretraining and instruction fine-tuning, "llava_v15p" represents the LLaVA 1.5 model that has only undergone pretraining, and "llava_v15" represents the model without pretraining or instruction fine-tuning.


## 🤝 Acknowledgement
This repository is built upon the following projects：
- [Xtuner]()
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [COIN]()
- [MCitLib]()

We sincerely thank these contributors.


## 🖊️ Citation


```bibtex
@misc{2025cvit,
    title={cvit: A Toolkit for Efficiently Sequential Fine-Tuning of Large VLMs},
    author={Dongyang Zhang},
    howpublished = {\url{https://github.com/zdyoung/cvit}},
    year={2025}
}

@misc{2025xxxx,
    title={xxxx},
    author={Dongyang Zhang, Junmin Liu et al.},
    howpublished = {xxxx},
    year={2025}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
