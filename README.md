<div align="center">
  <img src="images/cca.png" alt="Your Image" width="140px" style="float: left; margin-right: 1px;"/>
</div>

<h3 align="center"><a href="https://arxiv.org/abs/2410.15926" style="color:#9C276A">
Mitigating Object Hallucination via Concentric Causal Attention</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2410.15926-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.15926) <a href="https://huggingface.co/papers/2410.15926"></a> <a href='https://huggingface.co/xing0047/cca-llava-1.5-7b'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> <a href="https://huggingface.co/papers/2410.15926"></a> <a href='https://huggingface.co/datasets/xing0047/SPP'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spatial Position Probing-green'></a><br>

This is the official repository of the following paper and a project that study positional perception in LVLMs.
> **[Mitigating Object Hallucination via Concentric Causal Attention](https://arxiv.org/abs/2410.15926)**<br>
> NeurIPS 2024<br>
> Yun Xing*, Yiheng Li*, Ivan Laptev, Shijian Lu†<br>

## 🎉 News

- [2024/11/07] Information Flow visualization is available.
- [2024/10/22] [Paper](https://arxiv.org/abs/2410.15926) is available on arXiv.
- [2024/10/21] CCA-LLaVA supports evaluation of multiple benchmarks, including `pope`, `chair`, `amber` for hallucination, and `mmstar`, `gqa`, `seed`, `vizwiz_vqa`, `scienceqa` for general LVLM multiple-choice questions. Please refer to [this doc](https://github.com/xing0047/cca-llava/blob/main/docs/Eval.md) for details.
- [2024/09/27] CCA is accepted to NeurIPS 2024🎉.

## 🕹️ Approach
- We reveal that **object hallucination** is closely tied with **Rotary Position Encoding** (RoPE), a widely adopted positional dependency modeling design in existing LVLMs. Due to the **long-term decay** in RoPE, LVLMs suffer from **recency bias** and tend to hallucinate more when relevant visual cues are distant from instruction tokens (user query) in the multimodal input sequence.
- Motivated by this, we propose **Concentric Causal Attention (CCA)**, a simple yet effective positional alignment strategy that mitigates the impact of RoPE long-term decay in LVLMs by placing critical visual cues closer to user instructions, thereby alleviating object hallucinations. 
<div align="center">
  <img src="images/cca_attn.jpg" alt="Your Image" width="65%" style="float: left; margin-right: 1px;"/>
</div>

## 🔥 Spatial Position Probing

- To further verify effectiveness of our approach, we craft a large-scale object hallucination evaluation set, involving over 2,000,000 testing samples that are diverse in object spatial positions and object sizes. Our model surpasses LLaVA-1.5 across diverse spatial positions and object scales consistently. 
<div align="center">
  <img src="images/spatial_position_probing.png" alt="Your Image" width="95%" style="float: left; margin-right: 1px;"/>
</div>

## 🛠️ Install
```
conda create -n cca-llava python=3.10 -y
conda activate cca-llava
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e ".[train]"
pip install triton==2.1.0 pynvml==11.5.0 --upgrade
pip install flash-attn==2.5.8 --no-build-isolation --no-cache-dir
```

## 🤗 Model
- [cca-llava-1.5-7b](https://huggingface.co/xing0047/cca-llava-1.5-7b)

## 📜 Data
Please refer to [Data.md](https://github.com/xing0047/cca-llava/blob/main/docs/Data.md) for preparation of training data.

## 🌟 Train
CCA-LLaVA training pipeline follows [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). The training consists of two stages:
- **Step 1, pretraining**. Train a projector on a CC3M subset of ∼558K image-text pairs to connect a frozen pretrained vision encoder and a frozen LLM.
  ```
  bash scripts/v1_5/pretrain.cca-llava-1.5-7b.sh
  ```
- **Step 2, instruction tuning**. Fine-tune projector and LLM with ~665k multimodal instruction data.
  ```
  bash scripts/v1_5/finetune.cca-llava-1.5-7b.sh
  ```

## 🔍 Eval
Please refer to [Eval.md](https://github.com/xing0047/cca-llava/blob/main/docs/Eval.md) for details.

## 🕹️ Usage
The two core modifications `concentric positions` and `concentric causal masking` can be found in `llava/cca_utils` folder. To replace default causal scheme with our proposed cca, you can prepend following code to either training or evaluation code, subject to your own use case.
```
import transformers
from llava.cca_utils.cca import llamaforcausallm_forward, cca_forward 
transformers.models.llama.LlamaForCausalLM.forward = llamaforcausallm_forward
transformers.models.llama.LlamaModel.forward = cca_forward
```

## ✒️ Citation
```
@article{xing2024mitigating,
  title={Mitigating Object Hallucination via Concentric Causal Attention},
  author={Xing, Yun and Li, Yiheng and Laptev, Ivan and Lu, Shijian},
  journal={arXiv preprint arXiv:2410.15926},
  year={2024}
}
```

## ❤️ Acknowledgement
Thanks for their wonderful work!
- [llava](https://github.com/haotian-liu/LLaVA): the codebase we use to implement cca.
- [roformer](https://github.com/ZhuiyiTechnology/roformer): codebase where rope is initially proposed.
- [opera](https://github.com/shikiw/OPERA): an excellent approach that mitigates object hallucination. the codebase we use to implement CHAIR evaluation.
- [pope](https://github.com/RUCAIBox/POPE): a widely adopted object hallucination benchmark.
- [amber](https://github.com/junyangwang0410/AMBER): a recent comprehensive hallucination benchmark involving object, attribute and relation hallucination.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): a comprehensive evaluation toolkit on LVLMs. the codebase we use to implement general LVLM benchmark evaluations.

