<!-- # 📚 LLM Self-Training Research Paper List -->

<!-- > A curated collection of papers on Large Language Model self-training methods, focusing on data selection, internal representations, and self-improvement mechanisms. -->

# Self-training-Paper-List
Self-training relies on the model's internal features, output layer probability distributions, and other data to construct supervisory signals. It selects essential key data and designs more reasonable training objectives, achieving a leap in reasoning capabilities through training.

## 🎯 Research Focus

This repository organizes recent research papers on LLM self-training, particularly in the following areas:

- **Data Selection Methods** - Intelligent selection of high-quality training data
- **Internal Logits Utilization** - Leveraging model internal representations
- **Entropy Distribution Analysis** - Information entropy-based optimization strategies
- **Self-Improvement Mechanisms** - Unsupervised model enhancement techniques

## 📋 Paper Categories

### 1. Self-Training with Supervised Fine-Tuning (SFT)

#### 📖 Data Selection & Quality Control

**Self-training Large Language Models through Knowledge Detection**
- **Authors:** Wei Jie Yeo, Teddy Ferdinan, Przemyslaw Kazienko, Ranjan Satapathy, Erik Cambria
- **Venue:** Findings of EMNLP 2024
- **Core Contribution:** Proposes a self-training paradigm where LLMs autonomously curate labels and selectively train on unknown data samples identified through reference-free consistency method
- **Paper:** [arXiv:2406.11275](https://arxiv.org/abs/2406.11275) | [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.883/)
- **Code:** TBA

**Self Iterative Label Refinement via Robust Unlabeled Learning**
- **Authors:** Hikaru Asano, Tadashi Kozuno, Yukino Baba
- **Venue:** arXiv:2502.12565v1 (Feb 2025)
- **Core Contribution:** Proposes an iterative pipeline using unlabeled-unlabeled (UU) learning to optimize LLM-generated pseudo-labels for classification task SFT
- **Paper:** [arXiv:2502.12565](https://arxiv.org/abs/2502.12565)
- **Code:** TBA


#### 🛡️ Robustness & Self-Correction

**Enhancing LLM Robustness to Perturbed Instructions: An Empirical Study (SFT-SD)**
- **Authors:** Aryan Agrawal, Lisa Alazraki, Shahin Honarvar, Marek Rei
- **Venue:** arXiv:2504.02733 (April 2025)
- **Core Contribution:** Explores supervised fine-tuning self-denoising (SFT-SD) where LLMs use LoRA fine-tuning to recover from perturbed instructions
- **Paper:** [arXiv:2504.02733](https://arxiv.org/abs/2504.02733)
- **Code:** TBA

**S2r: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning (SFT Stage)**
- **Authors:** Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, Jia Li
- **Venue:** arXiv:2502.12853v1 (Feb 2025)
- **Core Contribution:** Framework's first stage uses SFT with curated data including dynamic trial-and-error trajectories to initialize self-verification and self-correction behaviors
- **Paper:** [arXiv:2502.12853](https://arxiv.org/abs/2502.12853)
- **Code:** TBA

### 2. Self-Training with RLHF & Alignment Methods (DPO, RLAIF, Self-Rewarding)

#### 🏆 Self-Rewarding & LLM-as-a-Judge

**Self-Rewarding Language Models**
- **Authors:** Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, Jason Weston
- **Venue:** arXiv:2401.10020 (Jan 2024)
- **Core Contribution:** Proposes LLM-as-a-Judge during training (iterative DPO) to provide reward signals, simultaneously improving instruction following and reward modeling capabilities
- **Paper:** [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) | [HuggingFace](https://huggingface.co/papers/2401.10020)
- **Code:** [lucidrains/self-rewarding-lm-pytorch](https://github.com/lucidrains/self-rewarding-lm-pytorch)



#### 🎯 Confidence-Based & Reasoning-Focused Methods

**Self-Training Large Language Models with Confident Reasoning (CORE-PO)**
- **Authors:** Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, Sungsoo Ahn
- **Venue:** arXiv:2505.17454v1 (May 2025)
- **Core Contribution:** Proposes CORE-PO method using online DPO to fine-tune LLMs to prefer high-quality reasoning paths based on reasoning-level confidence judgment
- **Paper:** [arXiv:2505.17454](https://arxiv.org/abs/2505.17454)
- **Code:** TBA

**Self-Generated Critiques Boost Reward Modeling for Language Models (Critic-RM)**
- **Authors:** Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, Melanie Kambadur, Dhruv Mahajan, Rui Hou
- **Venue:** NAACL 2025 (Long Papers)
- **Core Contribution:** Proposes Critic-RM framework enabling LLMs to improve reward modeling using self-generated critiques without external teacher model supervision
- **Paper:** [2025.naacl-long.573](https://aclanthology.org/2025.naacl-long.573/)
- **Code:** TBA

#### 🔄 Multi-Agent & Contrastive Methods

**MACPO: Weak-to-Strong Alignment via Multi-Agent Contrastive Preference Optimization**
- **Authors:** Yougang Lyu, Lingyong Yan, Zihan Wang, Dawei Yin, Pengjie Ren, Maarten de Rijke, Zhaochun Ren
- **Venue:** ICLR 2025
- **Core Contribution:** Studies weak-to-strong model alignment using DPO and self-generated negative data through contrasting responses from different models
- **Paper:** [arXiv:2410.07672](https://arxiv.org/abs/2410.07672)
- **Code:** TBA

**A Critical Evaluation of AI Feedback for Aligning Large Language Models**
- **Authors:** Archit Sharma, Sedrick Keh, Eric Mitchell, Chelsea Finn, Kushal Arora, Thomas Kollar
- **Venue:** NeurIPS 2024
- **Core Contribution:** Evaluates Learning from AI Feedback (LAIF) effectiveness, involving SFT with teacher model demonstrations followed by RL/DPO with critic model feedback
- **Paper:** [arXiv:2402.12366](https://arxiv.org/abs/2402.12366)
- **Code:** TBA

### 3. Self-Training with Selective Data Sampling

#### 📊 Consistency & Confidence-Based Selection

**Self-training Large Language Models through Knowledge Detection**
- **Authors:** Yeo Wei Jie, Teddy Ferdinan, Przemyslaw Kazienko, Ranjan Satapathy, Erik Cambria
- **Venue:** Findings of EMNLP 2024
- **Core Contribution:** Uses "reference-free consistency" method to selectively train on key data samples unknown to the model
- **Paper:** [arXiv:2406.11275](https://arxiv.org/abs/2406.11275) | [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.883/)
- **Code:** TBA

**Self-Training Large Language Models with Confident Reasoning**
- **Authors:** Hyosoon Jang, Yunhui Jang, Sungjae Lee, Jungseul Ok, Sungsoo Ahn
- **Venue:** arXiv:2505.17454v1 (May 2025)
- **Core Contribution:** Selects high-quality reasoning paths (pseudo-labels) for training based on model's own answer-level confidence and novel reasoning-level confidence
- **Paper:** [arXiv:2505.17454](https://arxiv.org/abs/2505.17454)
- **Code:** TBA

### 4. Self-Training with Reasoning & Internal Mechanisms

#### 🧠 Process-Level & Step-wise Methods

**Process-based Self-Rewarding Language Models**
- **Authors:** Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, Yeyun Gong
- **Venue:** arXiv:2503.03746 (Mar 2025)
- **Core Contribution:** Introduces long-thought reasoning, step-wise LLM-as-a-Judge, and step-wise preference optimization within self-rewarding paradigm for mathematical reasoning
- **Paper:** [arXiv:2503.03746](https://arxiv.org/abs/2503.03746)
- **Code:** TBA

**ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning**
- **Authors:** Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, Yaoxue Zhang
- **Venue:** arXiv:2505.04881 (May 2025)
- **Core Contribution:** Proposes framework to simplify reasoning chains by reinforcing model's confidence during inference, preventing redundant reflection step generation
- **Paper:** [arXiv:2505.04881](https://arxiv.org/abs/2505.04881)
- **Code:** TBA

#### 🔗 Chain-of-Thought Enhancement

**Self-Reasoning Language Models: Unfold Hidden Reasoning Chains**
- **Authors:** Hongru Wang, Deng Cai, Wanjun Zhong, Shijue Huang, Jeff Z. Pan, Zeming Liu, Kam-Fai Wong
- **Venue:** arXiv:2505.14116 (May 2025)
- **Core Contribution:** Introduces Self-Reasoning Language Model (SRLM) where model synthesizes longer CoT data and iteratively improves through self-training with few reasoning catalyst examples
- **Paper:** [arXiv:2505.14116](https://arxiv.org/abs/2505.14116)
- **Code:** TBA

#### 🎓 Autonomous Learning Paradigms

**LLMs Could Autonomously Learn Without External Supervision**
- **Authors:** Ke Ji, Junying Chen, Anningzhe Gao, Wenya Xie, Xiang Wan, Benyou Wang
- **Venue:** arXiv:2406.00606 (Jun 2024)
- **Core Contribution:** Presents Autonomous Learning paradigm freeing models from human supervision constraints, enabling self-education through direct text interaction
- **Paper:** [arXiv:2406.00606](https://arxiv.org/abs/2406.00606)
- **Code:** TBA



## 🤝 Contributing

We welcome contributions to this paper list! Please feel free to:

1. **Add new papers** - Submit a PR with paper details following the format
2. **Update information** - Correct any missing or outdated information
3. **Suggest categories** - Propose new categorization schemes
4. **Report issues** - Open an issue for any problems or suggestions

### Contributing Format
When adding papers, please include:
- Complete author list
- Venue (conference/journal/preprint)
- Core contribution summary
- Paper and code links
- Publication date

<!-- ## 📚 Citation

If you find this collection useful, please consider citing:

```bibtex
@misc{llm-self-training-papers,
  title={LLM Self-Training Research Paper List},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[username]/self-training}}
}
``` -->

---


<!-- For questions or suggestions, please open an issue or contact [maintainer-email]. -->

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star!**
<!-- 
[![GitHub stars](https://img.shields.io/github/stars/username/repo-name.svg?style=social&label=Star)](../../stargazers)
[![GitHub forks](https://img.shields.io/github/forks/username/repo-name.svg?style=social&label=Fork)](../../network/members) -->

</div>
