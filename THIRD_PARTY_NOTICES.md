# 引用与第三方依赖

本仓库是研究代码的源码整理。引用理论来源、重用具体软件和使用预训练权重是不同事项。
方法文档与论文应分别交代以下内容。

## MetaCompress

Yi Wang et al. *Rethinking Token Reduction for Large Vision-Language Models*.
CVPR 2026. [论文原文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html)。

本项目借鉴“用压缩矩阵进行 token 凸组合”与 decoder prediction preservation。
Tree/Column 的空间分区生成器不是 MetaCompress 原版；最终默认训练只使用 KL，
不应写成同时使用论文中行熵的完整原 loss。历史对照实现独立保留在源码中。

## MonkeyOCRv2

Yuliang Liu et al. *MonkeyOCRv2: A Visual-Text Foundation Model for Document AI*.
[arXiv:2607.11562](https://arxiv.org/abs/2607.11562)。

官方预训练来源：`zenosai/MonkeyOCRv2-B-Und`。官方项目声明代码与模型为 Apache 2.0；
数据标注和图像另有条款。第三方权重和远程模型实现通过明确下载操作取得，未附在本仓库。

## Qwen 与评测工具

原生模型来自 `Qwen/Qwen3-VL-8B-Instruct`，固定 revision 见下载实现。
UND 保持官方原 Qwen3-1.7B 解码器。Qwen、Transformers、PyTorch、DUE evaluator
等软件及权重受各自许可证约束。DUE corpus F1 和 WTQ denotation 使用 due-evaluator。
完整相关工作书目见 `docs/references.bib`；请按最终实际方法选取引用，不自动宣称创新优先权。

## 可选第三方 baseline

S²Prune 的适配代码仍在 `vqa/supplement_s2prune.py`；其上游 allocation 文件未随本仓库
分发。相关历史 baseline 需要依照原脚本登记的上游来源另行取得，不是核心 Tree/Column
运行所需依赖。官方 ms-swift 微调同样需要单独安装上游软件。

## 本项目授权状态

原工作目录没有给本项目新增源码指定独立的开源许可证。本次整理保留这一状态，
没有替权利人新增 MIT/Apache 等授权，也没有捏造作者姓名、论文 DOI 或 GitHub 地址。
推送到 GitHub 前，应由权利人决定项目 LICENSE；外部依赖的既有许可不受此影响。
