# Attribution and Third-Party Dependencies

This repository is a source release of research code. Attribution of methodological ideas, reuse of software, and use of pretrained weights are distinct obligations. Method descriptions and papers should identify each explicitly.

## MetaCompress

Yi Wang et al. *Rethinking Token Reduction for Large Vision-Language Models*. CVPR 2026. [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html).

This project draws on compression matrices for convex token aggregation and preservation of decoder predictions. The Tree/Column spatial partition generators differ from the original MetaCompress generator. The final default objective uses KL alone and should not be described as the complete original objective with row entropy. Historical comparison implementations are retained separately.

## MonkeyOCRv2

Yuliang Liu et al. *MonkeyOCRv2: A Visual-Text Foundation Model for Document AI*. [arXiv:2607.11562](https://arxiv.org/abs/2607.11562).

The official pretrained model is distributed as `zenosai/MonkeyOCRv2-B-Und`. The upstream project states that its code and model weights are released under Apache 2.0; dataset annotations and source images have separate terms. Third-party weights and remote model implementations are obtained through explicit downloads and are not included here.

## Qwen and evaluation tools

The native backbone is `Qwen/Qwen3-VL-8B-Instruct`; its pinned revision is specified in the downloader. The UND configuration retains the official Qwen3-1.7B decoder. Qwen, Transformers, PyTorch, DUE evaluator, and their associated software or weights remain subject to their respective licenses. DUE corpus F1 and WTQ denotation scoring use due-evaluator.

The related-work bibliography is available in `docs/references.bib`. Cite the methods actually used; the bibliography alone does not establish novelty or priority.

## Optional third-party baselines

The S²Prune adapter is retained in `vqa/supplement_s2prune.py`, but its upstream allocation implementation is not redistributed. Historical baseline runs require obtaining that implementation from the upstream source recorded by their scripts. It is not required for core Tree/Column execution. Official ms-swift fine-tuning likewise requires a separate upstream installation.

## Project licensing status

The original working directory did not specify a separate open-source license for the project's added code. This preparation preserves that status and does not assign MIT, Apache, or other licensing terms on behalf of the rights holder. No author identity, paper DOI, or repository URL has been invented. The rights holder should determine the project LICENSE before publication; existing third-party licenses remain unchanged.
