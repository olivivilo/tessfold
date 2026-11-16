# TessFold：冻结视觉语言模型的文档 Token 压缩

本仓库整理自当前开发版本，只包含代码、配置、公开来源锁定信息和复现说明。**不包含实验结果、模型/压缩器权重、数据集、特征缓存或虚拟环境。** 所有运行产物均被 `.gitignore` 排除。

两个模型路线有不同结构，请明确区分：

| 路线 | 冻结的主干 | P 的位置 | 生成器 |
|---|---|---|---|
| `und` | 官方 MonkeyOCRv2-B-Und，包含其原视觉塔、merger 和 Qwen3-1.7B | merger 前，或 merger 后、LLM 前 | 问题条件空间树 / 自适应列划分 |
| `native8b` | 完整原生 Qwen3-VL-8B-Instruct | 原生 merger 后，对最终视觉特征和三条 DeepStack 分支同步压缩 | Tree / Column / 独立 StableColumn 变体 |

原生 8B 路线使用自己的视觉塔与 LLM，不能描述为 MonkeyOCRv2-Und 换了一个 LLM。两条路线均冻结整个主干，仅训练新加入的压缩矩阵生成器。

## 从哪里开始

建议在仓库根目录运行命令。环境使用 Python 3.10 的 `venv`；需要 C++17 编译器 `g++`。

```bash
# CPU 安装与无需模型权重的回归检查
PYTHON_BIN=python3.10 DEVICE=cpu bash scripts/bootstrap_portable.sh
.venv/bin/python -m pytest
.venv/bin/python -m tessfold --help
```

CUDA / Ascend 需要先匹配设备驱动与 PyTorch wheel；详见 [环境构建](docs/INSTALL.md)。CPU 可以运行单元测试，但完整 8B 训练需要足够内存和计算资源。

## 模型、数据和训练入口

从 `configs/portable/` 复制一个配置到被 Git 忽略的 `configs/local/`，修改外部模型、数据路径与输出目录：

```bash
mkdir -p configs/local
cp configs/portable/native_tree50.json configs/local/run.json
# 编辑 configs/local/run.json 后：
.venv/bin/python -m tessfold download-model --config configs/local/run.json
.venv/bin/python -m tessfold validate --config configs/local/run.json
.venv/bin/python -m tessfold pilot --config configs/local/run.json --device npu
.venv/bin/python -m tessfold train --config configs/local/run.json --device npu
.venv/bin/python -m tessfold evaluate --config configs/local/run.json --device npu
```

`download-model` 是显式下载操作；其他命令只加载本地模型。数据准备见 [数据来源、格式与精确划分](docs/DATA.md)。发布仓库里只保存公开数据的来源信息与行索引重建配方，没有问答正文或答案。

多设备可把训练/评测命令改成：

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tessfold train --config configs/local/run.json --device npu
# 中断后，保持配置、代码、数据、模型及 world size 不变：
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tessfold train --config configs/local/run.json --device npu --resume
```

先导实验也支持 `--resume`。默认完整训练必须先通过同配置的先导门槛；通过后重新初始化生成器，不从先导权重续训。开发集选出的 checkpoint 在正式评测前锁定；正式测试分数不参与选模。

## 代码地图

| 路径 | 作用 |
|---|---|
| `tessfold/` | 新增可移植入口：配置、数据验证、来源锁定、训练、断点和独立评测 |
| `vqa/compression_generators_v2.py` | 原 UND 空间生成器，包括 Tree 与 Column |
| `vqa/tessfold_qwen3vl.py` | 原生 8B 的冻结主干接口、Tree/Column、四分支凸组合和 MRoPE |
| `vqa/tessfold_qwen3vl_stable.py` | 独立 StableColumn 变体，固定 detail 驱动分区 |
| `vqa/tessfold_native_partition.py` / `*.cpp` | 具有参考计算顺序的 C++ 分区实现及运行时构建 |
| `vqa/tessfold_*runtime.py`, `tessfold_prefix_cache.py` | 严格推理、分段候选、原生分区和独立问题 KV 管理 |
| `vqa/benchmark_metrics.py` | DocVQA / InfoVQA / ChartQA / OCRBench / DUE / WTQ / DT-VQA 指标 |
| `scripts/tessfold_accel/`, `tessfold_budget1075/`, `tessfold_repair/` | 保留的原实验训练、评测、计时、统计、审计与集群调度源码 |
| `scripts/` 其他模块 | 数据转换、既有压缩器基线/消融、诊断和报告生成代码 |
| `tests/` | 不下载基模的 CPU 数学、数据隔离、训练与恢复回归检查 |

完整入口与兼容性边界见 [ENTRYPOINTS](docs/ENTRYPOINTS.md)，结构与原实现对应关系见 [METHOD](docs/METHOD.md)。

## 复现的准确范围

`vqa/` 保留当前方法实现；仅为模型路径添加可配置入口。新增 `tessfold/` 去除了对历史结果目录的启动依赖，默认训练使用完整词表 `KL(teacher || student)`、teacher 自己生成的公共回答前缀，不使用 gold 训练，也不加行熵。

可移植入口采用统一的开发集门槛、原生任务指标和精确末批梯度权重；它不是所有历史实验协议的逐字替代。例如旧 UND 全训练有固定 epoch 选模和 field-EM 口径，与新入口不同。需要复现特定历史协议时，请使用保留的对应原脚本并重新生成它所需的登记文件。不同脚本的实际差异见 [REPRODUCIBILITY](docs/REPRODUCIBILITY.md)。

此源码整理经过离线单元测试、合成冻结 decoder 的训练/恢复/评测检查，以及公开清单索引重建检查。**本次整理没有重新进行真实 8B / UND 全量训练，也不以合成测试声明论文任务得分可复现。** 硬件、依赖、数据版本及数值后端仍影响结果。

## 发布前检查

```bash
git status --short
.venv/bin/python scripts/audit_source_repo.py
# 初始化已在本地完成。实际发布时填写自己的远程地址：
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git commit -m "Add TessFold reproducible source"
git push -u origin main
```

引用 MetaCompress、MonkeyOCRv2 和相关工作见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) 与 [BibTeX](docs/references.bib)。本整理不代替权利人决定新增代码的对外开源授权；项目授权状态也在该文件中说明。
