# 入口目录

## 推荐入口：可从空输出目录启动

| 命令 | 用途 |
|---|---|
| `python -m tessfold --help` | 查看全部接口 |
| `python -m tessfold download-model --config CONFIG` | 下载固定来源的所选基模 |
| `python -m tessfold download-data` | 显式下载公开数据；不在安装时运行 |
| `python -m tessfold prepare-cohorts` | 从固定公开转换清单按索引重建三组划分 |
| `python -m tessfold validate --config CONFIG` | 数据与图片互斥、内容和格式校验 |
| `python -m tessfold pilot --config CONFIG --device DEVICE` | 小规模多图训练和开发集检查 |
| `python -m tessfold train --config CONFIG --device DEVICE` | 先导合格后从同种子初始化完整训练 |
| `python -m tessfold evaluate --config CONFIG --device DEVICE` | 读取已锁定 checkpoint，独立生成 teacher/student 答案 |
| `python -m pytest` | 不下载基模的回归检查 |
| `python scripts/audit_source_repo.py` | 检查 Git 上传范围是否混入结果、权重等产物 |

`DEVICE` 为 `cpu` / `cuda` / `npu`。`auto` 自动选择 CUDA 或 CPU。
`pilot`、`train` 支持 `--resume`；`evaluate --stage pilot` 可评估先导选中模型，
正式论文实验应在预先锁定的完整训练模型上运行。

## 保留的原始实验入口

以下源码保留以解释和重建原实验，**不是新机器的即开即跑控制器**。
其中输出路径、Slurm 分区、CANN 路径和已登记实验目录具有历史协议含义。
各自的注册函数会创建运行文件，不能从本源码包推断对应结果或权重已存在。

| 子系统 | 原入口 |
|---|---|
| 原生 8B 训练 | `scripts/tessfold_accel/qwen_training.py` |
| 原生 StableColumn 训练 | `scripts/tessfold_accel/qwen_training_stable.py` |
| 共享缓存并发修复 | `scripts/tessfold_accel/run_qwen_safe.py` |
| 原生 8B 正式/时延评测 | `scripts/tessfold_accel/qwen_formal.py`, `qwen_eval_safe.py` |
| 8B 新增 10%/75% 实验 | `scripts/tessfold_budget1075/{common,run,controller}.py` |
| UND 空间压缩训练 | `scripts/train_compression_lab_v2{,_distributed,_repaired}.py` |
| UND 正式 benchmark | `scripts/benchmark_compression_lab_v2.py` |
| 系统严格推理与接口修复 | `scripts/tessfold_repair/` |
| 精确性/计时/显存/同页多问/真实 batch | `scripts/tessfold_accel/` 对应模块 |
| MetaCompress loss、生成器和基线 | `vqa/meta_compress.py`, `scripts/train_meta_compress*.py` |
| 第一/第二轮生成器搜索 | `scripts/train_compression_lab*.py` 与 `vqa/compression_generators*.py` |
| 早期官方 UND 复测与 LoRA | `scripts/run_official_und.sh` 等；ms-swift 需另装 |

独立的融合视觉、segmented、hard、packed、prefix-cache 等推理候选仍保留，
但不作为新入口的隐式默认后端。必须分别通过数值与输出一致性验证，
不能因为有候选代码就宣称与严格路径 bit-exact 或更快。

原报告生成脚本属于生成报告的**程序**；本仓库没有附这些程序过去生成的报告和分数。
若运行了旧控制器，它可能自动提交 Slurm 作业，请先审阅它的动作与集群配置。
