# 实现结构与训练方法

原 token 为 `X ∈ R^(N×D)`，生成器输出 `P ∈ R^(M×N)`。每行非负且和为 1，因此 `C = PX` 是原 token 的凸组合。`ratio` 表示保留率，`M = max(1,min(N,round(N*ratio)))`；例如 0.1 表示保留 10%，不是减少 10%。Python 的 `round` 使用 ties-to-even。

Tree/Column 先将视觉特征映射到较小的隐空间，使用局部卷积和打分网络计算 token 分数；再结合原始视觉特征的局部 detail 估计生成非空区域，每个输出 token 对应一个区域。区域内对分数做温度 softmax，区域外屏蔽；按代表 token 的原始顺序排序。分区、区域选择及代表 token 都是 detached 离散操作。梯度通过 soft 权重和后续冻结模型传播，不能宣称树分裂完全可微。

UND 的生成器读入 768 维视觉特征和 2048 维对齐特征；pre-merger 实验用零向量占据对齐特征槽位，避免读入后续 merger 输出。Tree 还读入冻结 LLM 的问题 embedding，并学习问题/视觉相关性。它是 decoder-dependent 任务训练，不要求 decoder-free。Column 主要采用列优先的二维切分。对应实现是 `SpatialGenerator('quadtree_question')` 和 `SpatialGenerator('col_adaptive')`。

UND pre-merger：`encoder -> P @ raw -> 官方 merger -> 原 LLM`；post-merger：先获得原始 raw 与 projected 特征，生成 P 后对 projected 特征压缩，再进入原 LLM。二者位置不同，不共享可直接互换的训练权重。

原生 Qwen3-VL-8B 的生成器输入为 merger 后 4096 维特征，隐层默认 96 维。相同 P 同时作用于最终特征与三个 DeepStack 分支；代表 token 保留原始 MRoPE 位置，问题/回答位置也沿用原索引。前端视觉塔和 merger 已执行，因此此位置不节省其计算量。`StableColumn` 使用固定 detail 驱动分区，只训练区域内 soft 权重，是独立结构变体。

训练 loss：teacher 为未压缩模型，先 greedy 生成回答 `y`。teacher 与 student 均在 `prompt + y[:-1]` 的公共前缀上前向，比较预测下一个 token 的完整词表分布：

`L = mean_t sum_v p_teacher(v|prefix_t) log(p_teacher(v|prefix_t)/p_student(v|prefix_t))`。

gold answer 仅用于开发和评测；训练不使用 gold CE，也不把前一题的回答或 KV 传给后一题。所有基模参数冻结，但 student 的 decoder 前向不使用 `no_grad`，否则会切断 P 的梯度。

默认 AdamW、零 weight decay、10% 线性 warmup 后 cosine 到 0.1 LR。默认 clip=5，UND 模板保留 clip=1 的候选配置。训练记录保留裁剪前梯度范数与裁剪比例。完整训练从固定种子重新初始化；先导检查点不作为 full 的起点。开发集 KL 至少降低 5%，生成宏分至少增加 0.25pp 才合格，保留原生实现的 teacher 饱和例外；合格 epoch 中按 dev 宏分最高、KL 最低选择。无合格 epoch 则锁定 initial 并保留失败结果。

该 KL 来自 MetaCompress 的预测保持思想。本项目的空间生成器、公共回答前缀的具体实现和最终默认不加行熵，均应与原论文区分。原实现 `vqa/meta_compress.py` 与 `vqa/decoder_dependent_loss.py` 保留供对照；不能把本仓库全部结构称为 MetaCompress 原版。引用见 `references.bib`。
