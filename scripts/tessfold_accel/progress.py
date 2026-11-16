"""Evidence-driven live report for compilation, prefix and native8B extension."""
import collections,json,subprocess
from pathlib import Path
import numpy as np
from scripts.tessfold_repair.common import ROOT,load,save,now
from scripts.tessfold_accel.live_io import live_lines as lines
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'

def main():
    queue=subprocess.run(['squeue','-h','-u','jchen524','-o','%i|%j|%T|%M|%R'],capture_output=True,text=True).stdout.strip()
    doc=['# TessFold 第二轮加速、KV 修复与新版8B迁移实验','',f'更新时间 UTC：{now()}。','',
        '**总入口**：`.venv/bin/python -m scripts.tessfold_accel.progress`。本报告仅从实际日志和完成标志重建；“已提交/正在运行”不等于完成。原20261114/修复20261115实验、官方UND和原生成器权重均保留。','',
        '## 目标和版本边界','',
        '依据tessfold.md继续减少压缩开销、验证视觉前缀复用、扩大数值及任务验证，并将约7B扩展落实为新版Qwen3-VL-8B上的压缩器训练。UND工程优化保持零重训；新版8B使用自己的视觉编码器和对齐，冻结全部基模参数，只训练新增P生成器。两者不混用分数。','',
        '## 编译切分与纯压缩开销','',
        '原实现每次重新生成/整数排序网格索引（Ascend回退CPU），Python循环切分，并构造稠密P。新严格路径缓存官方2×2block几何；C++编译保留FP32积分图、NumPy pairwise求和和cut/tie顺序；对小于2^24的唯一代表索引以FP32精确排序避免int64回退；保留原FP32 softmax与GEMM。编译参数禁止fast-math和FMA重结合。','']
    cpu=load(BASE/'native/cpu_partition_tests.json');doc+=[f'CPU独立原NumPy对照：{cpu.get("cases",0)}例，分区不一致{cpu.get("mismatches","未完成")}；含近均匀、lognormal和单行/列案例。入口`.venv/bin/python -m scripts.test_tessfold_native_partition`。','']
    for label in ['pilot1','validation1']:
        rr=[r for f in (BASE/label).glob('rows.rank*.jsonl') for r in lines(f)]
        if not rr:continue
        doc+=[f'### {label}','',f'已提交{len(rr)}例；完成标志：{(BASE/label/"complete.json").exists()}。','',
            '|路径|特征不逐位相等|生成token变化|纯压缩wall中位ms|','|---|---:|---:|---:|']
        for b in ['dense_exact','bucket_fp32','bucket_fold','hard_linear']:
            a=[r['variants'][b] for r in rr]
            doc.append(f'|{b}|{sum(not x["feature_equal"] for x in a)}|{sum(not x["generation_equal"] for x in a)}|{np.median([x["compressor_wall_ms"] for x in a]):.4f}|')
        doc+=['','pure wall是预热后重复调用整个压缩函数（含host传输、CPU分区及NPU聚合），不含merger。分阶段同步另单列，不用同步分段之和伪装无扰动总耗时。bucket为O(N)支持量候选，改变归约舍入且当前调度开销偏高；未推广为严格路径。Hard是独立模型变体。','']
    doc+=['### 图重放与真正线性聚合候选','',
        'NPU图捕获score与readout，中间保留精确CPU切分。捕获/编译费用和每shape计划常驻内存独立记录；不把首次开销隐藏为无条件冷启动优势。默认完整效率表使用无需图捕获的编译严格路径。','',
        '|试验|例数|完整结束|纯压缩wall中位ms|最小/最大ms|','|---|---:|---|---:|---|']
    for label in ['graph_full1','graph_full2','sparse2','packed1','host_pool1','hard_graph1']:
        rr=[r for f in (BASE/label).glob('rows.rank*.json') for r in load(f,[])]
        if not rr:continue
        v=[r['pure_compressor_ms'] for r in rr]
        doc.append(f'|{label}|{len(rr)}|{(BASE/label/"complete.json").exists()}|{np.median(v):.4f}|{min(v):.4f}/{max(v):.4f}|')
    doc+=['','sparse1最初诊断出现非有限值，未生成成功标志；保留失败日志。保留捕获常量并更换归约调用后，sparse2独立检查有限值、重复运行与分数，发现部分index_add非确定归约，未作为Bit-Exact推广。packed1使用确定性的局部补齐GEMM，host_pool1使用CPU SIMD并计入双向传输，均未获得优于严格路径的整体收益，保留为负结果。','',
        '## 完整效率复测与全量等价性','']
    n=sum(len(lines(p)) for p in (BASE/'system_exact/single').glob('results.rank*.jsonl'))
    formal=sum(len(lines(p)) for p in (BASE/'formal_equivalence').glob('rows.rank*.jsonl'))
    doc += [f'编译严格路径完整效率：{n}/9450记录，完成并审计：{(BASE/"system_exact/complete.json").exists()}。原90QA/15配置、自然EOS和fixed64各三轮、另1350内存，不改变问题或权重。入口`sbatch scripts/tessfold_accel/system.sh`；完成后`.venv/bin/python -m scripts.tessfold_accel.system_report`。','',
        f'正式全量等价性：{formal}/94260记录（23565QA×Column/Tree×25%/50%）。逐例P、PH和生成token对原完整结果检查，任何失败不自动放宽门槛。入口`sbatch scripts/tessfold_accel/formal.sh`。','',
        '## 视觉KV：定位、修复与限制','',
        '已定位：原prefix停在最后图像placeholder，改为包括官方vision_end/后续分隔符后，已知KLC/REF样例恢复相同；Tree/DeepForm样例仍不同。比较全prompt与单独prefix逐层KV，第一个差异可出现在不同层；不同矩阵形状和attention backend会改变舍入，切换eager也不保证原生SDPA输出一致。不是上一题答案KV污染。','',
        '新fork只复制DynamicCache的层对象，共享只读前缀tensor；DynamicLayer.update通过cat产生新storage，每题检查原cache长度、指针和version均不变。避免deepcopy完整前缀的额外开销。该方法仅支持标准DynamicLayer，不冒充paged-KV服务框架。特征缓存题目输入从原模板精确展开图像placeholder，不再每题重新读图/processor。','',
        '|场景|题数|完整结束|生成token变化|逐题分数变化和|','|---|---:|---|---:|---:|']
    for name in ['REF','Column']:
        p=BASE/'prefix_pilot1'/name;rr=[r for f in p.glob('rows.rank*.jsonl') for r in lines(f)]
        if rr:doc.append(f'|{name}|{len(rr)}|{(p/"complete.json").exists()}|{sum(not r["tokens_equal"] for r in rr)}|{sum(r["cached_score"]-r["reference_score"] for r in rr):.4f}|')
    doc+=['','prefix_pilot1包含24个真实10问页及两个历史失败页；不是全体8任务。若出现差异，不能作为严格无损KV推广；需要单独全量任务影响评估，公平比较REF与Column。Tree不能跨不同问题复用问题相关视觉前缀。两种prefix切点在因果意义上均合法，延长边界改变了kernel形状并修复某个已知样例，不代表发现了原切点的跨题上下文污染。','',
        '## Qwen3-VL-8B压缩迁移结构','',
        '此前仅有未压缩原生小验证；本轮已新增可训练P。插入点是原生visual merger之后、text decoder之前，最终4096维视觉特征和3条DeepStack视觉特征使用同一个P做凸组合。输出长度M=round(rN)，每行非负、和为1，区域互不重叠；保留最高权重代表位置及原生三维MRoPE坐标，问题/答案坐标保持原值。NativeSpatialRouter接收4096维特征，LayerNorm→Linear(4096,96)→空间depthwise/pointwise卷积→打分；Tree另用4096维问题embedding及可训练问题门，Column不看问题。','',
        '这与UND的768维pre-merger路径显著不同：8B本轮不节省其视觉encoder或原生merger计算，也不声称旧UND对齐能直接移植。身份路径已对8道原生56题组的官方HF token验证，并测试两个预算/两结构的KL反向梯度，基模参数梯度始终为空。','',
        '### 训练协议和先导门槛','',
        '训练图像与固定56题验证按document_id互斥，复用原20k训练池；先导512题优先每图一题、各任务平衡。Teacher为同一冻结8B的greedy短答案（64token），使用完整词表KL、相同teacher响应前缀，不拿gold作训练label。AdamW、weight_decay=0、clip=5、10%warmup后cosine至0.1LR；记录原gradient norm和clip multiplier，保存optimizer断点。默认batch32/3epochs，LR3e-4；未过门槛两结构另试1e-4和1e-3。','',
        '扩大训练要求同一个先导checkpoint同时满足：dev KL≤0.95×initial，且生成宏分至少提高0.25点（已接近teacher时允许teacher−0.25内）。只看固定dev，不按正式测试追选。全量训练使用原20k、4epochs及通过门槛的LR/结构，基模全冻结。','',
        '|训练版本|结构/预算|训练日志step数|完成|初始dev KL/分数|最终dev KL/分数|先导门槛|','|---|---|---:|---|---|---|---|']
    for label in ['qwen_pilot1','qwen_pilot_low','qwen_pilot_high','qwen_pilot_stable','qwen_full_pass1','qwen_full_tree25','qwen_full_stable50']:
        for out in sorted((BASE/label).glob('*_r*')):
            init=load(out/'initial.summary.json');end=load(out/'complete.json');epochs=sorted(out.glob('epoch*.summary.json'),key=lambda p:int(p.name.split('.')[0][5:]))
            last=load(epochs[-1]) if epochs else {}
            def val(r):return f'{r["kl"]:.4f}/{r["macro"]:.3f}' if r else '尚未完成'
            doc.append(f'|{label}|{out.name}|{len(lines(out/"training.jsonl"))}|{bool(end)}|{val(init)}|{val(last)}|{end.get("pilot_gate_passed","待评估")}|')
    dataset=load(BASE/'dataset_audit.json')
    if dataset:
        doc+=['','### 实际数据规模','', '|划分|QA|图像|任务分布|','|---|---:|---:|---|']
        for name,r in dataset['splits'].items():doc.append(f'|{name}|{r["questions"]}|{r["images"]}|{json.dumps(r["by_task"],ensure_ascii=False)}|')
        doc+=['','训练/验证/正式评测的document_id交集均为0；[逐文件签名与审计](../artifacts/tessfold_acceleration_20261115/dataset_audit.json)。']
    doc+=['','StableColumn是显式结构试验：支持区域仅由图像detail决定并在训练时保持不受score扰动，训练局部权重和温度；用于排查detached切分随score变化带来的不稳定，不冒充原分区规则的无损优化。通过先导门槛后才进入20k。','',
        '8B开发集七任务的DeepForm/KLC为DUE corpus F1；旧UND共享完整benchmark为field EM，不能把两者宏平均直接相减。最终8B正式评测应同时给出指标口径及未压缩teacher；小验证分数不是正式压缩成绩。','',
        '微批训练效率试验约1.46–1.50倍更快，但一个固定样例对的KL差0.050002略超预设0.05门槛，因此没有换入正式训练，也未为提高速度放宽门槛；失败记录保留。','']
    ceiling=load(BASE/'system_exact/single/amdahl_front_ceiling.json')
    if ceiling:
        v=ceiling['natural_eos'];doc += ['## 前端限制与作者融合注意力候选','',f'本次90QA冷启动条件下，REF前端（processor+完整encoder）平均{v["reference_front_mean_ms"]:.2f}ms、TTFT平均{v["reference_ttft_mean_ms"]:.2f}ms；即使后端全部计算降到零、前端保持不变，理想TTFT加速上限也仅{v["ideal_zero_postfront_ttft_upper_bound"]:.3f}倍。这是有限工作负载的Amdahl上界，不是实测加速或所有设备通用结论。','']
    fused=load(BASE/'fused_vision_pilot/complete.json')
    doc+=['在相同冻结视觉权重上新增独立候选：临时绑定作者已有VisionAscendAttention.forward（npu_prompt_flash_attention），替代此前部署选择的eager V2，不改官方源码。调用未显式指定next_tokens，后续本地接口文档与双向试验提示需检查注意力范围；不能假设只改变浮点舍入。须独立报告误差、生成和分数，不能继承Bit-Exact；只有encoder中位加速≥1.5倍且REF开发集分降幅≤2点才考虑进一步全量评测。','',
        f'融合注意力先导：`{json.dumps(fused,ensure_ascii=False) if fused else "尚未完成"}`。','',
        '## 最终验证的补充对照','',
        '8B正式主表为23565题×REF及四组dev所选压缩器，共117825条；另在预先hash固定的512图/2404题上评测四个初始P，共9616条，用同题训练前后对照检验泛化，正式总127441条。主表不混入这个子集。分阶段内存另1350次独立1ms采样，不并入2700条时延；独立memory pass必须对natural timing逐token相同。先导15次技术检查不参与模型选择。','',
        '[最终验证协议](TESSFOLD_FINAL_VALIDATION_PROTOCOL.md) · [与tessfold.md的逐项目标对照](TESSFOLD_TARGET_GAP.md)。','',
        '## 当前作业','', '```text',queue or '队列为空','```','',
        '## 尚不能写进论文的结论','',
        '在实测前不写亚毫秒完整压缩、无损O(N)soft替换、KV缓存必然无损、4倍并发吞吐或新版8B必然2.2–3.1倍加速。全请求峰值仍须保留encoder既有水位，解析FLOPs只算解析值。后续新增结果以各阶段complete和审计为准。']
    if (BASE/'all_stages_complete.json').exists():
        doc[1:1]=['','**本轮全部13阶段及最终审计已完成。** [最终结果速读](TESSFOLD_FINAL_RESULTS.md) · [目标差距](TESSFOLD_TARGET_GAP.md) · [8B及融合论文表](TESSFOLD_PAPER_EXTENSION_TABLES.md)。下文所有结论以完成的原始记录为准。']
    report=ROOT/'docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md'
    marker='\n\n## 已完成阶段的实测汇总'
    previous=report.read_text() if report.exists() else ''
    retained=marker+previous.split(marker,1)[1] if marker in previous else ''
    report.write_text('\n'.join(doc)+'\n'+retained)
    save(BASE/'progress.json',dict(time=now(),system_records=n,formal_records=formal,queue=queue))
    print(json.dumps(dict(report='docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md',system_records=n,formal_records=formal)))

if __name__=='__main__':main()
