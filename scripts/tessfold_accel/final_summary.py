"""Concise final entry, populated only from audited completed experiments."""
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    audit=load(BASE/'final_audit.json');assert audit['all_checks_passed']
    for stage,digest in audit['stage_complete_sha256'].items():assert sha(BASE/stage/'complete.json')==digest
    assert (BASE/'all_stages_complete.json').exists()
    native=load(BASE/'qwen_formal/complete.json');registry=load(BASE/'qwen_selected_runs.json');stats=load(BASE/'bidir_interleaved/statistics.json')
    names={'REF':'未压缩原生8B','Column_r0.25':'Column 25%','Tree_r0.25':'Tree 25%','Column_r0.5':'StableColumn 50%','Tree_r0.5':'Tree 50%'}
    methods=['REF','Column_r0.25','Tree_r0.25','Column_r0.5','Tree_r0.5'];tasks=['DocVQA','InfoVQA','ChartQA','OCRBench','DeepForm','KLC','WTQ','DT-VQA']
    doc=['# TessFold本轮最终结果与阅读入口','',f'结果汇总UTC：{now()}。**全部13阶段完成，最终审计通过，训练与评测任务已退出。**','',
        '本轮已完成原UND零重训工程优化，以及原生Qwen3-VL-8B压缩器四组20k×4epochs训练、独立全量测试。tessfold.md的预期并未全部达到；这里保留实际结果及限制。原模型、历史预测和负结果均未覆盖。','',
        '## 优先阅读','',
        '- [完整实验设计、过程、分数、时延、内存与曲线](TESSFOLD_ACCELERATION_AND_8B_RESULTS.md)','- [tessfold.md逐项目标对照与未达目标](TESSFOLD_TARGET_GAP.md)',
        '- [视觉注意力范围问题：直接算子证据及双向修复](TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md)',
        '- [实际原生8B架构和训练方法](TESSFOLD_NATIVE8B_METHOD.md) · [验证协议](TESSFOLD_FINAL_VALIDATION_PROTOCOL.md) · [环境及入口](TESSFOLD_RUNTIME_REPRODUCTION.md)',
        '- [UND实测论文表](TESSFOLD_PAPER_EFFICIENCY_TABLE.md) · [8B及融合扩展论文表和图](TESSFOLD_PAPER_EXTENSION_TABLES.md)','',
        '## 1. 原UND严格路径：完全保留原压缩结果','',
        '编译切分、缓存几何、避开整数排序CPU回退、消除多余merger。完整23565QA×4配置的94260条P/PH/生成检查全部相同。这里的无损是“新实现对原压缩模型相同”，不是“压缩与未压缩同分”。',
        '原FP32 softmax/GEMM仍保留，因此没有获得逐位相同的O(N)soft替代。448例纯压缩中位约2.20ms；完整90题自然EOS压缩+merger中位Column25%3.42ms、Tree25%4.10ms，原部署约26ms。完整效率9450条、额外交错2520条、864真实同页运行/8640题次、144真实batch均完成。','',
        '纯特征及免重复processor缓存与冷启动逐token相同。K=10公平缓存：REF313.86ms/题、Column235.09ms/题，加速1.335倍，95%页面CI[1.202,1.469]，均计首次摄入。prefix KV在完整测试中仍有token变化，Column共享macro差−0.0044pp、95%CI[−0.1307,+0.1244]；它是独立数值路径，不默认称无损。','',
        'Hard无稠密P读出的23565QA完整macro为38.2395，所有token与历史Hard一致；Soft Tree25%为38.4705，分数接近不等于等效检验。','',
        '## 2. 显式双向融合：本轮重要修复','',
        '作者Ascend分支传入mask却没有指定next_tokens，安装版本默认0。直接Q/K/V试验显示，1024长度下原调用接近受限分块注意力；显式pre/next=INTMAX接近完整双向参考。它仍与eager存在舍入差异，不能替代严格路径的Bit-Exact结论。首次原调用降16.68pp的负结果保留。修正版本56题先导通过后，完整独立评测23565QA×5配置，并额外完成2700条时延/450条独立内存及2160条同题交错测速。','',
        '共享宏分相对原eager变化：REF+0.1133、Column25%+0.2638、Tree25%−0.1956、Column50%−0.1947、Tree50%+0.0983pp，各宏分CI均跨零，不能据此宣称统计等效。具体任务变化见完整报告。','',
        '|同题交错比较，自然EOS|端到端加速|端到端95%页面CI|TTFT加速|TTFT 95%页面CI|','|---|---:|---|---:|---|']
    for r in stats['comparisons']:
        if r['mode']=='natural_eos' and ((r['reference_backend']=='EAGER' and r['reference_method']=='REF' and r['candidate_backend']=='BIDIR') or (r['reference_backend']=='BIDIR' and r['reference_method']=='REF')):
            ci=r['e2e_document_bootstrap_95_ci'];ti=r['ttft_document_bootstrap_95_ci']
            doc.append(f'|{r["reference_backend"]}/{r["reference_method"]}→{r["candidate_backend"]}/{r["candidate_method"]}|{r["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{r["ttft_speedup"]:.3f}×|[{ti[0]:.3f},{ti[1]:.3f}]|')
    doc+=['','同一融合后端的独立内存：REF全流程峰值4.279GiB，25%学生约3.736GiB。旧eager REF约6.410GiB，但两种部署是独立测量和数值路径，不能把融合带来的全部内存收益归因于P。','',
        '## 3. 原生8B：四组训练确实改善独立泛化','',
        '采用完整Qwen3-VL-8B-Instruct及其自己的视觉塔和merger；全部基模冻结，只训练新增空间P。P在原生merger后，同时作用于final视觉特征和3条DeepStack，保留代表MRoPE。它不是UND视觉塔与8B语言模型的嫁接实验。','',
        '每组先通过多图先导，再独立20k×4epochs；仅用文档互斥56QA dev选checkpoint。原生8B正式23565QA，DeepForm/KLC用DUE corpus F1，其余任务原生指标，八任务等权。另保存旧共享fieldEM口径，不能混算。','',
        '|模型|DocVQA ANLS|InfoVQA ANLS|ChartQA|OCRBench|DeepForm F1|KLC F1|WTQ|DT-VQA|原生macro|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        v=native['native_metric_results'][m];doc.append('|'+names[m]+'|'+'|'.join(f'{v["by_task"][t]:.2f}' for t in tasks)+f'|{v["macro"]:.2f}|')
    doc+=['','512图/2404QA独立固定子集的初始P与训练后P逐题配对，不参与选择：','',
        '|模型|初始原生macro|训练后原生macro|改善pp|','|---|---:|---:|---:|']
    for m in methods[1:]:
        v=native['initial_control_results']['comparisons'][m];doc.append(f'|{names[m]}|{v["initial"]["native_macro"]:.2f}|{v["selected"]["native_macro"]:.2f}|{v["native_macro_delta"]:+.2f}|')
    doc+=['','训练后macro在独立子集均改善，但25%保留率仍有较大质量损失；在这轮预定配置中Tree50%最接近teacher，完整macro低3.492个百分点，DocVQA低2.871个百分点。这是结果描述，不按测试结果回头选择其他epoch。','',
        '新版8B完整冷请求25%保留率TTFT仅约1.14–1.15倍、端到端约1.08倍，未达到参考2.2–3.1倍。实际原始LLM视觉token中位UND4960、原生8B945；8B本身先做原生merger，不能仅凭参数量预言更大压缩收益。前端均值132.10ms/TTFT220.85ms，保持前端不变的理想零后端上界约1.672倍。','',
        '## 4. 审计、产物及保留限制','',
        '最终330项源引用签名检查通过；四组12500步连续，LR/clip和实际optimizer状态吻合；初始+4epochs共1120条dev指标重算通过。1350条8B自然时延与正式输出相同，1350条独立内存与时延输出相同；双向融合1800条natural时延/内存与正式输出相同。全部trace签名及并行续跑前26990/34132条预测原样保留的检查通过。','',
        '- [最终审计JSON](../artifacts/tessfold_acceleration_20261115/final_audit.json) · [完成标志](../artifacts/tessfold_acceleration_20261115/all_stages_complete.json)',
        '- [完整分数及初始对照](../artifacts/tessfold_acceleration_20261115/qwen_formal/complete.json)',
        '- [逐题gold/teacher/student损失对照CSV](../artifacts/tessfold_acceleration_20261115/error_analysis/largest_losses.csv)',
        '- [完整测试/独立对照PDF图](../artifacts/tessfold_acceleration_20261115/figures/native8b_quality_and_controls.pdf)',
        '- [逐epoch损失、生成分数和梯度诊断](TESSFOLD_8B_TRAINING_DIAGNOSTICS.md)','',
        '没有达到严格soft<0.8ms、四倍实际吞吐或8B冷请求2.2–3.1倍加速。视觉KV减少不等于相同幅度的全设备峰值减少。固定dev较小、训练只有单seed；KLC/DeepForm原生8B基线本身偏低；teacher训练前缀35/20000条截断；这些限制仍保留。旧结果不删除，新数值变体不继承旧准确率。','',
        '结果重建：`.venv/bin/python -m scripts.tessfold_accel.finish_report --require-complete`；本页：`.venv/bin/python -m scripts.tessfold_accel.final_summary`。已完成阶段不需要重新提交训练。']
    path=ROOT/'docs/TESSFOLD_FINAL_RESULTS.md';path.write_text('\n'.join(doc)+'\n')
    save(BASE/'final_summary.json',dict(time=now(),report=str(path.relative_to(ROOT)),final_audit_sha256=sha(BASE/'final_audit.json'),complete_sha256=sha(BASE/'all_stages_complete.json')))
    print(path)

if __name__=='__main__':main()
