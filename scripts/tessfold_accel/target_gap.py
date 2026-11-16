"""Measured progress against tessfold.md; never fill targets as results."""
import argparse,json
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    p=argparse.ArgumentParser();p.add_argument('--publish-complete',action='store_true');a=p.parse_args()
    core=['system_exact','formal_equivalence','prefix_formal','interleaved','multi_deployment','batched_serving','qwen_formal','qwen_timing','qwen_memory','hard_formal','bidirectional_formal','bidirectional_timing','bidir_interleaved']
    stages={name:(BASE/name/'complete.json').exists() for name in core};complete=all(stages.values()) and bool(load(BASE/'final_audit.json').get('all_checks_passed'))
    if a.publish_complete:assert complete
    status='全部预设核心实测及最终审计完成' if complete else '仍在执行；已完成项与未完成项如下分开列出'
    rows=[('原UND及已训练P生成器零重训','保持全部原权重冻结；仅新版原生8B扩展训练新增P。源码/数据/checkpoint签名校验通过。','已落实'),
        ('新旧数学/生成100%等价','94,260条：23,565QA×两方法×25%/50%，全部P、PH和生成token一致。','已落实' if stages['formal_equivalence'] else '执行中'),
        ('soft完整压缩<0.8ms且O(N)','严格实现保留FP32稠密聚合。448例中位2.20ms；图重放8例中位1.57ms，未达到目标。分段等O(N)候选已出现特征/生成变化或速度退化。','未达成'),
        ('hard快速路径','不构造稠密P，24例图重放中位1.04ms、最小0.687ms。它是独立hard模型，不能继承soft无损或把最小值写成常规耗时。完整23,565QA另测。','完整评测完成' if stages['hard_formal'] else '完整评测执行中'),
        ('四维效率测量','完整9450条UND记录；阶段耗时/实际TTFT/独立内存与逻辑KV/同页多问，另2520条交错计时控制时间漂移。','已落实' if stages['system_exact'] and stages['interleaved'] else '执行中'),
        ('同页K=1/2/5/10真实问答摊销','24真实多问页，3方法、4场景、3轮，864页面运行/8640题次。REF同样获得缓存。纯特征路径完全相同，prefix另列数值变体。','已落实' if stages['multi_deployment'] else '执行中'),
        ('视觉KV缓存必然无损','两个因果合法切点均可能受分块prefill浮点路径影响；不是答案跨题污染。完整REF/Column各23,565题用于检查任务影响。','任务影响评估完成' if stages['prefix_formal'] else '任务影响评估执行中'),
        ('视觉KV减75%即四倍服务吞吐','这种推导不成立。已测真实同步batch1/2/4/8；long fixed64 batch8：REF1.385q/s，Column25%1.712q/s，约1.236倍。不是最大异步并发容量。','未达成四倍；已给实测'),
        ('约7B新版模型压缩迁移','采用更新原生Qwen3-VL-8B-Instruct，全部基模冻结；P放在其merger后，并同步压缩3条DeepStack。不是把UND视觉塔嫁接到8B。四组20k×4epochs，仅由dev选checkpoint。','正式训练评测完成' if stages['qwen_formal'] else '训练或正式评测中'),
        ('8B冷请求TTFT2.2–3.1倍','需看90QA实测及其置信区间；原文的预期不能预填成结果。原生8B与UND的插入边界和视觉前端不同。','已测，见下文' if stages['qwen_timing'] else '待正式测速'),
        ('8B压缩器训练确实改善泛化','除完整正式teacher/四模型外，预先固定512图/2404QA，逐模型比较初始P与dev所选P，共额外9616条生成。该子集不参与选择。','已完成' if stages['qwen_formal'] else '已接入，待训练结束评测'),
        ('实际下游内存与论文表','UND完整分阶段内存已测；8B另1350次独立1ms内存与完整时延逐token检查，保留encoder水位，提供实测LaTeX，不覆盖旧PDF。','已完成' if stages['qwen_memory'] else 'UND完成，8B待完成')]
    doc=['# tessfold.md目标对照、实际改进与剩余差距','',f'UTC：{now()}。**{status}**。','',
        '**执行入口**：`.venv/bin/python -m scripts.tessfold_accel.controller status|resume`。结果重建：`.venv/bin/python -m scripts.tessfold_accel.finish_report`；最终审计：`.venv/bin/python -m scripts.tessfold_accel.final_audit`。','',
        '[完整实验报告](TESSFOLD_ACCELERATION_AND_8B_RESULTS.md) · [UND效率明细](TESSFOLD_COMPILED_SYSTEM_RESULTS.md) · [实测论文表](TESSFOLD_PAPER_EFFICIENCY_TABLE.md)。','',
        '## 逐项目标对照','', '|参考方案期望|实际结构、证据和边界|状态|','|---|---|---|']
    doc += [f'|{goal}|{actual}|{state}|' for goal,actual,state in rows]
    fused=load(BASE/'fused_vision_pilot/complete.json')
    if fused:
        doc+=['','## 为什么不直接启用更快的视觉注意力','',
            f'作者提供的Ascend融合注意力在56题固定dev上的encoder中位加速{fused["median_encoder_speedup"]:.3f}倍，但REF逐题均分变化{fused["ref_mean_qa_delta_pp"]:+.3f}个百分点，未通过预设质量门槛。原调用的最终特征余弦中位约0.9515；随后检查到传mask而不指定next_tokens的语义风险，显式双向版本独立验证。不能把这个原调用解释为已保证数学相同、只有舍入差别。全部记录保留，不推广。另测试保留原舍入边界的query分块eager，只有通过特征及生成检查才考虑扩展。']
    bidir=load(BASE/'bidirectional_vision_pilot/complete.json')
    if bidir:
        doc+=['','## 显式双向融合：已通过先导的独立改进','',
            f'本地PFA接口文档提示传入mask时pre/next会限制注意力范围。作者调用默认next_tokens=0；独立版本显式设置双向INTMAX后，固定56题encoder中位加速{bidir["median_encoder_speedup"]:.3f}倍、REF均分变化{bidir["ref_mean_qa_delta_pp"]:+.3f}pp，通过原≥1.5倍且降幅≤2pp的先导门槛，已扩大为23565QA×5配置独立正式评测及3150次独立时延/内存。该变体仍不逐位相同，不能覆盖严格eager成绩；全量结果及配对CI见完整报告。']
    selected=load(BASE/'qwen_selected_runs.json')
    if selected:
        doc+=['','## 8B锁定的开发集选择','', '|配置|最终采用epoch|checkpoint|','|---|---|---|']
        for r in selected['runs']:doc.append(f'|{r["id"]}|{r["stage"]}|`{r["checkpoint"]}`|')
    bidir_stats=load(BASE/'bidir_interleaved/statistics.json')
    if bidir_stats:
        doc+=['','## 同题交错直接对照：原eager与显式双向部署','',
            '90题的每个题目/模式/重复内随机六配置，两轮，所有前端工作计入。以下是配对均值比。跨后端速度还须看完整质量，不能当作Bit-Exact收益。','',
            '|参考|候选|模式|E2E加速|95%页面CI|TTFT加速|','|---|---|---|---:|---|---:|']
        for v in bidir_stats['comparisons']:
            if v['mode']=='natural_eos' and v['reference_backend']=='EAGER' and v['reference_method']=='REF' and v['candidate_backend']=='BIDIR':
                ci=v['e2e_document_bootstrap_95_ci'];doc.append(f'|EAGER/REF|BIDIR/{v["candidate_method"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|{v["ttft_speedup"]:.3f}×|')
    timing=load(BASE/'qwen_timing/statistics.json')
    if timing:
        doc+=['','## 新版8B实测是否达到速度预期','', '|模型|模式|同题E2E加速|同题TTFT加速|TTFT 95%页面CI|','|---|---|---:|---:|---|']
        for v in timing['comparisons']:
            if v['stratum']=='all':
                ci=v['ttft_bootstrap_95_ci'];doc.append(f'|{v["model"]}|{v["mode"]}|{v["e2e_speedup"]:.3f}×|{v["ttft_speedup"]:.3f}×|[{ci[0]:.3f},{ci[1]:.3f}]|')
    full_bidir=load(BASE/'bidirectional_formal/complete.json')
    if full_bidir:
        bs=load(BASE/'bidirectional_statistics.json')
        doc+=['','## 双向融合的全量质量变化','',
            '同冻结UND权重、同23565题独立生成。共享fieldEM宏分差，相对原eager的任务分层文档bootstrap CI；CI跨零不等于数值或统计等效。','',
            '|配置|完整macro|相对eager差pp|95%页面CI|','|---|---:|---:|---|']
        for model,value in full_bidir['shared_fieldEM_results'].items():
            delta=bs['quality_vs_original_eager'][model];ci=delta['shared_fieldEM_macro_delta_task_stratified_document_bootstrap_95_ci']
            doc.append(f'|{model}|{value["macro"]:.4f}|{delta["shared_fieldEM_macro_delta"]:+.4f}|[{ci[0]:.4f},{ci[1]:.4f}]|')
    doc+=['','## 当前核心阶段','', '```json',json.dumps(stages,ensure_ascii=False,indent=2),'```','',
        '未完成阶段没有最终分数；已提交作业不等于通过。亚毫秒soft、全请求显存降75%、四倍并发和8B预期加速不会作为无依据的结论写入论文。历史失败、负结果和模型版本保留。']
    path=ROOT/'docs/TESSFOLD_TARGET_GAP.md';path.write_text('\n'.join(doc)+'\n')
    save(BASE/'target_gap.json',dict(time=now(),all_core_and_final_audit_complete=complete,stages=stages,tessfold_source_sha256=sha(ROOT/'tessfold.md'),report=str(path.relative_to(ROOT))))
    if a.publish_complete:
        note='**2026-11-15第二轮TessFold：全部核心实测与最终审计已完成。** [完整实验结果](docs/TESSFOLD_ACCELERATION_AND_8B_RESULTS.md) · [与tessfold.md目标的逐项差距](docs/TESSFOLD_TARGET_GAP.md) · [实测论文表](docs/TESSFOLD_PAPER_EFFICIENCY_TABLE.md)。未达到的亚毫秒soft、4倍并发或预期8B加速按实测如实保留。原模型和历史结果未覆盖。'
        readme=ROOT/'README.md';text=readme.read_text();parts=text.split('\n\n',2)
        if len(parts)>2 and parts[1].startswith('**当前继续任务'):text=parts[0]+'\n\n'+note+'\n\n'+parts[2]
        else:
            first=text.find('\n\n');text=text[:first]+'\n\n'+note+text[first:]
        readme.write_text(text)
        current=ROOT/'docs/CURRENT_PROGRESS.md';text=current.read_text();first=text.find('\n\n');local=note.replace('(docs/','(')
        text=text.replace('## 2026-11-15 第二轮TessFold加速与8B压缩迁移：正在进行','## 历史：第二轮TessFold完成前的执行记录',1)
        text=text[:first]+'\n\n'+local+'\n\n以下运行中描述均为已保留的历史记录；本轮最终状态以本段及最终报告为准。'+text[first:]
        current.write_text(text)
    print(json.dumps(dict(report=str(path),all_complete=complete)))

if __name__=='__main__':main()
