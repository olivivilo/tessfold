"""Measured operator semantics and corrected vision pilot, standalone report."""
from scripts.tessfold_repair.common import ROOT,load,save,now
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    r=load(BASE/'attention_semantics.json');assert len(r['rows'])==8
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=BASE/'figures';directory.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
    for ax,dtype in zip(axs,['torch.bfloat16','torch.float16']):
        rows=[x for x in r['rows'] if x['dtype']==dtype]
        for key,label in [('default_mask','Default next=0 + mask'),('explicit_bidirectional','Explicit bidirectional + mask'),('no_mask','No mask')]:
            ax.plot([x['length'] for x in rows],[x['variants'][key]['full_rms'] for x in rows],marker='o',label=label)
        ax.set(xlabel='Sequence length',ylabel='RMS error vs manual full attention',yscale='log',title=dtype.replace('torch.',''));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(directory/f'attention_window_semantics.{ext}',dpi=180)
    plt.close(fig)
    old=load(BASE/'fused_vision_pilot/complete.json');new=load(BASE/'bidirectional_vision_pilot/complete.json')
    doc=['# Ascend视觉融合注意力的范围缺省：直接算子证据与修复验证','',f'UTC：{now()}。','',
        '**入口**：`sbatch scripts/tessfold_accel/attention_semantics.sh`；本报告：`.venv/bin/python -m scripts.tessfold_accel.attention_semantics_report`。','',
        '[原始算子数值](../artifacts/tessfold_acceleration_20261115/attention_semantics.json) · [独立模型先导](../artifacts/tessfold_acceleration_20261115/bidirectional_vision_pilot/complete.json) · [误差图PDF](../artifacts/tessfold_acceleration_20261115/figures/attention_window_semantics.pdf)。','',
        '## 问题是什么','',
        'UND的视觉塔在同一图像内部使用双向注意力：一个图块可以读取前后任意图块的信息。官方作者的Ascend分支调用`npu_prompt_flash_attention`时传入图像分段mask，但没有显式指定`next_tokens`。本地安装接口的默认值是0；文档说明，sparse_mode0在**不传mask**时忽略pre/next，而在**传mask**时会利用这些范围做稀疏计算。若范围与mask不一致，不能假设算子一定执行完整双向注意力。','',
        '因此首次作者原调用与eager差异不能全部归因于浮点舍入。这里没有修改官方文件或模型权重；独立候选在同样mask下显式设`pre_tokens=next_tokens=2147483647`，确保范围容纳所有同图位置。不同图像间的mask仍保留。','',
        '## 如何直接验证','',
        '固定随机Q/K/V，12 heads、head_dim64，BF16和FP16各测试128/257/513/1024长度。手动执行QK、缩放、FP32 softmax转原dtype、AV，分别构造完整双向、严格下三角和64/128/256大小的块因果参考。相同Q/K/V分别调用默认mask版、显式双向mask版、不传mask版。无需任何模型或任务答案，直接量化输出RMS差异。','',
        '初版诊断预设“默认调用等于严格因果”的断言失败，原日志与原源文件保留；扩大参考后发现行为取决于kernel分块，不能笼统称为逐token因果注意力。','',
        '|dtype|长度|默认mask对完整RMS|显式双向对完整RMS|无mask对完整RMS|','|---|---:|---:|---:|---:|']
    for row in r['rows']:
        v=row['variants'];doc.append(f'|{row["dtype"]}|{row["length"]}|{v["default_mask"]["full_rms"]:.7f}|{v["explicit_bidirectional"]["full_rms"]:.7f}|{v["no_mask"]["full_rms"]:.7f}|')
    doc+=['','128长度下三种调用接近，说明小尺寸sanity check可能漏掉问题。1024长度下，默认调用的输出接近128块大小的块因果参考，而显式双向/无mask接近完整注意力。BF16下，默认对完整RMS约0.066886，但对128块因果约0.000396；显式双向对完整约0.000253。257/513非整块长度的行为不能概括为固定128块因果，因此这里不声称推导出了全部硬件tiling规则。','',
        '这些证据表明：在当前Ascend/CANN/torch_npu组合和调用方式下，默认范围会影响实际允许计算的块，显式双向修复了主要语义差异。它不是对所有设备、全部版本或作者训练实现的通用结论。即使范围修正，融合与eager仍有FP16/BF16舍入差别，不能称Bit-Exact。','',
        '## 对实际冻结UND的影响','',
        '固定56QA、原encoder/merger/LLM及P参数全部不变。原作者调用和显式双向分别对原eager路径做独立对照，参数和问题不按结果调整。','',
        '|调用|encoder中位加速|REF逐题均分变化pp|REF生成变化|是否通过原质量/速度门槛|','|---|---:|---:|---:|---|',
        f'|作者原调用|{old["median_encoder_speedup"]:.3f}×|{old["ref_mean_qa_delta_pp"]:+.4f}|{old["token_changes"]["REF"]}/56|False|',
        f'|显式双向|{new["median_encoder_speedup"]:.3f}×|{new["ref_mean_qa_delta_pp"]:+.4f}|{new["token_changes"]["REF"]}/56|{new["expansion_gate_passed"]}|','',
        '原调用最终视觉特征cosine中位约0.9515；显式双向56题最终特征cosine中位约0.999962，但逐位不等，最大绝对误差中位0.375、最大1.1015625。原门槛是encoder中位加速≥1.5倍且REF均分降幅≤2pp。显式双向通过后才进入23565QA×REF及四组压缩器的完整独立正式评测，另2700次时延+450次内存。正式结果未完成前不把56题趋势写成最终准确率。','',
        '## 对TessFold结论的边界','',
        '严格eager主实验保留：其94260条新旧压缩回归全部相同。显式双向是独立数值部署候选，所有任务分数重新生成。比较压缩收益时REF与学生使用同一显式双向前端，避免把融合encoder加速归因于P。若要比较eager整体部署和双向整体部署，则必须同时展示质量变化和完整冷启动测量。','',
        '参考源：本地torch_npu `_op_plugin_docs.py`和codegen schema的路径、SHA256与原文摘录见`bidirectional_attention_hypothesis.json`；原官方Ascend forward见`models/MonkeyOCRv2-B-Und/modeling_monkeyocrv2_vision.py`，本轮未修改。']
    (ROOT/'docs/TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md').write_text('\n'.join(doc)+'\n')
    print('docs/TESSFOLD_ATTENTION_WINDOW_DIAGNOSIS.md')

if __name__=='__main__':main()
