"""Report only committed real experiment records; never infer an unrun score."""
import csv,json,datetime
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/compression_lab'
def read(p):
 if not p.exists():return []
 out=[]
 for line in p.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:break
 return out
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def fmt(x):return '—' if x is None else f'{x:.3f}'
def main():
 protocol=load(BASE/'protocol.json');configs=[load(ROOT/p) for p in load(BASE/'screen_configs.json')]+[load(ROOT/p) for p in load(BASE/'followup_configs.json')]+[load(ROOT/p) for p in load(BASE/'pre_configs.json')]+[load(ROOT/p) for p in load(BASE/'expanded_configs.json')]+[load(ROOT/p) for p in load(BASE/'full_configs.json')];records=[];answers=[]
 baselines={}
 for position,folder in [('post-merger','baselines'),('pre-merger','baselines_pre')]:
  for ratio in [.1,.25,.5]:
   vals=[]
   for kind in ['mean','selection']:
    s=load(BASE/folder/f'{kind}_r{ratio:g}'/'validation.fixed.summary.json')
    if s:vals.append(s['student_macro'])
   if len(vals)==2:baselines[(position,ratio)]=max(vals)
 for cfg in configs:
  out=ROOT/cfg['output'];initial=load(out/'validation.initial.summary.json');final=load(out/f'validation.epoch{cfg["epochs"]}.summary.json');train=read(out/'training.jsonl')
  a=load(out/'train_audit.initial.summary.json');b=load(out/f'train_audit.epoch{cfg["epochs"]}.summary.json')
  status='完成' if (out/'complete.json').exists() else '基础设施失败' if (out/'failure.txt').exists() else '运行中' if initial or train else '未开始'
  drop=(initial['kl']-final['kl'])/max(initial['kl'],1e-12) if initial and final else None
  gain=final['student_macro']-initial['student_macro'] if initial and final else None
  baseline=baselines.get((cfg.get('insertion','post-merger'),cfg['ratio']))
  passed=status=='完成' and drop>=.05 and gain>=2 and baseline is not None and final['student_macro']>baseline
  rec={'name':cfg['name'],'phase':cfg.get('phase','initial_screen'),'kind':cfg['kind'],'ratio':cfg['ratio'],'status':status,'initial_score':initial.get('student_macro'),'final_score':final.get('student_macro'),'teacher_score':initial.get('teacher_macro'),'gain':gain,'initial_kl':initial.get('kl'),'final_kl':final.get('kl'),'kl_drop_percent':100*drop if drop is not None else None,'train_audit_initial_kl':a.get('kl'),'train_audit_final_kl':b.get('kl'),'fixed_best':baseline,'screen_gate':bool(passed),'steps':len(train),'max_gradient':max((r['grad_norm'] for r in train),default=None),'median_clip':sorted(r['clip_multiplier'] for r in train)[len(train)//2] if train else None,'total_update_l2':sum(r['update_l2'] for r in train)}
  records.append(rec)
  for path in sorted(out.glob('validation.*.jsonl')):
   for row in read(path):answers.append({'run':cfg['name'],'ratio':cfg['ratio'],**row})
 if records:
  with (BASE/'screen_summary.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 if answers:
  with (BASE/'answers.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(answers[0]));w.writeheader();w.writerows(answers)
 completed=sum(x['status']=='完成' for x in records);failed=sum(x['status']=='基础设施失败' for x in records)
 summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned_runs':len(records),'planned_screen_runs':len(load(BASE/'screen_configs.json')),'planned_followup_runs':len(load(BASE/'followup_configs.json')),'planned_placement_runs':len(load(BASE/'pre_configs.json')),'planned_expanded_runs':len(load(BASE/'expanded_configs.json')),'planned_full_runs':len(load(BASE/'full_configs.json')),'completed':completed,'failed':failed,'screen_passed':[x['name'] for x in records if x['screen_gate']],'records':records}
 summary_tmp=BASE/f'summary.{__import__("os").getpid()}.tmp';summary_tmp.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');summary_tmp.replace(BASE/'summary.json')
 stages=load(BASE/'stages.json').get('ratios',{})
 outcomes=['## 当前阶段结论','', '下表以 `stages.json` 的完整阶段门槛为准；逐方案表中的“通过初筛”不等于通过独立确认。','', '|token保留率 / 压缩倍数|最终阶段|结论|','|---|---|---|']
 stage_text={'no_pilot_pass':'初筛未通过，未扩大训练','expanded_failed_no_large_training':'扩大训练未通过，未启动完整训练','confirmation_failed_no_large_training':'独立确认未通过，未启动完整训练'}
 for ratio,state in stages.items():
  outcomes.append(f'|{float(ratio):.0%} / {1/float(ratio):g}×|`{state.get("stage","waiting")}`|{stage_text.get(state.get("stage"),"详见阶段状态")}|')
 outcomes+=['','### 扩大训练：672张图像 / 1,801道题','', '以下分数来自用于模型选择的28张图像/28道题；不能视为独立确认成绩。固定终点为第8epoch。','', '|保留率|seed|初始得分|训练后得分|验证KL初始→最后|','|---|---:|---:|---:|---|']
 for ratio,state in stages.items():
  for config_path,gate in zip(state.get('expanded_configs',[]),state.get('expanded_gates',[])):
   cfg=load(ROOT/config_path);a=gate['initial'];b=gate['final']
   outcomes.append(f'|{float(ratio):.0%}|{cfg.get("seed","—")}|{fmt(a.get("student_macro"))}|{fmt(b.get("student_macro"))}|{fmt(a.get("kl"))} → {fmt(b.get("kl"))}|')
 outcomes+=['','### 独立确认：56张图像 / 103道题','', '|保留率|seed|训练后得分|相对同seed初始化增益|增益95%置信区间|最强固定控制|通过|','|---|---:|---:|---:|---|---:|---|']
 confirmation_seen=False
 for ratio,state in stages.items():
  for check in state.get('confirmation_checks',[]):
   confirmation_seen=True;paired=check['paired_vs_initial'];ci=paired['cluster_bootstrap_95ci_points']
   outcomes.append(f'|{float(ratio):.0%}|{check["seed"]}|{fmt(check["score"])}|{fmt(paired["gain_points"])}|[{fmt(ci[0])}, {fmt(ci[1])}]|{fmt(max(check["fixed_controls"]))}|'+('是' if check['pass'] else '否')+'|')
 if confirmation_seen:
  outcomes+=['','置信区间按图像聚类bootstrap计算。两seed的点估计有提升，但区间均跨零，未满足预登记的下界>0要求；这表示尚无充分证据证明稳定改善，不能解释为已证明完全无效。50%保留率独立确认集的未压缩teacher宏平均为55.220；该集合与上面的筛选集不同，不能将两个集合的绝对分数变化当成训练退化。独立确认逐题原始结果见 `artifacts/compression_lab/expanded/expanded_spatial_cnn_r0.5_seed20261105/confirmation.epoch8.jsonl` 及seed20261106同名文件；含gold、teacher和student。','']
 doc=['# 冻结 UND 的新压缩生成器：实验协议与真实进度','',f'更新 UTC：{summary["updated_utc"]}。已完成 **{completed}/{len(records)}** 个训练配置（102个初筛/追加/位置消融，4个扩大训练），基础设施失败 {failed} 个。完整训练已登记 {summary["planned_full_runs"]} 个。**未完成的结果不填写，也不把启动作业视为验证成功。**','',
 '入口：设计见 [COMPRESSION_GENERATOR_DESIGNS.md](COMPRESSION_GENERATOR_DESIGNS.md)；代码 `vqa/compression_generators.py`；训练 `scripts/train_compression_lab.py`；统一命令 `bash scripts/run_compression_lab.sh status`。','',
 *outcomes, '## 实验协议','',
 '官方 MonkeyOCRv2-B-Und 全主干冻结（视觉塔、merger、Qwen3-1.7B 均冻结），只训练 P 生成器。插入点为 merger 后，C=P@X。三种保留率10%/25%/50%，即10×/4×/2× token 压缩；不是整机速度提升倍数。不同方案和预算独立训练。', '',
 '训练：AdamW，LR=1e-4，weight_decay=0，batch=4，3epochs，10% warmup + cosine 至0.1×LR，global grad clipping=1；记录裁剪前范数和实际更新。目标为温度1的全词表 KL；重叠支持方案加0.01×相对列负载方差，不使用 entropy loss。teacher greedy答案作为共同前缀，不以 gold 作为训练标签。', '',
 '数据来源追溯：DocVQA/InfoVQA使用lmms-lab原训练源，ChartQA使用ahmed-masry原训练源，DT-VQA使用shuozhang2原训练源，DeepForm/KLC/WTQ使用echo840/Monkey_Data内DUE问答；处理后的图像在data/full/images。没有额外使用MonkeyDocV2的纯OCR解析目标训练这些生成器。', '', '数据均来自既有七任务：DocVQA / InfoVQA / ChartQA / DT-VQA / WTQ / DeepForm / KLC。初筛没有OCRBench训练集，不把它虚构成第八个训练任务。新 validation / confirmation 从训练源选取，排除旧20k训练集与旧调参/确认/正式测试图像；筛选不依赖teacher是否正确。训练筛选样本取旧训练图，因此不能称其未经训练见过；新生成器从头训练。', '',
 '|阶段|图像|问答|用途|','|---|---:|---:|---|']
 for name in ['train','validation','confirmation','expanded_train']:
  v=protocol.get('cohorts',{}).get(name,{});doc.append(f'|{name}|{v.get("images","—")}|{v.get("questions","—")}|'+{'train':'24×3候选训练','validation':'初筛和模型选择','confirmation':'选定方案独立双seed确认（实际结果见前表）','expanded_train':'通过初筛才扩大训练'}[name]+'|')
 doc+=['','初筛门槛：最后第3epoch验证KL下降≥5%、问答宏平均相对初始提高≥2分，且胜过同预算mean与中心selection两项固定基线。28图的小验证集仅能筛选，不能给显著性结论。通过后在672图上多轮双seed训练，固定方案后才使用56图confirmation；图像聚类bootstrap下界需>0，且训练/验证KL有改善，才允许原20k数据全量训练。正式测试不得用于选择。','',
 '独立 greedy生成上限64，teacher/student一致。每次问答重建prompt，不传跨题KV缓存。native task macro均分；DeepForm/KLC这里是字段exact，**不是官方document-level F1**。答案、gold、teacher、student、KL及P统计写入逐题CSV。','',
 '## 参数追加实验（自适应探索，不是独立确认）','', '首批8组10%保留率实验验证KL几乎不变，梯度有限且未被裁剪。随后登记18组追加：6种结构×3预算，LR从1e-4提高至1e-3，epochs从3增至12，其余数据/结构/loss/门槛不变。原72组继续完成；追加结果单独标识，不把增加训练量伪装成新增架构。登记文件 `artifacts/compression_lab/followup_preregistration.json`。','', '新增位置消融：四种结构×三预算，merger前插入，12epochs/LR1e-3。只看原始H与问题输入embedding，2048维X输入槽置零。实际decoder输入=merger(P@H)，与post-merger的P@merger(H)不同；单独比较pre-merger固定基线，不能把位置变化误报为训练收益。登记 `pre_preregistration.json`。','', '扩大量实验固定协议：每预算选完所有候选后只保留一个家族，在672图/1801QA上从头初始化，两个seed=20261105/20261106，4-NPU DDP、global batch32、LR3e-4、8epochs。10%warmup+cosine、clip1不变。LR是小样本追加1e-3的0.3倍、更新步数约456（pilot168）；不沿用上一轮对完整数据LR缩小十多倍的策略。尾batch循环补齐，实际补齐数见complete.json。第8epoch固定为终点，不根据最后测试挑轮次。两seed须训练audit/validation KL各下降5%、分数提高2且胜同预算基线；之后才启用56图独立confirmation，配对图像聚类CI下界>0且优于固定基线方可进入原20k训练。', '', '仅确认通过后才启动原20k训练：fresh初始化、预先固定seed20261105、4epochs、global batch32、LR3e-4、相同scheduler/clip。最终只评第4epoch；全量23565QA、cap256报告teacher/initial/trained/mean/selection，正式测试成绩不回传给模型选择。未通过确认则不启动大规模训练。', '', '强基线追加（门槛收紧，非放宽）：50%预算的merger前pilot虽然胜同位置基线，绝对分数低于merger后mean。因此从扩大验证到完整训练的准入，同时要求胜过pre/post两个位置的mean/selection四项固定控制。初筛原门槛和已冻结家族不改写；登记strong_control_addendum.json；该门槛收紧发生在首次使用独立confirmation之前，后续实际确认结果见前表。', '', '## 固定基线','', '|保留率|均值|中心选择|','|---|---:|---:|']
 for ratio in [.1,.25,.5]:
  vals=[load(BASE/'baselines'/f'{kind}_r{ratio:g}'/'validation.fixed.summary.json').get('student_macro') for kind in ['mean','selection']]
  doc.append(f'|{ratio:.0%}|{fmt(vals[0])}|{fmt(vals[1])}|')
 doc+=['','merger前固定基线见 `artifacts/compression_lab/baselines_pre/`；每行门槛列使用与其插入位置匹配的固定基线。','', '## 逐方案初筛结果','', '分数与增益单位均为百分点。KL是生成teacher共同前缀上的分布差异；初筛通过≠独立确认通过≠全量成功。','', '|方案 / 阶段|保留率|状态|初始得分|最后得分|变化|验证KL初始→最后|KL降幅%|同预算强基线|门槛|','|---|---:|---|---:|---:|---:|---|---:|---:|---|']
 for r in records:
  doc.append(f'|{r["kind"]} / {r["phase"]}|{r["ratio"]:.0%}|{r["status"]}|{fmt(r["initial_score"])}|{fmt(r["final_score"])}|{fmt(r["gain"])}|{fmt(r["initial_kl"])} → {fmt(r["final_kl"])}|{fmt(r["kl_drop_percent"])}|{fmt(r["fixed_best"])}|'+('通过初筛' if r['screen_gate'] else '未通过/未完成')+'|')
 smoke=load(BASE/'smoke/complete.json')
 doc+=['','## 后续阶段状态', '', *[f'- 保留率{k}: `{v.get("stage","waiting")}`；选择的pilot `{v.get("selected_pilot",{}).get("name","无/尚未选择")}`。' for k,v in load(BASE/'stages.json').get('ratios',{}).items()], '', '阶段控制入口：`.venv/bin/python -m scripts.advance_compression_lab --submit`。使用文件锁和已登记状态避免重复推进；未通过门槛不会启动全量训练。', '', '## 梯度与冻结审计','',f'真实 UND decoder 梯度 smoke 完成：{smoke.get("schemes",0)}/24。明细 `artifacts/compression_lab/smoke/results.json`。检查生成器非零KL梯度、参数真实更新、P非负/行和1、主干可训练参数为0且所有主干grad=None。', '',
 f'身份矩阵与问答独立性审计：`artifacts/compression_lab/infrastructure/audit.json`；真实7任务，P=I的KL为{load(BASE/"infrastructure/audit.json").get("identity_max_abs_KL","尚未完成")}，作答与teacher逐项一致；穿插其他问题后返回原题，生成token与teacher缓存一致。', '', '每个run保留 `initial.pt`、`epoch*.pt`、`latest.pt`、代码快照与SHA、数据SHA、`training.jsonl`、`validation.*.jsonl`、训练audit loss。未完成epoch从已提交checkpoint继续，重复结果不计两次。基础设施失败修复前不能写为负面科学结果。','',
 '## 产物与命令','', '```bash','bash scripts/run_compression_lab.sh prepare','bash scripts/run_compression_lab.sh smoke','bash scripts/run_compression_lab.sh cache','bash scripts/run_compression_lab.sh baseline','bash scripts/run_compression_lab.sh screen','bash scripts/run_compression_lab.sh status','bash scripts/run_compression_lab.sh report','```','',
 '- `artifacts/compression_lab/protocol.json`：预登记门槛、划分与参数。','- `artifacts/compression_lab/screen_summary.csv`：含实际梯度/裁剪/更新汇总。','- `artifacts/compression_lab/answers.csv`：筛选验证集逐题teacher/student/gold/loss；独立confirmation结果另存上述confirmation.*.jsonl，不混入本CSV。','- `artifacts/compression_lab/literature/retrieval.json`：本轮HTTP原文检索记录。','',
 '训练/验证逐epoch曲线：`artifacts/compression_lab/plots/loss_*_keep*.pdf`；问答与固定基线曲线：`score_*_keep*.pdf`。运行 `.venv/bin/python -m scripts.plot_compression_lab` 更新。', '', '[初筛10% loss](../artifacts/compression_lab/plots/loss_initial_screen_keep0.1.pdf) · [初筛25% loss](../artifacts/compression_lab/plots/loss_initial_screen_keep0.25.pdf) · [初筛50% loss](../artifacts/compression_lab/plots/loss_initial_screen_keep0.5.pdf) · [追加25% loss](../artifacts/compression_lab/plots/loss_higher_update_budget_keep0.25.pdf) · [追加25%问答](../artifacts/compression_lab/plots/score_higher_update_budget_keep0.25.pdf)', '', '截至本报告，仅表格中真实完成的数据构成实验结果；尚未确认高压缩比优于基线，不声称达到论文成绩。']
 doc_tmp=ROOT/f'docs/COMPRESSION_GENERATOR_EXPERIMENTS.{__import__("os").getpid()}.tmp';doc_tmp.write_text('\n'.join(doc)+'\n');doc_tmp.replace(ROOT/'docs/COMPRESSION_GENERATOR_EXPERIMENTS.md')
 print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
if __name__=='__main__':main()
