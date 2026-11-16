"""Generate second-round tables exclusively from committed experiment artifacts."""
import csv,json,datetime,os
from pathlib import Path
from collections import Counter
from scripts.prepare_compression_lab_v2 import ROOT,BASE
from scripts.train_compression_lab_v2 import read
CONTROL_CACHE={}
def refresh(path):
 # The shared FS can cache negative lookups on CPU/NPU nodes independently.
 # Readdir the parent and directory before checking newly committed result files.
 try:list(path.parent.iterdir());list(path.iterdir())
 except FileNotFoundError:pass
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def fmt(v):return '—' if v is None else f'{v:.3f}'
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def controls(ratio,cohort='validation'):
 key=(ratio,cohort)
 if key in CONTROL_CACHE:return CONTROL_CACHE[key]
 folder=BASE/'baselines' if cohort=='validation' else BASE/'confirmation_baselines'
 result=[load(folder/f'{position}_{kind}_r{ratio:g}'/f'{cohort}.fixed.summary.json').get('student_macro')
 for position in ['pre-merger','post-merger'] for kind in ['mean','selection','cell_mean','cell_selection']]
 CONTROL_CACHE[key]=result;return result
def export_answers():
 """Include only actual generation, with explicit run/stage/cohort; no seed averaging."""
 fields=['run','kind','insertion','ratio','id','document_id','task','cohort','stage','question','answers',
 'teacher_prediction','student_prediction','teacher_score','student_score','kl','tokens','compressed_tokens',
 'entropy','row_max_mean','distinct_row_argmax','hit_token_limit','teacher_hit_token_limit']
 configs=[load(ROOT/p) for name in ['screen_configs.json','expanded_configs.json','full_configs.json'] for p in load(BASE/name)]
 answer_tmp=BASE/f'answers.{os.getpid()}.tmp'
 with answer_tmp.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for cfg in configs:
   out=ROOT/cfg['output']
   for pattern in ['validation.*.jsonl','confirmation.*.jsonl']:
    for path in sorted(out.glob(pattern)):
     for row in read(path):
      if row.get('student_score') is None:continue
      writer.writerow({'run':cfg['name'],'kind':cfg['kind'],'insertion':cfg['insertion'],'ratio':cfg['ratio'],**row,'answers':json.dumps(row['answers'],ensure_ascii=False)})
 answer_tmp.replace(BASE/'answers.csv')
def gate(cfg):
 out=ROOT/cfg['output'];a=load(out/'validation.initial.summary.json');b=load(out/f'validation.epoch{cfg["epochs"]}.summary.json')
 ta=load(out/'train_audit.initial.summary.json');tb=load(out/f'train_audit.epoch{cfg["epochs"]}.summary.json');cc=controls(cfg['ratio']);reasons=[]
 if not all([a,b,ta,tb]) or None in cc or 'student_macro' not in b:return {'pass':False,'reasons':['incomplete']}
 if b['kl']>a['kl']*.95:reasons.append('validation KL decrease <5%')
 if tb['kl']>ta['kl']*.95:reasons.append('train audit KL decrease <5%')
 if b['student_macro']<a['student_macro']+2:reasons.append('score gain <2points')
 if b['student_macro']<=max(cc):reasons.append('not above all eight fixed controls')
 return {'pass':not reasons,'reasons':reasons,'initial':a,'final':b,'fixed_controls':cc}
def main():
 CONTROL_CACHE.clear()
 refresh(BASE)
 for folder in ['runs','expanded','full','confirmation_baselines']:refresh(BASE/folder)
 configs=[load(ROOT/p) for listname in ['screen_configs.json','expanded_configs.json','full_configs.json'] for p in load(BASE/listname)]
 stages=load(BASE/'stages.json').get('ratios',{})
 from scripts.compression_lab_v2_jobs import live_jobs
 queued=live_jobs()
 records=[]
 for cfg in configs:
  out=ROOT/cfg['output'];refresh(out);a=load(out/'validation.initial.summary.json');b=load(out/f'validation.epoch{cfg["epochs"]}.summary.json');training=read(out/'training.jsonl');gg=gate(cfg)
  status='完成' if (out/'complete.json').exists() else '基础设施失败' if (out/'failure.txt').exists() else '运行中' if a or training else '未开始'
  if cfg['phase']=='v2_full_20k' and status!='完成':
   job=stages.get(f'{cfg["ratio"]:g}',{}).get('full_job')
   status='训练中/待调度' if str(job) in queued else '已中断/待恢复'
  cc=controls(cfg['ratio']);best=max(cc) if None not in cc else None
  records.append({'name':cfg['name'],'kind':cfg['kind'],'insertion':cfg['insertion'],'ratio':cfg['ratio'],'phase':cfg['phase'],'seed':cfg['seed'],
   'status':status,'initial_score':a.get('student_macro'),'final_score':b.get('student_macro'),'teacher_score':a.get('teacher_macro'),
   'gain':b['student_macro']-a['student_macro'] if 'student_macro' in a and 'student_macro' in b else None,
   'initial_kl':a.get('kl'),'final_kl':b.get('kl'),'kl_drop_percent':100*(1-b['kl']/a['kl']) if a and b else None,
   'fixed_best':best,'gate':gg['pass'],'gate_reasons':'; '.join(gg['reasons']),'steps':len(training),
   'max_gradient':max((r['grad_norm'] for r in training),default=None),
   'median_clip':sorted(r['clip_multiplier'] for r in training)[len(training)//2] if training else None,
   'total_update_l2':sum(r['update_l2'] for r in training)})
 if records:
  csv_tmp=BASE/f'summary.{os.getpid()}.csv.tmp'
  with csv_tmp.open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
  csv_tmp.replace(BASE/'summary.csv')
 summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned':len(records),'status':dict(Counter(r['status'] for r in records)),
 'pilot_passed':[r['name'] for r in records if r['gate'] and r['phase']=='v2_screen'],'records':records}
 save(BASE/'summary.json',summary)
 stages=load(BASE/'stages.json').get('ratios',{});protocol=load(BASE/'protocol.json')
 doc=['# 第二轮24种新压缩生成器：协议、过程与实际结果','',f'更新UTC：{summary["updated_utc"]}；训练配置状态：`{json.dumps(summary["status"],ensure_ascii=False)}`。登记{len(records)}组，未运行不填写结果。',
 '', '**入口：`bash scripts/run_compression_lab_v2.sh status`**。结构见 [COMPRESSION_GENERATOR_V2_DESIGNS.md](COMPRESSION_GENERATOR_V2_DESIGNS.md)。生成器代码 `vqa/compression_generators_v2.py`及其数值修复模块；实际训练入口`scripts/train_compression_lab_v2_repaired.py`，公共训练循环位于`train_compression_lab_v2.py`，不要绕过修复入口直接训练sparse_relu。所有实验使用venv；NPU用`.venv-und`，CPU报告用`.venv`。旧实验不覆盖。','',
 '## 当前阶段','', '|保留率 / token压缩倍数|阶段|选定家族 / 位置|','|---|---|---|']
 for ratio in [.1,.25,.5]:
  state=stages.get(f'{ratio:g}',{});chosen=state.get('selected_pilot',{})
  doc.append(f'|{ratio:.0%} / {1/ratio:g}×|{state.get("stage","初筛中/尚未选择")}|{chosen.get("kind","—")} / {chosen.get("insertion","—")}|')
 doc+=['','完整初筛完成数：'+str(sum(r['status']=='完成' and r['phase']=='v2_screen' for r in records))+'/144；通过配置数：'+str(len(summary['pilot_passed']))+'/144。各预算通过数：'+', '.join(f'{ratio:.0%}: '+str(sum(r['gate'] and r['phase']=='v2_screen' and r['ratio']==ratio for r in records))+'/48' for ratio in [.1,.25,.5])+'。通过表示满足本轮门槛，不表示已接近teacher。']
 doc+=['','|完整训练保留率|已记录更新 / 计划|最新完整epoch|该epoch筛选集生成分|','|---|---:|---:|---:|']
 for ratio,state in stages.items():
  if not state.get('full_config'):continue
  cfg=load(ROOT/state['full_config']);directory=ROOT/cfg['output'];logs=read(directory/'training.jsonl')
  epoch=max((e for e in range(cfg['epochs']+1) if load(directory/f'validation.{"initial" if e==0 else "epoch"+str(e)}.summary.json').get('student_macro') is not None),default=0)
  result=load(directory/f'validation.{"initial" if epoch==0 else "epoch"+str(epoch)}.summary.json')
  doc.append(f'|{float(ratio):.0%}|{max((r["step"] for r in logs),default=0)} / 2500|{epoch} / {cfg["epochs"]}|'+fmt(result.get('student_macro'))+'|')
 doc+=['','epoch0是初始化；日志步数不等于checkpoint已提交步数，中断时按既定checkpoint恢复。此处是训练进度，最终结果见正式评测。']
 doc+=['','## 固定初筛候选（模型选择分数）','', '本表只按末轮分数展示已完成候选；全部同预算候选完成前不冻结模型，也不把最高分自动视为通过。跨预算均使用同一新筛选集，不能与第一轮不同集合的绝对分数直接比较。','',
 '|保留率|候选 / 位置|初始化→末轮|增益|未压缩teacher|最强固定控制|完整初筛门槛|','|---|---|---|---:|---:|---:|---|']
 for ratio in [.1,.25,.5]:
  done=[r for r in records if r['phase']=='v2_screen' and r['ratio']==ratio and r['status']=='完成']
  if not done:continue
  best=max(done,key=lambda r:r['final_score'])
  doc.append(f'|{ratio:.0%}|{best["kind"]} / {best["insertion"]}|{fmt(best["initial_score"])} → {fmt(best["final_score"])}|{fmt(best["gain"])}|{fmt(best["teacher_score"])}|{fmt(best["fixed_best"])}|'+('通过' if best['gate'] else best['gate_reasons'])+'|')
 doc+=['','## 24种方案的固定末轮横向比较','',
 '同一新筛选集、固定epoch6的实际生成分数；没有跨seed平均。✓表示同时满足train/validation KL与作答门槛。最高分仅用于登记的初筛选择，不能视为独立测试成绩。','',
 '|方案|10% pre|10% post|25% pre|25% post|50% pre|50% post|','|---|---:|---:|---:|---:|---:|---:|']
 names=list(dict.fromkeys(r['kind'] for r in records if r['phase']=='v2_screen'))
 lookup={(r['kind'],r['ratio'],r['insertion']):r for r in records if r['phase']=='v2_screen'}
 for name in names:
  cells=[]
  for ratio in [.1,.25,.5]:
   for position in ['pre-merger','post-merger']:
    r=lookup.get((name,ratio,position),{});cells.append(fmt(r.get('final_score'))+(' ✓' if r.get('gate') else ''))
  doc.append('|'+name+'|'+'|'.join(cells)+'|')
 doc+=['','[24×6实际得分热图 PDF](../artifacts/compression_lab_v2/plots/screen_score_matrix.pdf) · [PNG](../artifacts/compression_lab_v2/plots/screen_score_matrix.png)。']
 doc+=['','## 数据与冻结设置','', '官方MonkeyOCRv2-B-Und的encoder、merger、Qwen3-1.7B全冻结，只训练P生成器。pre输出merger(PH)，post输出PX；pre不读取X。teacher未压缩，teacher和student每题独立greedy生成，上限64。各任务native macro百分数；DeepForm/KLC为字段exact，**不是官方文档级F1**。不以模型生成前缀KL代替独立作答分数。', '',
 '计分：DocVQA/InfoVQA为原始字符ANLS；ChartQA为5%数值容差relaxed accuracy；DT-VQA为max(答案包含accuracy, ANLS)；WTQ为DUE denotation accuracy；DeepForm/KLC为字段exact。初筛/确认的宏平均为七任务等权。正式评测另外含OCRBench（原生答案包含规则），为八任务等权宏平均；不能将两个集合/任务范围的绝对均分直接解释为训练增减。','',
 '|集合|图像|问答|用途|','|---|---:|---:|---|']
 for name,meta in protocol.get('cohorts',{}).items():doc.append(f'|{name}|{meta["images"]}|{meta["questions"]}|'+{'train':'144组初筛训练','validation':'全新模型选择集','confirmation':'全新独立确认，仅冻结候选后使用','expanded_train':'双seed扩大训练','train_audit':'固定训练loss审计'}[name]+'|')
 doc+=['','数据来自DocVQA/InfoVQA/ChartQA/DT-VQA/WTQ/DeepForm/KLC既有训练源，不使用MonkeyDocV2纯OCR目标。新heldout按图像排除第一轮全部heldout、旧调参/确认集、旧20k训练和正式val/test；实际划分SHA和图像存在/互斥检查见 `protocol.json`、`data_integrity.json`。这不保证官方模型预训练时从未见过来源数据。', '',
 '## 训练与选择协议','', '初筛24结构×2插入位置×3预算=144配置，全部fresh init。AdamW LR1e-3/wd0/batch8/6epochs，10%warmup+cosine至0.1LR，global clip1。目标仅全词表KL(teacher||student)，T1，teacher自身greedy响应为共同前缀；不加entropy/列塌缩正则。每epoch记录validation和train-audit KL；epoch0/3/6独立生成，固定epoch6选择，不能追选中间最好epoch。', '',
 '初筛门槛：两种KL各下降≥5%，生成宏平均比初始提高≥2分，且胜过pre/post两位置的一维mean/selection与二维cell_mean/cell_selection共8项固定控制。每预算只选一个架构/位置，在672图双seed（20261105/20261106）fresh训练8epochs，LR3e-4、batch32、4NPU DDP，其余scheduler/clip不变。两seed均通过后，才使用新112图confirmation；相对各自初始分数的配对图像聚类95%CI下界>0且优于固定控制，才进入原20k训练。未通过则不自动进入全量训练；用户明确授权的后续扩展另行登记，保留原失败结论。','',
 '完整阶段在首次使用新confirmation之前另登记full_training_addendum.json：确认通过的每预算家族从头初始化，在原20k数据上固定seed20261105、4epochs、LR3e-4、global batch32、clip1、同warmup/cosine。原20k指20,000QA/16,731图像：DT-VQA9883、DocVQA3204、ChartQA2337、InfoVQA2021、WTQ1153、KLC1109、DeepForm293。沿用原始shuffle，未做任务平衡，训练混合比例与任务宏平均权重不同。固定epoch4后才做23565QA正式评测、cap256，teacher/initial/trained以及所选位置mean/selection/cell_mean/cell_selection四项控制，正式分数不用于选模型。正式控制是同位置四项；准入门槛始终比较跨位置八项。', '',
 '离散分区/排序detach，硬选择采用有偏ST；详细边界见设计表。梯度、clip倍率、实际更新写入逐step日志。全部候选和位置完成真实UND梯度检查后才启动初筛，不把基础设施错误当负结果。初筛曾由8个独立NPU进程重分配到14个，各run仍为单NPU，沿已提交epoch checkpoint恢复，不改优化器/训练数据/顺序；未提交epoch从checkpoint重放。作业沿革见initial_submissions.json。','',
 '完整阶段根据资源运行：10%预算8 NPU、25%预算4 NPU，正式评测4 NPU；global batch32、数据顺序、目标、LR、seed和固定epoch均不变。曾申请16 NPU但无整节点资源，均在未启动时撤回；25%曾运行8 NPU但在首个epoch前因IO中断，恢复时不足8卡，退回原定4卡，从同seed重放。所有未提交日志、initial和源码清单均保留在`infrastructure/`，不是根据得分重启。2500个global batch在4/8/16 rank下覆盖相同样本，审计见`infrastructure/full_compute_repartition.json`；10%预算4/8卡初始化逐tensor位级相同，见`full_initialization_identity.json`。浮点梯度累积顺序不同，不声称训练位级复现；扩大训练仍是4 NPU。','',
 '## 数值修复与原始失败保留','', 'sparse_relu原实现对极小正权重行的分母加1e-12，导致行和不再为1，已被评测断言拦下。修复模块`vqa/compression_generators_v2_repair.py`对该方案的非空P行精确再归一化，空行仍回退二维均值。其6个配置全部从头重跑，不混用原checkpoint；两项原失败日志/checkpoint保留于`infrastructure/failed_sparse_original/`，详情`sparse_repair.json`。这属于实现数值错误，不记作架构负结果。其他23项保留原代码SHA及checkpoint；训练入口以`train_compression_lab_v2_repaired.py`显式分派，修复源码加入该类run的manifest。', '',
 '完整阶段新遇到并发缓存写入EEXIST：两个作业同时计算同一未缓存teacher条目，目标已由另一进程提交时，共享FS的rename报错。新增纯IO入口`scripts/compression_lab_v2_io.py`（正式评测为`benchmark_compression_lab_v2_io.py`）：仅对不可变features/teachers/logits缓存的该异常，要求目标与待写对象所有tensor和metadata逐项、位级相同，才接受已提交文件；不同内容或checkpoint写入错误一律不吞掉。独立边界检查见`infrastructure/cache_eexist_guard_test.json`，故障/重放日志见`cache_eexist_repair.json`与原Slurm日志。实际wrapper SHA另写各run的`io_runtime.rank*.json`，科学计算源码SHA不变；未提交更新从既定checkpoint重放，不按分数重启。','',
 '另一共享FS故障为缓存存在性检查后紧接着load仍报ENOENT；版本化入口`compression_lab_v2_io_retry.py`仅对此类不可变缓存执行有限次刷新目录/重试，真实缺失最终仍报错，检查点不走重试。记录见`cache_read_visibility_repair.json`，wrapper SHA另存`io_retry_runtime.rank*.json`，不修改已签名科学源码。','',
 '曾用空闲NPU预计算原20k的冻结teacher缓存，从训练首轮的反向顺序提前读取，随后因资源争用停止并保留所有已提交缓存；入口`scripts/slurm_compression_lab_v2_prefetch.sh`，记录在`infrastructure/full_cache_prefetch/`及`full_cache_prefetch_submission.json`。该过程模型更新为0、不计算候选分数，不改变训练样本、顺序或KL定义。','',
 '## 固定控制（新筛选集）','', '|保留率|pre mean|pre selection|pre cell mean|pre cell selection|post mean|post selection|post cell mean|post cell selection|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for ratio in [.1,.25,.5]:doc.append(f'|{ratio:.0%}|'+ '|'.join(fmt(v) for v in controls(ratio))+'|')
 doc+=['','## 前轮已训练模型在同一新筛选集上的补充对照','', '仅评前轮4个已完成扩大训练模型的既定epoch8，不追选旧checkpoint；作为同集合参照，不改变本轮预登记门槛。该对照不是在旧确认集上重复选模型。','',
 '|前轮模型|新筛选集宏平均|KL|','|---|---:|---:|']
 for path in sorted((BASE/'legacy_controls').glob('*/validation.epoch8.summary.json')):
  val=load(path);doc.append(f'|{path.parent.name}|{fmt(val["student_macro"])}|{fmt(val["kl"])}|')
 doc+=['','## 逐方案训练结果','', '分数/增益为百分点，KL降幅为百分数。下表通过仅表示该训练阶段的连续loss与生成门槛通过，独立确认以阶段表为准。','',
 '|方案 / 阶段 / seed|位置|保留率|状态|初始分|末轮分|增益|KL初始→末轮|KL降幅%|强固定控制|阶段门槛|','|---|---|---:|---|---:|---:|---:|---|---:|---:|---|']
 for r in records:doc.append(f'|{r["kind"]} / {r["phase"]} / {r["seed"]}|{r["insertion"]}|{r["ratio"]:.0%}|{r["status"]}|{fmt(r["initial_score"])}|{fmt(r["final_score"])}|{fmt(r["gain"])}|{fmt(r["initial_kl"])} → {fmt(r["final_kl"])}|{fmt(r["kl_drop_percent"])}|{fmt(r["fixed_best"])}|'+('通过' if r['gate'] else r['gate_reasons'])+'|')
 doc+=['','## 独立确认','', 'CI检验的是相对同seed初始化的改善；“超过固定控制”是分数点估计条件，不等同于已检验两者差异显著。两seed分别报告，不能平均后掩盖一个seed失败。', '',
 '|保留率|seed|初始化|训练后|未压缩teacher|相对初始化增益|图像聚类95%CI|最强固定控制|通过|','|---|---:|---:|---:|---:|---:|---|---:|---|']
 anyconfirmation=False
 for ratio,state in stages.items():
  for check in state.get('confirmation_checks',[]):
   anyconfirmation=True;paired=check['paired_vs_initial'];ci=paired['cluster_bootstrap_95ci_points']
   doc.append(f'|{float(ratio):.0%}|{check["seed"]}|{fmt(check.get("initial_score"))}|{fmt(check["score"])}|{fmt(check.get("teacher_score"))}|{fmt(paired["gain_points"])}|[{fmt(ci[0])}, {fmt(ci[1])}]|{fmt(max(check["fixed_controls"]))}|'+('是' if check['pass'] else '否')+'|')
 if not anyconfirmation:doc+=['','尚无已完成的独立确认结果；不能宣称验证成功。']
 if anyconfirmation:
  doc+=['','### 当前证据支持什么','',
   '初筛较好的主要是自适应行/列/矩形覆盖和全局选择；更复杂的原型、Sinkhorn、可变形路由没有自动带来更高分。这里是共同训练设置下的比较，不能解释为这些家族在充分调参后仍必然劣势。','']
  for ratio,state in stages.items():
   checks=state.get('confirmation_checks',[])
   if not checks:continue
   for c in checks:
    teacher=c.get('teacher_score');retention=100*c['score']/teacher if teacher else None
    doc.append(f'- 保留{float(ratio):.0%} / seed{c["seed"]}：确认分数{fmt(c["score"])}，相对teacher下降{fmt(teacher-c["score"])}分，分数保留率{fmt(retention)}%；相对初始化增益{fmt(c["paired_vs_initial"]["gain_points"])}分。')
   if not all(c['pass'] for c in checks):
    doc+=['',f'保留{float(ratio):.0%}未满足“两seed都通过”的预登记要求，原自动流程因此停止该预算的大规模训练。置信区间跨零表示本轮证据未通过门槛，不等于证明该生成器完全无效；不能丢掉失败seed，只保留另一个。']
  doc+=['','10%/25%的显著学习收益仍不等于达成“高压缩下接近teacher”：必须同时看绝对差距和完整基准。所有确认结果来自112图/209QA、七任务宏平均；样本较少，正式评测仍应单独报告。']
 if anyconfirmation:
  doc+=['','### 独立确认逐任务结果','',
   '各seed保留独立结果。这里的逐任务分数用于解释已冻结模型的表现，不据此重选架构、epoch或seed。','',
   '|保留率|seed|任务|题数|teacher|初始|训练后|得分上升 / 持平 / 下降题数|','|---|---:|---|---:|---:|---:|---:|---|']
  for ratio,state in stages.items():
   if not state.get('confirmation_checks'):continue
   for cfgpath in state['expanded_configs']:
    cfg=load(ROOT/cfgpath);out=ROOT/cfg['output'];before={r['id']:r for r in read(out/'confirmation.initial.jsonl')};after=read(out/'confirmation.epoch8.jsonl')
    for task in sorted({r['task'] for r in after}):
     rr=[r for r in after if r['task']==task];n=len(rr)
     t=100*sum(r['teacher_score'] for r in rr)/n;a=100*sum(before[r['id']]['student_score'] for r in rr)/n;b=100*sum(r['student_score'] for r in rr)/n
     differences=[r['student_score']-before[r['id']]['student_score'] for r in rr]
     counts=[sum(d>1e-12 for d in differences),sum(abs(d)<=1e-12 for d in differences),sum(d< -1e-12 for d in differences)]
     doc.append(f'|{float(ratio):.0%}|{cfg["seed"]}|{task}|{n}|{fmt(t)}|{fmt(a)}|{fmt(b)}|'+ ' / '.join(map(str,counts))+'|')
 doc+=['','## 选定生成器的实际矩阵统计','',
  '以下在同一56题筛选集统计实际前向P；仅显示initial和既定末轮均完整的run。entropy是平均逐行熵，最大权重是平均逐行最大值；它们是训练后的描述，不用于重选模型。没有对两个seed平均。','',
  '|阶段 / 保留率 / seed|平均行熵 初始→末轮|平均行最大权重 初始→末轮|末轮最小不同argmax行占比|末轮最大行和误差|','|---|---:|---:|---:|---:|']
 for state in stages.values():
  paths=list(state.get('expanded_configs',[]))
  if state.get('full_config'):paths.append(state['full_config'])
  for path in paths:
   cfg=load(ROOT/path);out=ROOT/cfg['output']
   if not (out/'complete.json').exists():continue
   aa=read(out/'validation.initial.jsonl');bb=read(out/f'validation.epoch{cfg["epochs"]}.jsonl')
   if len(aa)!=56 or len(bb)!=56:continue
   def avg(rr,key):return sum(r[key] for r in rr)/len(rr)
   unique=min(r['distinct_row_argmax']/r['compressed_tokens'] for r in bb)
   error=max(r['row_sum_error'] for r in bb)
   doc.append(f'|{cfg["phase"]} / {cfg["ratio"]:.0%} / {cfg["seed"]}|{avg(aa,"entropy"):.4f} → {avg(bb,"entropy"):.4f}|{avg(aa,"row_max_mean"):.4f} → {avg(bb,"row_max_mean"):.4f}|{unique:.4f}|{error:.2e}|')
 doc+=['','global_topk的真实前向和评测均为one-hot，因此行熵始终为0；其训练收益来自选中了哪些原token，不是熵下降。自适应矩形方案的行最大值增大意味着区域内权重更集中，但本轮没有加入entropy正则。P统计不能独立证明VQA质量。']
 doc+=['','## 完整训练与正式评测','', '原流程只在独立确认通过后登记20k训练。50%由用户后续明确授权扩展，原失败确认不改写；参见[50%扩展协议](COMPRESSION_GENERATOR_V2_KEEP50_FULL.md)。正式评测固定23565QA、cap256，不参与模型选择。下面留空表示尚无完整正式结果，不能用初筛或672图分数冒充。', '',
 '|保留率|teacher|initial|trained|mean|selection|cell_mean|cell_selection|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for ratio,state in stages.items():
  full=state.get('benchmark_summary',{})
  if full:
   doc.append(f'|{float(ratio):.0%}|'+ '|'.join(fmt(full['macro'].get(v)) for v in ['teacher','initial','trained','mean','selection','cell_mean','cell_selection'])+'|')
 for ratio,state in stages.items():
  full=state.get('benchmark_summary',{})
  if full:
   mm=full['macro'];retention=100*mm['trained']/mm['teacher'] if mm['teacher'] else None
   doc+=['',f'保留率{float(ratio):.0%}：trained相对teacher下降{fmt(mm["teacher"]-mm["trained"])}分，分数保留率{fmt(retention)}%；相对本位置最强固定控制提升{fmt(mm["trained"]-max(mm[k] for k in ["mean","selection","cell_mean","cell_selection"]))}分。分数保留率不代表运行速度。']
 for ratio,state in stages.items():
  full=state.get('benchmark_summary',{})
  if full:
   doc+=['',f'保留率{float(ratio):.0%}逐任务分数：','', '|任务|teacher|initial|trained|mean|selection|cell_mean|cell_selection|','|---|---:|---:|---:|---:|---:|---:|---:|']
   for task,values in sorted(full['tasks'].items()):doc.append(f'|{task}|'+ '|'.join(fmt(values.get(v)) for v in ['teacher','initial','trained','mean','selection','cell_mean','cell_selection'])+'|')
   cfg=load(ROOT/state['full_config']);rel='../'+cfg['output']+'/benchmark'
   doc+=['',f'配对trained-vs-initial：`{json.dumps(full["paired_trained_vs_initial"],ensure_ascii=False)}`。正式逐题作答：[CSV]({rel}/answers.csv) · [图像/gold/teacher/initial/trained对照]({rel}/review.html)。','']
   if 'student_token_limit_counts' in full:
    doc+=['',f'达到256 token上限的题数：student各变体`{json.dumps(full["student_token_limit_counts"],ensure_ascii=False)}`，teacher {full["teacher_token_limit_count"]}题。这些题仍计入分数，没有删除或用短输出替代。','']
 recovery=load(BASE/'infrastructure/recovery_20261106/recovery.json')
 if recovery:
  doc+=['','## 2026-11-06 中断恢复','',
   '控制器改为查询当前用户队列，过期作业号视为已退出；调度器本身的错误仍上报。阶段转换先落盘，再执行缓存辅助作业清理，避免已提交评测因清理失败而丢失作业号。控制器对完整训练/评测最多自动恢复3次，不重新选择架构、seed或epoch。','',
   '不可变features/teachers/logits缓存的ENOENT使用最多12次目录刷新与有限退避，通过已经打开的文件句柄加载；真正缺失仍报错。并发EEXIST仍要求tensor与metadata逐项相同。checkpoint不伪装成缓存、不会吞掉读取或内容错误。','',
   '10%原训练仅记录463步，没有提交epoch checkpoint，因此按原seed和顺序从初始化重放；旧日志、初始权重、manifest和源代码均保留于infrastructure/recovery_20261106/。训练循环仅增加每25步提交latest.pt，固定epoch4权重、训练数据、KL、优化器和调度器均不变。25%已完成checkpoint不重训。','',
   'Ascend会在导入时替换torch.load，因此缓存补丁在导入后端后安装，并在真实NPU上确认初始化后仍生效。另已核对重放初始权重逐tensor相同、周期checkpoint包含optimizer状态，实际从step150恢复成功。重放的首个更新与原日志一致，随后NPU数值轨迹并非逐位相同，未声称完全确定性。','',
   '[恢复记录与源码hash](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/recovery.json) · [故障边界验证](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/checks.json) · [NPU后端安装顺序验证](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/npu_backend_order.json) · [checkpoint核对](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/checkpoint_audit.json) · [重放数值对照](../artifacts/compression_lab_v2/infrastructure/recovery_20261106/replay_trajectory_audit.json)。正式评测每预算23,565题、cap256，使用所选位置的四个固定控制，不将小筛选集成绩代替正式结果。']
 smoke=load(BASE/'smoke/complete.json')
 doc+=['','## 审计、逐题答案与曲线','',f'真实UND梯度检查：{smoke.get("cases",0)}/48（24结构×2位置），明细 `smoke/results.json`；主干可训练参数为0，主干参数grad均为空，P非负/行和1且生成器参数真实更新。', '',
 '每个run保留initial/epoch*/latest checkpoint、源代码快照与SHA、数据SHA、training.jsonl、逐epoch固定train-audit/validation KL。逐题作答在 `runs/<name>/validation.{initial,epoch3,epoch6}.jsonl`，扩大实验在 `expanded/<name>/`，独立确认在 `confirmation.*.jsonl`；均含question/gold/teacher/student/score/KL/P统计。', '',
 '汇总CSV：[summary.csv](../artifacts/compression_lab_v2/summary.csv)。逐题汇总：[answers.csv](../artifacts/compression_lab_v2/answers.csv)（运行report时更新，仅含真实生成的记录）。[浏览图像/gold/teacher/student](../artifacts/compression_lab_v2/review.html)可按预算、位置、方案、任务和集合筛选。训练/验证逐epoch曲线在 `artifacts/compression_lab_v2/plots/`。','',
 '|初筛保留率|pre训练/验证KL|post训练/验证KL|pre作答得分|post作答得分|','|---|---|---|---|---|',
 *[f'|{ratio:.0%}|'+ '|'.join(f'[{metric} {position}](../artifacts/compression_lab_v2/plots/{metric}_v2_screen_{position}_keep{ratio:g}.pdf)' for metric,position in [('loss','pre-merger'),('loss','post-merger'),('score','pre-merger'),('score','post-merger')])+'|' for ratio in [.1,.25,.5]], '',
 '```bash','bash scripts/run_compression_lab_v2.sh status','bash scripts/run_compression_lab_v2.sh report','bash scripts/run_compression_lab_v2.sh plot','bash scripts/run_compression_lab_v2.sh resume','```','',
 '与前轮相比，本轮同时改变了支持集、部分网络结构、head初始化幅度和初筛数据/更新量；因此跨轮提高不能单独归因于二维支持。补充旧模型同集合比较能减少评测集差异，但不能代替严格单因素消融。初筛最高分存在多候选选择偏差；确认区间按任务分层/图像聚类，未做跨三个预算的多重比较校正，正式测试不用于反复选择。', '',
 '报告只陈述已完成结果；高压缩比目标是否达到，必须看独立确认与最终基准，不能用方案数量或loss下降替代。']
 target=ROOT/'docs/COMPRESSION_GENERATOR_V2_EXPERIMENTS.md';tmp=target.with_name(target.name+f'.{os.getpid()}.tmp');tmp.write_text('\n'.join(doc)+'\n');tmp.replace(target)
 print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
if __name__=='__main__':
 main()
 if '--answers' in __import__('sys').argv:export_answers()
