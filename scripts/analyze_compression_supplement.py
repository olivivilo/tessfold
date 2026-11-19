"""Read-only T1--T6 exports, paired image bootstrap and seed aggregation.

No scientific result is published until complete-cohort score/identity audits pass.
"""
from __future__ import annotations
import collections,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scripts.supplement_common import ROOT,BASE,load,save,lines,sha,legacy,benchmark_rows,summary

TASKS=['ChartQA','DT-VQA','DeepForm','DocVQA','InfoVQA','KLC','OCRBench','WTQ']


def answers(path,variant='trained'):
    out={}
    for p in sorted(Path(path).glob('predictions.rank*.jsonl')):
        with p.open() as f:
            for ln in f:
                r=json.loads(ln)
                if r['variant']==variant:
                    assert r['id'] not in out;out[r['id']]=r
    return out


def paired_maps(aa,bb,repeat=10000):
    assert aa.keys()==bb.keys() and aa
    groups=collections.defaultdict(list);doc_task=collections.defaultdict(set)
    for k,a in aa.items():
        b=bb[k]
        assert a['task']==b['task'] and a['document_id']==b['document_id'] and a['answers']==b['answers']
        doc=a['document_id'];doc_task[doc].add(a['task'])
        groups[a['task']].append((doc,100*(a['student_score']-b['student_score'])))
    rng=np.random.default_rng(20261111);draws=np.zeros(repeat);point=0.
    if all(len(x)==1 for x in doc_task.values()):
        for task,rows in sorted(groups.items()):
            dd=collections.defaultdict(list)
            for d,x in rows:dd[d].append(x)
            sums=np.array([sum(x) for x in dd.values()]);counts=np.array([len(x) for x in dd.values()])
            point+=sums.sum()/counts.sum()/len(groups)
            for start in range(0,repeat,100):
                ix=rng.integers(len(dd),size=(min(100,repeat-start),len(dd)))
                draws[start:start+len(ix)]+=(sums[ix].sum(1)/counts[ix].sum(1))/len(groups)
    else:
        docs=sorted(doc_task);index={d:i for i,d in enumerate(docs)}
        sums={t:np.zeros(len(docs)) for t in groups};counts={t:np.zeros(len(docs)) for t in groups}
        for t,rr in groups.items():
            for d,x in rr:sums[t][index[d]]+=x;counts[t][index[d]]+=1
            point+=sums[t].sum()/counts[t].sum()/len(groups)
        for start in range(repeat):
            while True:
                ix=rng.integers(len(docs),size=len(docs))
                if all(counts[t][ix].sum()>0 for t in groups):break
            draws[start]=sum(sums[t][ix].sum()/counts[t][ix].sum() for t in groups)/len(groups)
    ci=np.quantile(draws,[.025,.975]).tolist()
    return dict(difference_points=float(point),image_cluster_95ci_points=ci,questions=len(aa),images=len(doc_task),
                tasks=len(groups),bootstrap_repetitions=repeat,bootstrap_seed=20261111,
                interpretation='positive' if ci[0]>0 else ('negative' if ci[1]<0 else 'not_detected'))


def average_seed_maps(pairs):
    keys=pairs[0][0].keys();assert all(a.keys()==b.keys()==keys for a,b in pairs)
    aa={};bb={}
    for key in keys:
        a,b=pairs[0][0][key],pairs[0][1][key]
        aa[key]=dict(a,student_score=sum(x[key]['student_score'] for x,y in pairs)/len(pairs))
        bb[key]=dict(b,student_score=sum(y[key]['student_score'] for x,y in pairs)/len(pairs))
    return aa,bb


def export_xlsx(results,pairs,mechanisms,timing,budgets,time_rows):
    import openpyxl
    wb=openpyxl.load_workbook(ROOT/'feedback/paper_experiment_tables.xlsx')
    # Accept the original workbook's serialized sheet names, but export English
    # titles. Unicode escapes preserve legacy identifiers, not displayed prose.
    legacy_titles = {
        'T1 Main results': 'T1 \u4e3b\u7ed3\u679c',
        'T2 Historical results': 'T2 \u5386\u53f2\u7ed3\u679c',
        'T3 Mechanism comparisons': 'T3 \u673a\u5236\u5bf9\u7167',
        'T4 Same-page question swap': 'T4 \u540c\u9875\u6362\u9898',
        'T5 System efficiency': 'T5 \u7cfb\u7edf\u6548\u7387',
        'T6 Paired comparisons and seeds': 'T6 \u914d\u5bf9\u4e0e\u79cd\u5b50',
    }
    for title, legacy in legacy_titles.items():
        if legacy in wb and title not in wb:
            wb[legacy].title = title
    s=wb['T1 Main results']
    # Replace historical screenshot estimates with unrounded machine aggregates.
    for row in range(6,34):
        method=s.cell(row,1).value;ratio=s.cell(row,3).value
        if method is None or not isinstance(ratio,(int,float)):continue
        key=(method,float(ratio),20261105)
        if key not in results:continue
        r=results[key];s.cell(row,5,'Formal per-question verification');s.cell(row,6,r['macro']);s.cell(row,7,r['macro']/57.00409130135063)
        for col,t in enumerate(TASKS,8):s.cell(row,col,r['by_task'].get(t))
    s=wb['T2 Historical results']
    import re
    for row in range(6,41):
        label=s.cell(row,1).value
        if not isinstance(label,str):continue
        match=re.fullmatch(r'([CT])-(10|25|50|75)-(pre|post)',label)
        if match:
            symbol,percent,pos=match.groups();ratio=int(percent)/100
            kind='quadtree_question' if symbol=='T' else 'col_adaptive'
            ss=load(ROOT/legacy(kind,ratio,'pre-merger' if pos=='pre' else 'post-merger')/'benchmark/summary.json')
            s.cell(row,5,ss['macro']['initial']);s.cell(row,6,ss['macro']['trained']);s.cell(row,7,ss['macro']['trained']/ss['macro']['teacher'])
            ci=ss['paired_trained_vs_initial']['cluster_bootstrap_95ci_points'];s.cell(row,8,ci[0]);s.cell(row,9,ci[1])
            for col,task in enumerate(TASKS,10):s.cell(row,col,ss['tasks'][task]['trained'])
        elif label in ['mean','selection','cell_mean','cell_selection']:
            ratio=float(s.cell(row,3).value)
            ss=load(ROOT/legacy('quadtree_question',ratio)/'benchmark/summary.json')
            s.cell(row,6,ss['macro'][label]);s.cell(row,7,ss['macro'][label]/ss['macro']['teacher'])
            for col,task in enumerate(TASKS,10):s.cell(row,col,ss['tasks'][task][label])
    s=wb['T1 Main results']
    for row in range(39,64):
        method=s.cell(row,1).value;ratio=s.cell(row,3).value
        key=(method,ratio,20261105)
        if key not in budgets:continue
        b=budgets[key]
        for col,key in [(4,'images'),(5,'N_mean'),(6,'target_mean'),(7,'retention_mean'),(8,'error_mean'),(9,'abs_error_mean'),(10,'actual_p10'),(11,'actual_p50'),(12,'actual_p90'),(13,'exact_fraction'),(14,'first_mean'),(15,'suffix_mean')]:s.cell(row,col,b[key])
    s=wb['T3 Mechanism comparisons']
    for row,method in [(6,'C'),(7,'V'),(8,'T'),(9,'J'),(10,'H'),(11,'A')]:
        r=results.get((method,.25,20261105))
        if not r:continue
        s.cell(row,8,r['macro'])
        for col,t in enumerate(TASKS,10):s.cell(row,col,r['by_task'].get(t))
    for row,method in [(6,'C'),(7,'V'),(8,'T'),(9,'J'),(10,'H'),(11,'A')]:
        if (method,.25,20261105) in results:s.cell(row,9,results[('T',.25,20261105)]['macro']-results[(method,.25,20261105)]['macro'])
    for row,name in [(15,'V-C_r0.25'),(16,'T-V_r0.25'),(17,'T-J_r0.25'),(18,'T-H_r0.25'),(19,'T-A_r0.25')]:
        if name in pairs:s.cell(row,3,pairs[name]['difference_points'])
    s=wb['T4 Same-page question swap']
    for row,v in [(6,'QQ'),(7,'QW'),(8,'WQ'),(9,'WW')]:
        r=results.get((v,.25,20261105))
        if not r:continue
        for col,val in [(5,r['images']),(6,r['rows']),(7,r['tasks']),(8,r['macro'])]:s.cell(row,col,val)
        for col,t in enumerate(TASKS,10):s.cell(row,col,r['by_task'].get(t,'NA'))
        if v in mechanisms:s.cell(row,18,mechanisms[v]['boundary_change_mean']);s.cell(row,19,mechanisms[v]['weight_l1_mean'])
    s=wb['T4 Same-page question swap']
    qq=results.get(('QQ',.25,20261105))
    if qq:
        for col,val in [(5,qq['images']),(6,qq['rows']),(7,qq['tasks']),(8,qq['macro'])]:s.cell(10,col,val)
        for col,t in enumerate(TASKS,10):s.cell(10,col,qq['by_task'].get(t,'NA'))
        for row,task in enumerate(TASKS,24):
            s.cell(row,2,mechanisms['QQ']['task_counts'].get(task,0));s.cell(row,3,mechanisms['QQ']['task_images'].get(task,0))
    for row,v in [(6,'QQ'),(7,'QW'),(8,'WQ'),(9,'WW')]:
        if (v,.25,20261105) in results:s.cell(row,9,results[('QQ',.25,20261105)]['macro']-results[(v,.25,20261105)]['macro'])
    for row,name in [(17,'QQ-WQ'),(18,'QQ-QW'),(19,'QQ-WW'),(20,'QQ-QW-WQ+WW')]:
        if name in pairs:
            r=pairs[name];s.cell(row,3,r['difference_points']);s.cell(row,4,r['image_cluster_95ci_points'][0]);s.cell(row,5,r['image_cluster_95ci_points'][1])
    s=wb['T5 System efficiency']
    for row,method in [(6,'REF'),(7,'T'),(8,'C'),(9,'M'),(10,'S'),(12,'G*'),(13,'R')]:
        r=timing.get(('Gstar' if method=='G*' else method)+'_natural_eos')
        if r is None:continue
        rr=[x for x in time_rows if x['method']==r['method'] and x['mode']=='natural_eos']
        s.cell(row,4,float(np.mean([x['first_layer_visual_tokens'] for x in rr])));s.cell(row,5,float(np.mean([x['suffix_visual_tokens'] for x in rr])))
        s.cell(row,6,float(np.mean([(x['first_layer_visual_tokens']+27*x['suffix_visual_tokens'])/28 for x in rr])))
        s.cell(row,7,r['median_output_tokens'])
        for col,val in enumerate(np.quantile([x['actual_tokens'] for x in rr],[.1,.5,.9]),8):s.cell(row,col,float(val))
        for col,key in [(11,'ttft_p50_ms'),(12,'ttft_p95_ms'),(13,'e2e_p50_ms'),(14,'e2e_p95_ms'),(15,'peak_allocated_gib'),(16,'median_e2e_speedup')]:s.cell(row,col,r[key])
        for col,key in [(17,'vision_and_native_merger_ms'),(18,'compression_extra_ms'),(19,'prefill_and_first_token_ms')]:s.cell(row,col,float(np.mean([x[key] for x in rr])))
        s.cell(row,20,float(np.mean([x['end_to_end_ms']-x['ttft_ms'] for x in rr])));s.cell(row,21,len({x['id'] for x in rr}))
    s.cell(20,2,'Hardware/software: see efficiency/runtime.rank*.json; model-resident batch1, NPU BF16, no visual cache')
    s.cell(21,2,'90 fixed QA by N tercile, 3 repeats; natural EOS and fixed64 separated in machine sheet')
    s.cell(22,2,'10 warmup requests/method; per-request allocated peak includes resident model; only current generator resident')
    s.cell(23,2,'S first block N then K; R actual processor budget logged per image')
    s=wb['T6 Paired comparisons and seeds']
    for row,seed in [(20,20261105),(21,20261106),(22,20261107)]:
        name=f'T-M_seed{seed}'
        if name not in pairs:continue
        p=pairs[name];s.cell(row,2,results[('T',.25,seed)]['macro']);s.cell(row,3,results[('M',.25,seed)]['macro']);s.cell(row,4,p['difference_points']);s.cell(row,5,p['image_cluster_95ci_points'][0]);s.cell(row,6,p['image_cluster_95ci_points'][1]);s.cell(row,7,p['questions']);s.cell(row,8,p['images'])
    main=pairs.get('T-M_three_seed_per_question_mean')
    if main:
        s.cell(31,2,main['difference_points']);s.cell(31,3,main['image_cluster_95ci_points'][0]);s.cell(31,4,main['image_cluster_95ci_points'][1]);s.cell(31,5,main['questions']);s.cell(31,6,main['images']);s.cell(31,8,'final_statistics.json')
        for col,method in [(2,'T'),(3,'M')]:
            vals=[results[(method,.25,seed)]['macro'] for seed in [20261105,20261106,20261107]]
            s.cell(23,col,float(np.mean(vals)));s.cell(24,col,float(np.std(vals,ddof=1)))
        vals=[pairs[f'T-M_seed{seed}']['difference_points'] for seed in [20261105,20261106,20261107]]
        s.cell(23,4,float(np.mean(vals)));s.cell(24,4,float(np.std(vals,ddof=1)))
    for row,name in [(6,'T-M_r0.25'),(7,'TP-M_r0.25'),(8,'T-S_r0.25'),(10,'T-G*_r0.25'),(11,'T-R_r0.25'),(12,'T-M_r0.5'),(13,'T-S_r0.5'),(15,'T-G*_r0.5'),(16,'T-R_r0.5')]:
        if name in pairs:
            r=pairs[name];s.cell(row,7,r['difference_points']);s.cell(row,8,r['image_cluster_95ci_points'][0]);s.cell(row,9,r['image_cluster_95ci_points'][1]);s.cell(row,10,r['questions']);s.cell(row,11,r['images']);s.cell(row,12,r['interpretation'])
            ratio=.5 if '_r0.5' in name else .25
            method_a='TP' if name.startswith('TP-') else 'T';method_b=name.split('-')[1].split('_')[0]
            s.cell(row,5,results[(method_a,ratio,20261105)]['macro']);s.cell(row,6,results[(method_b,ratio,20261105)]['macro'])
    # Always include machine-unrounded full exports, including rows not in template.
    for legacy in ['\u673a\u5668\u5b8c\u6574\u5206\u6570', '\u673a\u5668\u914d\u5bf9\u7edf\u8ba1', '\u673a\u5668\u6548\u7387']:
        if legacy in wb:
            del wb[legacy]
    for title in ['Full scores','Paired statistics','Efficiency']:
        if title in wb:del wb[title]
    sh=wb.create_sheet('Full scores');sh.append(['method','ratio','seed','macro','QA','images',*TASKS])
    for (method,ratio,seed),r in sorted(results.items()):sh.append([method,ratio,seed,r['macro'],r['rows'],r['images'],*[r['by_task'].get(t,'NA') for t in TASKS]])
    sh=wb.create_sheet('Paired statistics');sh.append(['comparison','difference','95CIlo','95CIhi','QA','images','bootstrap'])
    for name,r in pairs.items():sh.append([name,r['difference_points'],*r['image_cluster_95ci_points'],r['questions'],r['images'],r['bootstrap_repetitions']])
    sh=wb.create_sheet('Efficiency');sh.append(['method','mode','median_ttft_ms','p95_ttft_ms','median_e2e_ms','p95_e2e_ms','peak_allocated_GiB','median_output_tokens','QA_measurements'])
    for key,r in timing.items():sh.append([r['method'],r['mode'],r['ttft_p50_ms'],r['ttft_p95_ms'],r['e2e_p50_ms'],r['e2e_p95_ms'],r['peak_allocated_gib'],r['median_output_tokens'],r['measurements']])
    complete=load(BASE/'efficiency/complete.json') and all(('M',.25,seed) in results for seed in [20261105,20261106,20261107]) and len(load(BASE/'merged_results.json').get('results',{}))==16
    wb.save(BASE/('paper_experiment_tables_completed.xlsx' if complete else 'paper_experiment_tables_progress.xlsx'))


def plot_training(reg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=BASE/'plots';out.mkdir(exist_ok=True)
    for record in reg['runs']:
        if record['category']!='full_training':continue
        directory=ROOT/record['output'];rr=lines(directory/'training.jsonl')
        if not rr:continue
        fig,axes=plt.subplots(1,3,figsize=(14,3.6))
        epochs=list(range(5));val=[load(directory/f'validation.{"initial" if e==0 else "epoch"+str(e)}.summary.json') for e in epochs]
        train=[np.mean([x['kl'] for x in rr if x['epoch']==e]) for e in range(1,5)]
        axes[0].plot(range(1,5),train,'o-',label='train KL');axes[0].plot(epochs,[r.get('kl',float('nan')) for r in val],'o-',label='validation KL')
        axes[1].semilogy([r['step'] for r in rr],[max(r['grad_norm'],1e-12) for r in rr],alpha=.5,label='pre-clip grad norm')
        axes[1].semilogy([r['step'] for r in rr],[max(r['clip_multiplier'],1e-12) for r in rr],alpha=.5,label='clip multiplier')
        axes[2].plot(epochs,[r.get('student_macro',float('nan')) for r in val],'o-',label='validation generation macro')
        axes[2].axhline(val[0].get('teacher_macro',float('nan')),ls=':',label='teacher')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[0].set_xlabel('epoch');axes[1].set_xlabel('update');axes[2].set_xlabel('epoch')
        fig.suptitle(record['name']);fig.tight_layout();fig.savefig(out/(record['name']+'.pdf'));plt.close(fig)


def analyze():
    merged=load(BASE/'merged_results.json');reg=load(BASE/'registration.json');results={};maps={};mechanisms={};budgets={}
    def add(method,ratio,seed,rr):
        results[(method,ratio,seed)]=summary(list(rr.values()));maps[(method,ratio,seed)]=rr
        pages={}
        for row in rr.values():pages.setdefault(row['document_id'],row)
        pp=list(pages.values());nn=np.asarray([x['tokens'] for x in pp]);kk=np.asarray([max(1,round(x['tokens']*ratio)) for x in pp]);actual=np.asarray([x['compressed_tokens'] for x in pp])
        budgets[(method,ratio,seed)]=dict(images=len(pp),N_mean=float(nn.mean()),target_mean=float(kk.mean()),retention_mean=float((actual/nn).mean()),error_mean=float((actual-kk).mean()),abs_error_mean=float(abs(actual-kk).mean()),actual_p10=float(np.quantile(actual,.1)),actual_p50=float(np.median(actual)),actual_p90=float(np.quantile(actual,.9)),exact_fraction=float((actual==kk).mean()),first_mean=float(np.mean([x.get('first_layer_visual_tokens',x['compressed_tokens']) for x in pp])),suffix_mean=float(actual.mean()))
    for ratio in [.1,.25,.5,.75]:
        for method,kind,pos in [('T','quadtree_question','pre-merger'),('C','col_adaptive','pre-merger'),('TP','quadtree_question','post-merger')]:
            # Formal old answers have already passed their independent suite audit.
            rr=answers(ROOT/legacy(kind,ratio,pos)/'benchmark');add(method,ratio,20261105,rr)
            if method=='T' and ratio in [.25,.5]:
                for v in ['mean','selection','cell_mean','cell_selection']:
                    vv=answers(ROOT/legacy(kind,ratio,pos)/'benchmark',v);add(v,ratio,20261105,vv)
    add('global',.25,20261105,answers(ROOT/legacy('global_topk',.25)/'benchmark'))
    # Full original teacher as a separate score map.
    rr=maps[('T',.25,20261105)];tt={k:dict(r,student_score=r['teacher_score'],compressed_tokens=r['tokens'],first_layer_visual_tokens=r['tokens'],suffix_visual_tokens=r['tokens']) for k,r in rr.items()};add('REF',1.,20261105,tt)
    for record in reg['runs']:
        if record['name'] not in merged['results']:continue
        cfg=load(ROOT/record['config']);ratio=cfg['ratio'];seed=cfg.get('seed',20261105);method=cfg['method']
        for variant in cfg.get('variants',['trained']):
            rr=answers(ROOT/record['output']/'benchmark',variant);label=variant if method in ['HA','QSWAP'] else ('G*' if method=='Gstar' else method)
            add(label,ratio,seed,rr)
            if method=='QSWAP':mechanisms[variant]=dict(boundary_change_mean=float(np.mean([r['trace']['boundary_change'] for r in rr.values()])),weight_l1_mean=float(np.mean([r['trace']['weight_l1_mean'] for r in rr.values()])),task_counts=dict(collections.Counter(r['task'] for r in rr.values())),task_images={t:len({r['document_id'] for r in rr.values() if r['task']==t}) for t in TASKS})
    pairs={}
    def pair(name,a,b):
        if a in maps and b in maps:pairs[name]=paired_maps(maps[a],maps[b])
    for ratio in [.25,.5]:
        for method in ['M','S','G*','R','C','TP']:
            pair(f'T-{method}_r{ratio:g}',('T',ratio,20261105),(method,ratio,20261105))
        pair(f'TP-M_r{ratio:g}',('TP',ratio,20261105),('M',ratio,20261105))
        akey=('T',ratio,20261105);bkey=('R',ratio,20261105)
        if bkey in maps:
            common={k:r for k,r in maps[bkey].items() if r['compressed_tokens']==r['target_tokens']}
            if common:
                pairs[f'T-R_exact_budget_r{ratio:g}']=paired_maps({k:maps[akey][k] for k in common},common)
                pairs[f'T-R_exact_budget_r{ratio:g}']['subset_rule']='actual resized tokens equal target; no answer/score selection; same observed task set only'

    for method in ['V','J','H','A']:pair(f'T-{method}_r0.25',('T',.25,20261105),(method,.25,20261105))
    pair('V-C_r0.25',('V',.25,20261105),('C',.25,20261105))
    for v in ['QW','WQ','WW']:pair('QQ-'+v,('QQ',.25,20261105),(v,.25,20261105))
    if all((m,.25,20261105) in maps for m in ['QQ','QW','WQ','WW']):
        aa={k:dict(r,student_score=r['student_score']-maps[('QW',.25,20261105)][k]['student_score']) for k,r in maps[('QQ',.25,20261105)].items()}
        bb={k:dict(r,student_score=r['student_score']-maps[('WW',.25,20261105)][k]['student_score']) for k,r in maps[('WQ',.25,20261105)].items()}
        pairs['QQ-QW-WQ+WW']=paired_maps(aa,bb)
    seedpairs=[]
    for seed in [20261105,20261106,20261107]:
        a=('T',.25,seed);b=('M',.25,seed)
        pair(f'T-M_seed{seed}',a,b)
        if a in maps and b in maps:seedpairs.append((maps[a],maps[b]))
    seed_summary={}
    if len(seedpairs)==3:
        a,b=average_seed_maps(seedpairs);pairs['T-M_three_seed_per_question_mean']=paired_maps(a,b)
        gains=[pairs[f'T-M_seed{s}']['difference_points'] for s in [20261105,20261106,20261107]]
        seed_summary=dict(T_mean_macro=float(np.mean([results[('T',.25,z)]['macro'] for z in [20261105,20261106,20261107]])),T_sample_sd=float(np.std([results[('T',.25,z)]['macro'] for z in [20261105,20261106,20261107]],ddof=1)),M_mean_macro=float(np.mean([results[('M',.25,z)]['macro'] for z in [20261105,20261106,20261107]])),M_sample_sd=float(np.std([results[('M',.25,z)]['macro'] for z in [20261105,20261106,20261107]],ddof=1)),mean_difference_points=float(np.mean(gains)),sample_sd_difference_points=float(np.std(gains,ddof=1)),paired_point_matches=bool(abs(np.mean(gains)-pairs['T-M_three_seed_per_question_mean']['difference_points'])<1e-10))
    timing={};time_rows=[r for p in (BASE/'efficiency').glob('timings.rank*.jsonl') for r in lines(p)]
    if (BASE/'efficiency/complete.json').exists():
        groups=collections.defaultdict(list)
        for r in time_rows:groups[(r['method'],r['mode'])].append(r)
        for (method,mode),rr in groups.items():
            timing[method+'_'+mode]=dict(method=method,mode=mode,ttft_p50_ms=float(np.median([r['ttft_ms'] for r in rr])),ttft_p95_ms=float(np.quantile([r['ttft_ms'] for r in rr],.95)),e2e_p50_ms=float(np.median([r['end_to_end_ms'] for r in rr])),e2e_p95_ms=float(np.quantile([r['end_to_end_ms'] for r in rr],.95)),peak_allocated_gib=max(r['peak_allocated_gib'] for r in rr),median_output_tokens=float(np.median([r['new_tokens'] for r in rr])),measurements=len(rr))
        for r in timing.values():r['median_e2e_speedup']=timing['REF_'+r['mode']]['e2e_p50_ms']/r['e2e_p50_ms']
    exported=[dict(method=m,ratio=r,seed=s,**v) for (m,r,s),v in sorted(results.items())]
    all_complete=bool(timing) and len(seedpairs)==3 and len(merged['results'])==16
    save(BASE/('final_statistics.json' if all_complete else 'partial_statistics.json'),dict(results=exported,paired=pairs,seed_summary=seed_summary,mechanisms=mechanisms,timing=timing,budgets=[dict(method=m,ratio=r,seed=s,**v) for (m,r,s),v in budgets.items()],analysis_source_sha256=sha(Path(__file__))))
    export_xlsx(results,pairs,mechanisms,timing,budgets,time_rows)
    plot_training(reg)
    doc=['# Supplementary Experiments: Full Paired Statistics and System Efficiency','', 'Use only question-audited formal results. Differences are pp; image bootstrap and training-seed variance are separate. Primary comparison is 25% T(pre)−M(post); 50%/others are secondary/exploratory. Multiple unadjusted CIs do not establish simultaneous confirmatory advantages.','', '|Method|Budget|seed|Macro average|DocVQA|ChartQA|DeepForm|OCRBench|','|---|---|---:|---:|---:|---:|---:|---:|']
    for x in exported:doc.append(f'|{x["method"]}|{x["ratio"]:.0%}|{x["seed"]}|{x["macro"]:.4f}|{x["by_task"].get("DocVQA",float("nan")):.4f}|{x["by_task"].get("ChartQA",float("nan")):.4f}|{x["by_task"].get("DeepForm",float("nan")):.4f}|{x["by_task"].get("OCRBench",float("nan")):.4f}|')
    doc+=['','## Paired Differences Across Methods and Mechanisms','', '|Comparison|Difference|95% image-cluster CI|QA|Images|Interval direction|','|---|---:|---|---:|---:|---|']
    for k,r in pairs.items():doc.append(f'|{k}|{r["difference_points"]:.4f}|[{r["image_cluster_95ci_points"][0]:.4f}, {r["image_cluster_95ci_points"][1]:.4f}]|{r["questions"]}|{r["images"]}|{r["interpretation"]}|')
    doc+=['','## Three Training Seeds','',json.dumps(seed_summary,ensure_ascii=False),'','## System Efficiency','', '|Method|Generation settings|TTFT p50 ms|TTFT p95 ms|E2E p50 ms|E2E p95 ms|Peak allocated (GiB)|Speed relative to REF|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in timing.values():doc.append(f'|{r["method"]}|{r["mode"]}|{r["ttft_p50_ms"]:.2f}|{r["ttft_p95_ms"]:.2f}|{r["e2e_p50_ms"]:.2f}|{r["e2e_p95_ms"]:.2f}|{r["peak_allocated_gib"]:.3f}|{r["median_e2e_speedup"]:.3f}|')
    primary=pairs.get('T-M_three_seed_per_question_mean')
    if primary:
        direction={'positive':'Supports higher T macro under this protocol/budget and three evaluated seeds','negative':'Lower T macro under this protocol; superiority to MetaCompress is unsupported','not_detected':'No clear difference detected; superiority and equivalence are unsupported'}[primary['interpretation']]
        doc+=['','## Interpretation of the Measured Results','',f'Primary T−M difference: {primary["difference_points"]:+.4f} percentage points; 95% image CI {primary["image_cluster_95ci_points"]}: {direction}. Sample SD across training seeds is reported separately, not replaced by image intervals.','', 'Question-path/leaf-aggregation interpretations are fixed-checkpoint interventions; H/A are not freshly trained methods. Secondary intervals are exploratory, not simultaneous confirmation. Judge natural/fixed64 speed separately on matched hardware/requests; record slower-than-REF results, not token compression as speedup.']
    doc+=['','[Training/validation and gradient curves](../artifacts/compression_supplement_20261111/plots/). ','', '[Complete machine-readable statistics](../artifacts/compression_supplement_20261111/final_statistics.json) · [Completed workbook and full-precision supplementary tables](../artifacts/compression_supplement_20261111/paper_experiment_tables_completed.xlsx). The original feedback workbook is not overwritten.']
    if not all_complete:
        doc.insert(2,'**Historical interim audit: additional experiments remain incomplete; this is not the final supplementary outcome.**')
        doc=[line.replace('final_statistics.json','partial_statistics.json').replace('paper_experiment_tables_completed.xlsx','paper_experiment_tables_progress.xlsx') for line in doc]
    (ROOT/'docs/COMPRESSION_SUPPLEMENT_STATISTICS.md').write_text('\n'.join(doc)+'\n')
    save(BASE/('analysis_complete.json' if all_complete else 'analysis_progress.json'),dict(paired_comparisons=len(pairs),three_seed_complete=len(seedpairs)==3,timing_complete=bool(timing),xlsx=True,source_sha256=sha(Path(__file__))))

if __name__=='__main__':analyze()
