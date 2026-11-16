"""Join frozen decoder-free losses to independent, unaveraged VQA trial outcomes."""
import argparse,collections,csv,hashlib,json,warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/token_proxy/random_v1_20x';OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def corr(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])
def rho(x,y):return corr(stats.rankdata(x),stats.rankdata(y))
def rankcenter(x,axis):
    r=stats.rankdata(x,axis=axis);return r-r.mean(axis=axis,keepdims=True)
def bootcorr(x,y,ixdraw):
    # Exact Pearson bootstrap via per-image sufficient statistics. For pooled
    # Spearman ranks are recomputed separately below, not frozen implicitly.
    shape=x.shape;x=x.reshape(len(x),-1);y=y.reshape(len(y),-1)
    suff=np.stack([x.sum(1),y.sum(1),(x*x).sum(1),(y*y).sum(1),(x*y).sum(1)],1)
    sums=suff[ixdraw].sum(1);n=ixdraw.shape[1]*x.shape[1]
    cov=sums[:,4]-sums[:,0]*sums[:,1]/n;vx=sums[:,2]-sums[:,0]**2/n;vy=sums[:,3]-sums[:,1]**2/n
    values=cov/np.sqrt(np.maximum(vx*vy,1e-30));return list(map(float,np.quantile(values,[.025,.975])))

def analyze(item):
    name,x,y,teacher,tasks,ratios,families=item
    rng=np.random.default_rng(20261022);draw=rng.integers(len(x),size=(1000,len(x)))
    xx=x.reshape(len(x),12,2);yy=y.reshape(len(y),12,2)
    xpair=rankcenter(xx,2);ypair=rankcenter(yy,2)
    xq=rankcenter(x,1);yq=rankcenter(y,1)
    xr=rankcenter(xx.reshape(len(x),6,4),2);yr=rankcenter(yy.reshape(len(y),6,4),2)
    dx=xx[:,:,1]-xx[:,:,0];dy=yy[:,:,1]-yy[:,:,0]
    con=((dx*dy)>0).sum(1);dis=((dx*dy)<0).sum(1);count=con+dis;bc=con[draw].sum(1);bn=count[draw].sum(1)
    ci=list(map(float,np.quantile(bc/np.maximum(1,bn),[.025,.975])))
    # True image-cluster bootstrap with reranking for pooled Spearman.
    pooled=[]
    for ix in draw:
        v=rho(x[ix].ravel(),y[ix].ravel())
        if v is not None:pooled.append(v)
    # Exact two-way residualization of global ranks, balanced complete design.
    rx=stats.rankdata(x).reshape(x.shape);ry=stats.rankdata(y).reshape(y.shape)
    def residual(r):return r-r.mean(1,keepdims=True)-r.reshape(len(x),12,2).mean((0,2))[None,:,None].repeat(2,2).reshape(1,24)+r.mean()
    result={'loss':name,'pooled_spearman':rho(x.ravel(),y.ravel()),'pooled_image_bootstrap_ci95':list(map(float,np.quantile(pooled,[.025,.975]))),
      'teacher_positive_spearman':rho(x[teacher>0].ravel(),y[teacher>0].ravel()),'teacher_perfect_spearman':rho(x[teacher==1].ravel(),y[teacher==1].ravel()),
      'within_question':corr(xq.ravel(),yq.ravel()),'within_question_ci95':bootcorr(xq,yq,draw),
      'within_question_ratio':corr(xr.ravel(),yr.ravel()),'within_question_ratio_family':corr(xpair.ravel(),ypair.ravel()),'within_question_ratio_family_ci95':bootcorr(xpair,ypair,draw),
      'partial_global_rank_question_config':corr(residual(rx).ravel(),residual(ry).ravel()),
      'paired_seed_ordering':{'pairs':int(dx.size),'concordant':int(con.sum()),'discordant':int(dis.sum()),'outcome_ties':int((dy==0).sum()),'loss_ties_outcome_nontie':int(((dx==0)&(dy!=0)).sum()),'accuracy':float(con.sum()/max(1,count.sum())),'image_bootstrap_ci95':ci},
      'loss_quantiles':list(map(float,np.quantile(x,[0,.01,.25,.5,.75,.99,1]))),
      'per_task':{},'per_ratio_family':{},'controls':{},'high_retention':{}}
    for task in sorted(set(tasks)):
        mask=tasks==task
        paircon=con[mask].sum();pairdis=dis[mask].sum()
        result['per_task'][task]={'images':int(mask.sum()),'pooled_spearman':rho(x[mask].ravel(),y[mask].ravel()),'within_question_config':corr(xpair[mask].ravel(),ypair[mask].ravel()),'seed_order_accuracy':float(paircon/max(1,paircon+pairdis)),'nontie_pairs':int(paircon+pairdis),
                               'teacher_positive_rho':rho(x[mask&(teacher>0)].ravel(),y[mask&(teacher>0)].ravel())}
    for i in range(12):
        key=f'{ratios[i*2]}/{families[i*2]}'
        result['per_ratio_family'][key]={'pooled_spearman':rho(xx[:,i].ravel(),yy[:,i].ravel()),'within_question':corr(xpair[:,i].ravel(),ypair[:,i].ravel())}
    high=np.array(ratios)>=.625
    result['high_retention']={'pooled_spearman':rho(x[:,high].ravel(),y[:,high].ravel()),'within_question_config':corr(xpair[:,np.array(ratios)[::2]>=.625].ravel(),ypair[:,np.array(ratios)[::2]>=.625].ravel())}
    print(json.dumps({'analyzed':name,'rho':result['pooled_spearman'],'conditional':result['within_question_ratio_family'],'ordering':result['paired_seed_ordering']['accuracy']}),flush=True)
    return name,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--require-exact',action='store_true');p.add_argument('--require-line',action='store_true');a=p.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text());samples=[json.loads(s) for s in (BASE/'samples.jsonl').read_text().splitlines()]
    assert hashlib.sha256((BASE/'samples.jsonl').read_bytes()).hexdigest()==protocol['samples_sha256']
    assert hashlib.sha256((ROOT/'vqa/token_proxy.py').read_bytes()).hexdigest()==protocol['generator_sha256']
    losses={};diagnostics=[]
    for sample in samples:
        d=json.loads((OUT/'losses'/f"{sample['document_id']}.json").read_text())
        for r in d['trials']:assert r['id'] not in losses;losses[r['id']]=r['loss'];diagnostics.append(r['diagnostics'])
    for directory,required in [('exact_ot',a.require_exact),('line_losses',a.require_line)]:
        complete=all((OUT/directory/f"{r['document_id']}.json").exists() for r in samples)
        if required:assert complete,f'{directory} incomplete'
        if complete:
            for sample in samples:
                d=json.loads((OUT/directory/f"{sample['document_id']}.json").read_text())
                for r in d['trials']:losses[r['id']].update(r['loss'])
    rows=[]
    for path in sorted(BASE.glob('trials.rank*.jsonl')):
        for r in map(json.loads,path.read_text().splitlines()):
            old=r['loss'];r['loss']={**losses[r['id']],'V1':old['total'],'V1_rec':old['rec']}
            if 'evidence_line_question' in r['loss']:
                r['loss']['evidence_combined_line_question']=r['loss']['evidence_text_question']+r['loss']['evidence_number_question']+r['loss']['evidence_line_question']
            rows.append(r)
    assert len(rows)==93626 and len(losses)==93626
    fields=['id','sample_id','document_id','task','question','image','answers','kind','family','ratio','replicate','matrix_seed','matrix_file','teacher_prediction','teacher_score','student_prediction','student_score','score_drop'];names=list(rows[0]['loss'])
    with (OUT/'trials.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields+['loss_'+n for n in names]);w.writeheader()
        for r in rows:w.writerow({**{k:r[k] for k in fields},**{'loss_'+k:v for k,v in r['loss'].items()}})
    rr=sorted([r for r in rows if r['kind']=='random'],key=lambda r:(r['document_id'],r['ratio'],r['family'],r['replicate']))
    assert len(rr)==86424
    y=np.array([r['score_drop'] for r in rr]).reshape(-1,24);teacher=np.array([r['teacher_score'] for r in rr])[::24];tasks=np.array([r['task'] for r in rr])[::24]
    ratios=[r['ratio'] for r in rr[:24]];families=[r['family'] for r in rr[:24]]
    items=[(n,np.array([r['loss'][n] for r in rr]).reshape(-1,24),y,teacher,tasks,ratios,families) for n in names]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:metrics=dict(pool.map(analyze,items))
    for name in names:
        for rate in [.25,.5]:
            cr=[r for r in rows if r['kind']=='mean_pool_control' and r['ratio']==rate];metrics[name]['controls'][str(rate)]=rho([r['loss'][name] for r in cr],[r['score_drop'] for r in cr])
    statsdiag={k:{'min':float(min(d[k] for d in diagnostics)),'median':float(np.median([d[k] for d in diagnostics])),'max':float(max(d[k] for d in diagnostics))} for k in diagnostics[0]}
    summary={'protocol':protocol,'images':len(y),'random_trials':len(rr),'controls':len(rows)-len(rr),'bootstrap_repeats':1000,'student_zero_scores':sum(r['student_score']==0 for r in rr),'teacher_positive_images':int((teacher>0).sum()),'metrics':metrics,'diagnostics':statsdiag,
       'per_task':{t:{'images':int(sum(tasks==t)),'teacher_mean':float(teacher[tasks==t].mean()),'zero_student_trials':sum(r['student_score']==0 for r in rr if r['task']==t)} for t in sorted(set(tasks))},
       'source_audit':json.loads((BASE/'audit.json').read_text()),'code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['vqa/encoder_candidate_losses.py','vqa/encoder_evidence.py','scripts/eval_encoder_candidates.py','scripts/train_encoder_evidence.py','scripts/exact_encoder_transport_npu.py','scripts/train_encoder_relation.py','scripts/eval_encoder_relation.py','scripts/report_encoder_candidates.py']},'head_sha256':hashlib.sha256((OUT/'evidence_head.pt').read_bytes()).hexdigest(),'line_head_sha256':hashlib.sha256((OUT/'line_head.pt').read_bytes()).hexdigest() if a.require_line else None,'has_full_token_OT':a.require_exact,'has_line_probe':a.require_line}
    if a.require_exact:
        ot_diags=[json.loads((OUT/'exact_ot'/f"{s['document_id']}.json").read_text()) for s in samples]
        # Each trial diagnostic includes the teacher-self solves cached per image.
        dd=[r for d in ot_diags for r in d['trials']]
        summary['full_token_transport_diagnostics']={k:{'min':float(min(d[k] for d in dd)),'median':float(np.median([d[k] for d in dd])),'max':float(max(d[k] for d in dd))} for k in ['max_marginal_error','max_iterations','log_fallback']}
        summary['full_token_transport_diagnostics']['trials_above_1e_6']=sum(d['max_marginal_error']>1e-6 for d in dd)
        summary['full_token_transport_diagnostics']['trial_count']=len(dd)
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # Preserve counterexamples as complete independent paired trials.
    examples={}
    for name in ['coverage_pos_question','sinkhorn_full_question_excess' if a.require_exact else 'sinkhorn_question_excess','subspace_question','evidence_combined_line_question' if a.require_line else 'evidence_combined_question']:
        candidates=[]
        for i in range(0,len(rr),2):
            a1,b1=sorted(rr[i:i+2],key=lambda r:r['loss'][name])
            if a1['loss'][name]<b1['loss'][name] and a1['score_drop']>b1['score_drop']:
                candidates.append({'loss_name':name,'low_loss':a1,'high_loss':b1})
        candidates.sort(key=lambda p:p['low_loss']['score_drop']-p['high_loss']['score_drop'],reverse=True)
        examples[name]=candidates[:8]
    (OUT/'counterexamples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    plot(summary,rr)

def plot(summary,rr):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['coverage_pos_question','sinkhorn_full_question_excess' if summary['has_full_token_OT'] else 'sinkhorn_question_excess','evidence_combined_line_question' if summary['has_line_probe'] else 'evidence_combined_question','subspace_question']
    fig,axes=plt.subplots(2,2,figsize=(12,8));rng=np.random.default_rng(24);idx=rng.choice(len(rr),size=min(16000,len(rr)),replace=False)
    for ax,n in zip(axes.flat,names):
        for family,col in [('local','#2175ad'),('global','#d8752b')]:
            rows=[rr[i] for i in idx if rr[i]['family']==family];ax.scatter([r['loss'][n] for r in rows],[r['score_drop'] for r in rows],s=3,alpha=.2,color=col,label=family,rasterized=True)
        m=summary['metrics'][n];ax.set_title(f"{n}\npooled={m['pooled_spearman']:.3f}; conditional={m['within_question_ratio_family']:.3f}");ax.set_xlabel('Loss');ax.set_ylabel('Teacher score - student score')
    axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'correlation_scatter.pdf');fig.savefig(OUT/'correlation_scatter.png',dpi=140);plt.close(fig)
    h=json.loads((OUT/'probe_metrics.json').read_text())['history'];fig,ax=plt.subplots(figsize=(6,4));ax.plot([r['epoch'] for r in h],[r['training_weighted_BCE'] for r in h],label='Train');ax.plot([r['epoch'] for r in h],[r['holdout_weighted_BCE'] for r in h],label='Image-held-out eval');ax.set_xlabel('Epoch');ax.set_ylabel('Weighted binary cross entropy');ax.legend();fig.tight_layout();fig.savefig(OUT/'probe_training_loss.pdf');plt.close(fig)
if __name__=='__main__':
    import multiprocessing;multiprocessing.set_start_method('spawn');main()
