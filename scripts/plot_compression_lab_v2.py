"""Publish fixed-cohort train/validation KL curves and actual generation score curves."""
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.report_compression_lab_v2 import ROOT,BASE,load,controls,export_answers,read
def overview(out):
 records=[r for r in load(BASE/'summary.json').get('records',[]) if r['phase']=='v2_screen']
 if not records:return
 names=list(dict.fromkeys(r['kind'] for r in records));columns=[(ratio,pos) for ratio in [.1,.25,.5] for pos in ['pre-merger','post-merger']]
 lookup={(r['kind'],r['ratio'],r['insertion']):r for r in records};scores=np.full((len(names),len(columns)),np.nan)
 for i,name in enumerate(names):
  for j,(ratio,pos) in enumerate(columns):
   value=lookup.get((name,ratio,pos),{}).get('final_score')
   if value is not None:scores[i,j]=value
 teacher=next((r['teacher_score'] for r in records if r.get('teacher_score') is not None),100.)
 fig,ax=plt.subplots(figsize=(10,13));im=ax.imshow(scores,aspect='auto',vmin=0,vmax=teacher,cmap='viridis')
 ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(columns)),[f'keep {ratio:.0%}\n{pos}' for ratio,pos in columns])
 for i,name in enumerate(names):
  for j,(ratio,pos) in enumerate(columns):
   value=scores[i,j]
   if np.isfinite(value):
    passed=lookup[(name,ratio,pos)]['gate']
    ax.text(j,i,f'{value:.1f}'+(' *' if passed else ''),ha='center',va='center',fontsize=8,color='black' if value>teacher*.6 else 'white')
 ax.set_title(f'24 new generators: fixed epoch6 selection scores\nTeacher = {teacher:.2f}; * = pilot gate passed, not independent confirmation')
 fig.colorbar(im,ax=ax,label='7-task native macro (%)',shrink=.6);fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(out/f'screen_score_matrix.{ext}',dpi=160)
 plt.close(fig)
def main():
 configs=[load(ROOT/p) for name in ['screen_configs.json','expanded_configs.json','full_configs.json'] for p in load(BASE/name)]
 out=BASE/'plots';out.mkdir(exist_ok=True)
 for phase in sorted({c['phase'] for c in configs}):
  for position in ['pre-merger','post-merger']:
   for ratio in [.1,.25,.5]:
    selected=[c for c in configs if c['phase']==phase and c['insertion']==position and c['ratio']==ratio]
    if not selected:continue
    nr=math.ceil(len(selected)/4);fig,axs=plt.subplots(nr,4,figsize=(17,3.1*nr),squeeze=False)
    sf,saxs=plt.subplots(nr,4,figsize=(17,3.1*nr),squeeze=False)
    for cfg,ax,sax in zip(selected,axs.flat,saxs.flat):
     directory=ROOT/cfg['output']
     if phase=='v2_full_20k':
      batches=defaultdict(list)
      for record in read(directory/'training.jsonl'):batches[record['epoch']].append(record)
      # Original full split has exactly20,000QA and global batch32, no padding.
      per_epoch=math.ceil(len(read(ROOT/cfg['train_rows']))/cfg['batch_size'])
      complete={e:rr for e,rr in batches.items() if len(rr)==per_epoch and len({r['step'] for r in rr})==per_epoch}
      xx=sorted(complete);yy=[sum(r['kl'] for r in complete[e])/per_epoch for e in xx]
      ax.plot(xx,yy,'s-',label='all train batches: epoch mean',markersize=3)
     for cohort,label in [('train_audit','fixed train audit'),('validation','selection validation')]:
      xx=[];yy=[]
      for epoch in range(cfg['epochs']+1):
       v=load(directory/f'{cohort}.{"initial" if epoch==0 else "epoch"+str(epoch)}.summary.json')
       if v:xx.append(epoch);yy.append(v['kl'])
      ax.plot(xx,yy,'o-',label=label,markersize=3)
     xx=[];yy=[];teacher=None
     for epoch in range(cfg['epochs']+1):
      v=load(directory/f'validation.{"initial" if epoch==0 else "epoch"+str(epoch)}.summary.json')
      if 'student_macro' in v:xx.append(epoch);yy.append(v['student_macro']);teacher=v['teacher_macro']
     sax.plot(xx,yy,'o-',markersize=3,label='student')
     cc=controls(ratio)
     if None not in cc:sax.axhline(max(cc),ls='--',color='grey',label='best of 8 fixed')
     if teacher is not None:sax.axhline(teacher,ls=':',color='green',label='uncompressed teacher')
     title=cfg['kind']+(f' seed{cfg["seed"]}' if phase!='v2_screen' else '')
     ax.set_title(title,fontsize=9);sax.set_title(title,fontsize=9)
     ax.set_ylabel('teacher-prefix KL');sax.set_ylabel('native task macro (%)')
     for a in [ax,sax]:a.set_xlabel('epoch');a.grid(alpha=.2);a.legend(fontsize=6)
    for a in list(axs.flat)[len(selected):]+list(saxs.flat)[len(selected):]:a.set_visible(False)
    for f,metric in [(fig,'loss'),(sf,'score')]:
     f.suptitle(f'Frozen UND v2: {phase}, {position}, keep {ratio:.0%}')
     f.tight_layout(rect=(0,0,1,.97));f.savefig(out/f'{metric}_{phase}_{position}_keep{ratio:g}.pdf');plt.close(f)
 overview(out);export_answers()
 from scripts.review_compression_lab_v2 import main as review
 review()
 from scripts.review_compression_lab_v2_benchmark import main as benchmark_review
 benchmark_review();print(out)
if __name__=='__main__':main()
