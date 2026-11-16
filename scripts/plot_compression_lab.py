"""Epoch-matched fixed audit / independent validation KL and generation scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/compression_lab'
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def main():
 configs=[load(ROOT/p) for filename in ['screen_configs.json','followup_configs.json','pre_configs.json','expanded_configs.json'] for p in load(BASE/filename)];out=BASE/'plots';out.mkdir(exist_ok=True)
 for phase in sorted({c.get('phase','initial_screen') for c in configs}):
  for ratio in [.1,.25,.5]:
   selected=[c for c in configs if c['ratio']==ratio and c.get('phase','initial_screen')==phase]
   if not selected:continue
   import math
   nr=math.ceil(len(selected)/4)
   fig,axs=plt.subplots(nr,4,figsize=(17,3.5*nr),squeeze=False);scorefig,saxs=plt.subplots(nr,4,figsize=(17,3.5*nr),squeeze=False)
   for cfg,ax,sax in zip(selected,axs.flat,saxs.flat):
    directory=ROOT/cfg['output'];folder='baselines_pre' if cfg.get('insertion')=='pre-merger' else 'baselines';epochs=list(range(cfg['epochs']+1))
    for cohort,label in [('train_audit','fixed train audit'),('validation','independent val')]:
     xx=[];yy=[]
     for epoch in epochs:
      val=load(directory/f'{cohort}.{"initial" if not epoch else "epoch"+str(epoch)}.summary.json')
      if val:xx.append(epoch);yy.append(val['kl'])
     ax.plot(xx,yy,'o-',label=label,markersize=3)
    xx=[];yy=[]
    for epoch in epochs:
     val=load(directory/f'validation.{"initial" if not epoch else "epoch"+str(epoch)}.summary.json')
     if val:xx.append(epoch);yy.append(val['student_macro'])
    sax.plot(xx,yy,'o-',markersize=3)
    baseline=[load(BASE/folder/f'{kind}_r{ratio:g}'/'validation.fixed.summary.json').get('student_macro') for kind in ['mean','selection']]
    for v,label,col in zip(baseline,['mean','selection'],['grey','orange']):
     if v is not None:sax.axhline(v,ls='--',label=label,color=col,linewidth=.8)
    ax.set_title(cfg['kind'],fontsize=9);sax.set_title(cfg['kind'],fontsize=9)
    for axis in [ax,sax]:axis.set_xlabel('epoch');axis.grid(alpha=.2);axis.legend(fontsize=6)
    ax.set_ylabel('teacher-prefix KL');sax.set_ylabel('native macro score (%)')
   fig.suptitle(f'Frozen UND: keep {ratio:.0%}; matched fixed-cohort KL');scorefig.suptitle(f'Frozen UND: keep {ratio:.0%}; greedy answers on independent images')
   fig.tight_layout(rect=(0,0,1,.98));scorefig.tight_layout(rect=(0,0,1,.98))
   fig.savefig(out/f'loss_{phase}_keep{ratio:g}.pdf');scorefig.savefig(out/f'score_{phase}_keep{ratio:g}.pdf');plt.close(fig);plt.close(scorefig)
 print(out)
if __name__=='__main__':main()
