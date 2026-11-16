"""Standalone training and validation figures from recorded observations only."""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.report_meta_compress_search import BASE,ROOT,read,collect


def main():
    for run in sorted((BASE/'runs').glob('*')):
        log=run/'training.jsonl'
        if run.name=='smoke' or not log.exists():continue
        records=read(log)
        if not records:continue
        data=collect(run);per_epoch=math.ceil(len(read(ROOT/data['config']['train_rows']))/data['config']['batch_size'])
        epochs=sorted(e for e in {r['epoch'] for r in records} if len({r['step'] for r in records if r['epoch']==e})==per_epoch)
        stages=data['stages']
        fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
        for ax,key,label in zip(axes.flat,['prediction','objective','entropy','collapse'],
                ['Teacher-prefix KL (nats)','Weighted objective','Mean row entropy','Max column mass']):
            xs=[r['step'] for r in records];ys=[r[key] for r in records]
            ax.plot(xs,ys,lw=.5,alpha=.35,label='train batch')
            epx=[max(r['step'] for r in records if r['epoch']==e) for e in epochs]
            epy=[np.mean([r[key] for r in records if r['epoch']==e]) for e in epochs]
            ax.plot(epx,epy,'o-',label='train completed-epoch mean')
            vx=[];vy=[]
            for stage,s in stages.items():
                if not s['complete']:continue
                name=stage.split('/')[0]
                if name=='initial':step=0
                elif name.startswith('epoch'):
                    ep=int(name[5:]);matches=[r['step'] for r in records if r['epoch']==ep]
                    if not matches:continue
                    step=max(matches)
                else:continue
                vx.append(step);vy.append(s[key])
            if vx:
                order=np.argsort(vx);ax.plot(np.array(vx)[order],np.array(vy)[order],'s-',label='held-out search')
            ax.set(xlabel='Optimizer step',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax=axes.flat[4]
        for key,label in [('score','student'),('teacher','teacher')]:
            points=[]
            for stage,s in stages.items():
                if not s['complete']:continue
                name=stage.split('/')[0]
                if name=='initial':epoch=0
                elif name.startswith('epoch'):epoch=int(name[5:])
                else:continue
                points.append((epoch,s[key]))
            if points:
                points.sort();ax.plot(*zip(*points),'o-',label=label)
        ax.set(xlabel='Epoch',ylabel='Search native-score macro (%)',ylim=(0,100));ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax=axes.flat[5];ax.plot([r['step'] for r in records],[r['lr'] for r in records],label='LR')
        ax.set(xlabel='Optimizer step',ylabel='Learning rate');ax.grid(alpha=.2)
        twin=ax.twinx();twin.plot([r['step'] for r in records],[r['clip_multiplier'] for r in records],color='orange',alpha=.6)
        twin.set(ylabel='Clip multiplier',ylim=(0,1.01))
        fig.suptitle(run.name)
        fig.savefig(run/'curves.png',dpi=160);fig.savefig(run/'curves.pdf');plt.close(fig)


if __name__=='__main__':main()
