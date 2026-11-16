"""Compact read-only status, only committed log records and real completions."""
import json,subprocess
from scripts.tessfold_repair.common import ROOT,load,save,now
from scripts.tessfold_accel.live_io import live_lines
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    jobs=subprocess.check_output(['squeue','-h','-u','jchen524','-o','%i|%j|%T|%M|%R'],text=True).strip().splitlines()
    training=[]
    for label in ['qwen_full_pass1','qwen_full_tree25','qwen_full_stable50']:
        for p in sorted((BASE/label).glob('*_r*')):
            m=load(p/'manifest.json')
            if not m:continue
            rows=live_lines(p/'training.jsonl');last=rows[-1] if rows else {};total=__import__('math').ceil(m['train_questions']/m['batch'])*m['epochs']
            epochs=[load(f) for f in sorted(p.glob('epoch*.summary.json'))];best=max(epochs,key=lambda r:r['macro']) if epochs else {}
            training.append(dict(run=label+'/'+p.name,step=last.get('step',0),total=total,complete=bool(load(p/'complete.json')),
                initial=load(p/'initial.summary.json').get('macro'),best_observed_dev=best.get('macro'),best_observed_stage=best.get('stage'),
                recent_gradient_norm=last.get('gradient_norm')))
    stages={}
    for label in ['prefix_formal','hard_formal','tiled_vision_pilot','qwen_memory_smoke','qwen_formal','qwen_timing','qwen_memory']:
        stages[label]=dict(records=sum(len(live_lines(p)) for p in (BASE/label).glob('rows.rank*.jsonl')),complete=bool(load(BASE/label/'complete.json')))
    result=dict(time=now(),jobs=jobs,training=training,stages=stages)
    save(BASE/'watch_snapshot.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
