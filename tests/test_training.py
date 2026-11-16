"""Tiny synthetic frozen decoder: test execution, not a claimed VQA result."""
import json
from pathlib import Path
import torch
from torch import nn
import pytest
from tessfold.runtime import lr_factor,shard_batch
from tessfold.training import run_train,run_evaluate,eligible
from tessfold.config import load_config
from tests.test_config import fixture_config


class TinyEngine:
    def __init__(self,cfg,device):
        self.device=device
        with torch.random.fork_rng():
            torch.manual_seed(127)
            self.model=nn.Linear(4096,8,bias=False).to(device).eval().requires_grad_(False)
    def extract(self,row):
        g=torch.Generator().manual_seed(sum(map(ord,row['id'])))
        return dict(x=torch.randn(8,4096,generator=g))
    def prepare(self,row,features):
        return dict(x=features['x'].to(self.device),shape=(2,4),question=torch.ones(2,4096,device=self.device))
    def route(self,router,d,ratio):return router(d['x'],d['shape'],d['question'],ratio)
    def logits(self,d,response,route=None):
        x=d['x'] if route is None else route[0]@d['x']
        weighted=x*torch.linspace(.1,1,len(x),device=self.device)[:,None]
        return self.model(weighted.mean(0))[None,None].expand(1,len(response),-1)
    def greedy(self,d,route=None,cap=64):return [int(self.logits(d,[0],route).argmax(-1))]
    def decode(self,ids):return str(ids[0])
    def token_count(self,d):return len(d['x'])


def test_ddp_short_batch_weighting():
    # After DDP's rank averaging, each real item has weight 1/N and padding 0.
    for n in [1,2,3,9]:
        for world in [1,2,4]:
            weights=[0.]*n;counts=[]
            for rank in range(world):
                shard=shard_batch(list(range(n)),rank,world);counts.append(len(shard))
                for ix,w in shard:weights[ix]+=w/world
            assert len(set(counts))==1
            assert weights==pytest.approx([1/n]*n)


def test_scheduler_and_gate():
    assert lr_factor(0,100)==pytest.approx(.1)
    assert lr_factor(99,100)==pytest.approx(.1)
    initial=dict(kl=1.,macro=30.,teacher=dict(macro=60.))
    cfg=dict(min_kl_reduction=.05,min_macro_gain=.25)
    assert not eligible(initial,dict(kl=.5,macro=29.),cfg)
    assert eligible(initial,dict(kl=.9,macro=31.),cfg)


def test_synthetic_pilot_full_resume_eval(tmp_path):
    path,cfg=fixture_config(tmp_path)
    # Only provenance hashing needs a model directory; no weights downloaded.
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{"synthetic":true}')
    cfg.update(pilot_limit=1,pilot_epochs=1,epochs=1,batch_size=1,lr=.001,
               checkpoint_every=1,cache_memory_entries=1,require_pilot=False)
    path.write_text(json.dumps(cfg));cfg=load_config(path)
    runtime=(0,1,0,torch.device('cpu'));torch.set_num_threads(2)
    pilot=run_train(cfg,runtime,pilot=True,engine_factory=TinyEngine)
    assert pilot['updates']==1 and pilot['backbone_trainable']==0
    full=run_train(cfg,runtime,engine_factory=TinyEngine)
    assert full==run_train(cfg,runtime,resume=True,engine_factory=TinyEngine)
    before=Path(cfg['output'])/'full/initial.pt';after=Path(cfg['output'])/'full/epoch1.pt'
    a=torch.load(before,weights_only=False)['router'];b=torch.load(after,weights_only=False)['router']
    assert any(not torch.equal(a[k],b[k]) for k in a)
    result=run_evaluate(cfg,runtime,engine_factory=TinyEngine)
    assert result['student']['questions']==1
    # Completed-run idempotence does not bypass fingerprint validation.
    (model/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='Resume signature'):run_train(cfg,runtime,resume=True,engine_factory=TinyEngine)


def test_interrupted_epoch_commit_resumes_exactly(tmp_path,monkeypatch):
    import tessfold.training as training
    _,cfg=fixture_config(tmp_path)
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    cfg.update(epochs=2,batch_size=1,checkpoint_every=1,require_pilot=False)
    runtime=(0,1,0,torch.device('cpu'));torch.set_num_threads(2)
    original=training.save_torch
    def interrupted(path,state):
        original(path,state)
        if Path(path).name=='latest.pt' and state.get('step')==1:
            raise RuntimeError('simulated interruption after committed update')
    monkeypatch.setattr(training,'save_torch',interrupted)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        run_train(cfg,runtime,engine_factory=TinyEngine)
    monkeypatch.setattr(training,'save_torch',original)
    run_train(cfg,runtime,resume=True,engine_factory=TinyEngine)
    reference=dict(cfg,output=str(tmp_path/'uninterrupted'))
    run_train(reference,runtime,engine_factory=TinyEngine)
    resumed=torch.load(Path(cfg['output'])/'full/epoch2.pt',weights_only=False)
    expected=torch.load(Path(reference['output'])/'full/epoch2.pt',weights_only=False)
    for key in expected['router']:
        torch.testing.assert_close(resumed['router'][key],expected['router'][key],rtol=0,atol=0)
