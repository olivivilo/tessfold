import numpy as np
import pytest
import torch
from vqa.compression_generators_v2 import adaptive_rectangles,SpatialGenerator
from vqa.tessfold_native_partition import partition
from vqa.tessfold_qwen3vl import NativeSpatialRouter
from vqa.tessfold_qwen3vl_stable import StableColumn


@pytest.mark.parametrize('kind',['Tree','Column','StableColumn'])
@pytest.mark.parametrize('ratio',[.1,.25,.5,.75,1.])
def test_native_convex_routes_and_gradients(kind,ratio):
    torch.set_num_threads(2);torch.manual_seed(11)
    module=StableColumn('Column') if kind=='StableColumn' else NativeSpatialRouter(kind)
    x=torch.randn(24,4096);q=torch.randn(3,4096)
    p,reps=module(x,(4,6),q,ratio)
    assert p.shape==(max(1,round(24*ratio)),24)
    torch.testing.assert_close(p.sum(-1),torch.ones(len(p)))
    assert p.min()>=0 and torch.all(reps[1:]>reps[:-1])
    if ratio<1:
        (p@x).square().mean().backward()
        assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in module.parameters())
    else:
        torch.testing.assert_close(p@x,x,rtol=0,atol=0)


@pytest.mark.parametrize('kind',['quadtree_question','col_adaptive'])
def test_und_pre_post_convexity(kind):
    torch.manual_seed(9);module=SpatialGenerator(kind)
    raw=torch.randn(24,768);x=torch.randn(24,2048);q=torch.randn(3,2048);grid=torch.tensor([[1,4,6]])
    for xx in (torch.zeros_like(x),x):
        c,p=module(xx,grid,.25,raw=raw,question=q)
        assert p.shape==(6,24)
        torch.testing.assert_close(p.sum(-1),torch.ones(6))
        torch.testing.assert_close(c,p@xx)
        (p@raw).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


@pytest.mark.parametrize('mode',['tree','col'])
def test_compiled_partition_matches_independent_reference(mode):
    rng=np.random.default_rng(41)
    for h,w in [(1,9),(8,1),(4,6),(12,10)]:
        for ratio in [.1,.25,.5,.75,1.]:
            for image in [np.ones((h,w),np.float32),rng.uniform(.01,5,(h,w)).astype(np.float32)]:
                m=max(1,round(h*w*ratio))
                expected=adaptive_rectangles(torch.from_numpy(image),h,w,m,mode)
                np.testing.assert_array_equal(expected,partition(image,m,mode))
