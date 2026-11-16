"""Numerical checks of formulas, optimized solves, and useful gradients."""
import json
import torch
from vqa.token_proxy import BalancedMatrix, DenseMatrix, VisualProxy, random_matrix, patch_geometry


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    u, yy, xx = patch_geometry((1, 8, 10))
    x = torch.randn(80, 32)
    proxy = VisualProxy(x, u, torch.ones(80), 7, attention_dim=16)
    checks = []
    for family in ('local', 'global'):
        p = random_matrix(u, .375, family, 99)
        dense = p.dense()
        torch.testing.assert_close(dense, dense.log().softmax(-1), atol=1e-7, rtol=1e-6)
        assert dense.min() > 0
        torch.testing.assert_close(dense.sum(-1), torch.ones(p.m))
        assert p.diagnostics()['column_mass_min_relative'] >= .15
        c, loss = proxy(p)
        c2, loss2 = proxy(DenseMatrix(dense))
        torch.testing.assert_close(c, c2, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(p.reconstruct(c, 1e-4), DenseMatrix(dense).reconstruct(c2, 1e-4), atol=3e-6, rtol=1e-4)
        for k in loss:
            torch.testing.assert_close(loss[k], loss2[k], atol=5e-6, rtol=3e-4)
        logits = dense.log().requires_grad_()
        _, gradloss = proxy(DenseMatrix(logits.softmax(-1)))
        gradloss['total'].backward()
        assert torch.isfinite(logits.grad).all() and logits.grad.norm() > 0
        checks.append({'family': family, 'dense_formula_agreement': True,
                       'loss': float(loss['total']), 'gradient_norm': float(logits.grad.norm())})
    _, identity = proxy(DenseMatrix(torch.eye(len(x))))
    assert identity['rec'] < .00101 and abs(identity['rel']) < 1e-5 and abs(identity['op']) < 1e-5
    assert sorted(zip(yy.tolist(), xx.tolist())) == [(y, x) for y in range(8) for x in range(10)]
    print(json.dumps({'checks': checks, 'identity': {k: float(v) for k,v in identity.items()}}, indent=2))


if __name__ == '__main__':
    main()
