"""Decoder-free lossplus V1 and reproducible, balanced random convex matrices.

This module never imports a language model, projector, tokenizer, or answers.
Matrices are represented by disjoint sparse rows plus a strictly positive uniform
floor. This admits an exact diagonal-plus-rank-one reconstruction solve.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def patch_geometry(grid, merge_size=2):
    t, h, w = map(int, grid)
    assert t == 1, 'Only single-page images in this experiment'
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    order = lambda a: a.reshape(h//merge_size, merge_size, w//merge_size, merge_size).permute(0, 2, 1, 3).flatten()
    yy, xx = order(y), order(x)
    return torch.stack(((xx.float()+.5)/w, (yy.float()+.5)/h), -1), yy, xx


def image_edge_weights(pixel_values, grid, image_mean, image_std, patch_size=14):
    """Undo processor normalization/patchification, then Sobel in the actual pixels."""
    u, yy, xx = patch_geometry(grid)
    _, h, w = map(int, grid)
    patches = pixel_values.cpu().float().reshape(-1, 3, patch_size, patch_size)
    patches = patches * torch.tensor(image_std)[None, :, None, None] + torch.tensor(image_mean)[None, :, None, None]
    image = torch.empty(3, h, w, patch_size, patch_size)
    image[:, yy, xx] = patches.permute(1, 0, 2, 3)
    image = image.permute(0, 1, 3, 2, 4).reshape(3, h*patch_size, w*patch_size)
    gray = (image * torch.tensor([.299, .587, .114])[:, None, None]).sum(0)[None, None]
    sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
    edge = F.conv2d(padded, sx).abs() + F.conv2d(padded, sx.transpose(-1, -2)).abs()
    energy = F.avg_pool2d(edge, patch_size, patch_size)[0, 0, yy, xx]
    weights = 1 + 2*(energy-energy.min())/(energy.max()-energy.min()+1e-8)
    return u, weights


def _local_groups(u, m, rng):
    """Randomized spatial bisection with balanced, non-overlapping group sizes."""
    sizes = np.full(m, len(u)//m, dtype=int)
    sizes[:len(u) % m] += 1
    rng.shuffle(sizes)
    output = []

    def split(indices, target):
        if len(target) == 1:
            output.append(indices)
            return
        extent = np.ptp(u[indices], axis=0)
        axis = int(np.argmax(extent * rng.uniform(.8, 1.2, 2)))
        order = np.lexsort((rng.random(len(indices)), u[indices, axis]))
        left = len(target)//2
        cut = int(target[:left].sum())
        split(indices[order[:cut]], target[:left])
        split(indices[order[cut:]], target[left:])

    split(np.arange(len(u)), sizes)
    return output


@dataclass
class BalancedMatrix:
    group: torch.Tensor           # [N], one support row per original token
    weight: torch.Tensor          # [N], support-row softmax weights
    m: int
    floor: float = 1e-6

    @property
    def n(self):
        return len(self.group)

    def aggregate(self, x, weights=None):
        w = self.weight if weights is None else weights
        shape = (self.m,) + tuple(x.shape[1:])
        return torch.zeros(shape, dtype=x.dtype, device=x.device).index_add(0, self.group, x*w.reshape((-1,) + (1,)*(x.ndim-1)))

    def compress(self, x):
        return (1-self.floor)*self.aggregate(x) + self.floor*x.mean(0)

    def dense(self):
        a = torch.zeros(self.m, self.n, dtype=self.weight.dtype)
        a[self.group, torch.arange(self.n)] = self.weight
        return (1-self.floor)*a + self.floor/self.n

    def reconstruct(self, c, ridge):
        # PP^T = diag((1-f)^2 sum_i A_ji^2) + (2f-f^2)/N * 11^T.
        d = (1-self.floor)**2*torch.zeros(self.m, dtype=c.dtype).index_add(0, self.group, self.weight.square()) + ridge
        beta = (2*self.floor-self.floor**2)/self.n
        invd = d.reciprocal()
        z = c*invd[:, None]
        z = z - invd[:, None]*(beta*z.sum(0)/(1+beta*invd.sum()))
        return (1-self.floor)*self.weight[:, None]*z[self.group] + self.floor/self.n*z.sum(0)

    def regularizers(self, u):
        norm2 = (1-self.floor)**2*torch.zeros(self.m).index_add(0, self.group, self.weight.square()) + (2*self.floor-self.floor**2)/self.n
        beta = (2*self.floor-self.floor**2)/self.n
        inv = norm2.reciprocal()
        div = beta**2*(inv.sum().square()-inv.square().sum())/max(1, self.m*(self.m-1))
        centers = self.compress(u)
        loc = (self.compress(u.square().sum(-1))-centers.square().sum(-1)).mean()
        mass = (1-self.floor)*self.weight + self.floor*self.m/self.n
        cov = (0.2*self.m/self.n-mass).clamp_min(0).square().mean()
        return div, loc, cov

    def diagnostics(self):
        sizes = torch.bincount(self.group, minlength=self.m)
        mass = (1-self.floor)*self.weight + self.floor*self.m/self.n
        rowmax = torch.zeros(self.m).scatter_reduce_(0, self.group, self.weight, reduce='amax')*(1-self.floor)+self.floor/self.n
        return {'n': self.n, 'm': self.m, 'retained_fraction': self.m/self.n,
                'support_size_min': int(sizes.min()), 'support_size_max': int(sizes.max()),
                'singleton_rows': int((sizes == 1).sum()), 'row_sum_max_error': float((self.aggregate(torch.ones(self.n))-1).abs().max()),
                'min_entry': self.floor/self.n, 'max_entry': float(rowmax.max()),
                'non_singleton_max_entry': float(rowmax[sizes > 1].max()) if (sizes > 1).any() else None,
                'column_mass_min_relative': float(mass.min()/(self.m/self.n)),
                'column_mass_max_relative': float(mass.max()/(self.m/self.n))}


def random_matrix(u, ratio, family, seed, logit_bound=2., uniform_mix=.3):
    """Random logits, bounded softmax, balanced coverage, no repeated support.

    P=(1-f)[(1-eta) softmax(S_local)+eta/group_size] + f/N.
    All entries are positive and equivalently P=softmax(log(P), dim=-1).
    No feature, question, answer, downstream score, or proxy loss is used.
    """
    rng = np.random.default_rng(seed)
    n = len(u)
    m = max(1, int(n*ratio))
    if family == 'local':
        groups = _local_groups(u.numpy(), m, rng)
    elif family == 'global':
        groups = list(np.array_split(rng.permutation(n), m))
    else:
        raise ValueError(family)
    # Keep row order as close as possible to the original block traversal.
    groups.sort(key=lambda a: float(a.mean()))
    group = np.empty(n, dtype=np.int64)
    weight = np.empty(n, dtype=np.float32)
    for j, idx in enumerate(groups):
        logits = rng.uniform(-logit_bound, logit_bound, len(idx))
        w = np.exp(logits-logits.max())
        w = (1-uniform_mix)*w/w.sum() + uniform_mix/len(idx)
        group[idx], weight[idx] = j, w
    return BalancedMatrix(torch.from_numpy(group), torch.from_numpy(weight), m)


class DenseMatrix:
    """Generic differentiable implementation, also the independent test oracle."""
    def __init__(self, p):
        self.p = p

    def compress(self, x):
        return self.p @ x

    def reconstruct(self, c, ridge):
        return self.p.T @ torch.linalg.solve(self.p @ self.p.T + ridge*torch.eye(len(self.p)), c)

    def regularizers(self, u):
        m, n = self.p.shape
        norm = F.normalize(self.p, dim=-1)
        sims = (norm @ norm.T).square()
        div = (sims.sum()-sims.diag().sum())/max(1, m*(m-1))
        loc = (self.p @ u.square().sum(-1)-(self.p @ u).square().sum(-1)).mean()
        cov = (0.2*m/n-self.p.sum(0)).clamp_min(0).square().mean()
        return div, loc, cov


class VisualProxy:
    """lossplus.md section 12 V1; fixed probes, teacher detached from gradients."""
    def __init__(self, x, u, edge_weights, probe_seed, projection_seed=20261016,
                 attention_dim=128, tau_op=1., tau_rel=.1, alpha=.25,
                 ridge=1e-4, delta=1e-3):
        self.x, self.u, self.w = x.detach().float(), u, edge_weights
        self.ridge, self.delta, self.tau_op, self.tau_rel = ridge, delta, tau_op, tau_rel
        self.da = min(attention_dim, x.shape[-1])
        gen = torch.Generator().manual_seed(projection_seed)
        self.wk = torch.linalg.qr(torch.randn(x.shape[-1], self.da, generator=gen), mode='reduced').Q
        self.wv = torch.linalg.qr(torch.randn(x.shape[-1], self.da, generator=gen), mode='reduced').Q
        gen = torch.Generator().manual_seed(probe_seed)
        self.anchors = torch.randperm(len(x), generator=gen)[:min(256, len(x))]
        self.candidates = torch.randperm(len(x), generator=gen)[:min(512, len(x))]
        self.z = F.layer_norm(self.x, (x.shape[-1],))
        angles = u[:, :, None] * (2.**torch.arange(4)*math.pi)
        self.phi = alpha*torch.cat((angles.sin(), angles.cos()), -1).flatten(1)
        rt = self.relation(self.z)
        self.rt = rt.softmax(-1).detach()
        self.logrt = rt.log_softmax(-1).detach()
        self.kx, self.vx = self.z @ self.wk, self.z @ self.wv
        random_q = F.normalize(torch.randn(64, self.da, generator=gen), dim=-1)
        idx = torch.randperm(len(x), generator=gen)[:min(64, len(x))]
        self.q = torch.cat((random_q, F.normalize(self.kx[idx], dim=-1)))
        attn = self.attention(self.kx)
        self.yt = (attn @ self.vx).detach()
        self.teacher_attention_entropy_fraction = float(-(attn*attn.clamp_min(1e-30).log()).sum(-1).mean()/math.log(len(x)))
        self.teacher_attention_max = float(attn.max(-1).values.mean())

    def relation(self, z):
        r = torch.cat((F.normalize(z, dim=-1), self.phi), -1)
        return r[self.anchors] @ r[self.candidates].T / self.tau_rel

    def attention(self, k):
        return (self.q @ k.T/(math.sqrt(self.da)*self.tau_op)).softmax(-1)

    def __call__(self, matrix):
        c = matrix.compress(self.x)
        xhat = matrix.reconstruct(c, self.ridge)
        zhat = F.layer_norm(xhat, (xhat.shape[-1],))
        rec = (((self.z-zhat).square()+self.delta**2).sqrt().mean(-1)*self.w).sum()/self.w.sum()
        # Stable KL is equivalent to the specified distribution KL, without eps bias.
        rel = (self.rt*(self.logrt-self.relation(zhat).log_softmax(-1))).sum(-1).mean()
        zc = F.layer_norm(c, (c.shape[-1],))
        ys = self.attention(zc @ self.wk) @ (zc @ self.wv)
        cos = (1-(self.yt*ys).sum(-1)/(self.yt.norm(dim=-1)*ys.norm(dim=-1)+1e-8)).mean()
        l2 = ((self.yt-ys).square().sum(-1)/(self.yt.square().sum(-1)+1e-8)).mean()
        op = cos+.5*l2
        div, loc, cov = matrix.regularizers(self.u)
        reg = .05*div+.01*loc+.05*cov
        total = .5*rec+rel+2*op+.1*reg
        return c, {'total': total, 'rec': rec, 'rel': rel, 'op': op, 'op_cos': cos, 'op_l2': l2,
                   'reg': reg, 'div': div, 'loc': loc, 'cov': cov,
                   'A_rec': rec, 'B_rec_rel': rec+rel, 'C_rec_op': rec+op,
                   'D_rec_rel_op': rec+rel+op,
                   'unsup_section8': rec+rel+2*op+.1*reg}
