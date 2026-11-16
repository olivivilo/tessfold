"""MetaCompress-style image-only compression for MonkeyOCRv2-Und.

The implementation follows Wang et al. (CVPR 2026): a position-aware query
pool, key projection, diagonal channel weights, and row-wise softmax produce
P in R^{m x n}; compressed visual tokens are P @ X.  The generator is kept
separate from the frozen MonkeyOCRv2 vision tower, merger, and Qwen decoder so
that it can be checkpointed and evaluated independently.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MetaCompress(nn.Module):
    """Dynamic-resolution MetaCompress generator.

    Inputs are MonkeyOCRv2 visual features ``x`` with shape ``[n, d]``:
    d=2048 after official merger (paper placement), or 768 before merger.
    The corresponding ``[t, h, w]`` grid retains official 2x2 block ordering.
    Queries use
    the supplemental's fractional max pooling with kernel size three.
    Absolute 2-D position is represented by
    learned row/column tables and projected into the feature dimension.
    """

    def __init__(self, feature_dim: int = 2048, compression_dim: int = 64,
                 max_grid: int = 512, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.compression_dim = compression_dim
        self.max_grid = max_grid
        self.spatial_merge_size = spatial_merge_size
        # The paper calls this a position embedding layer.  Separable tables
        # keep the number of parameters independent of the image resolution.
        pos_dim = min(64, compression_dim)
        self.row_embedding = nn.Embedding(max_grid, pos_dim)
        self.col_embedding = nn.Embedding(max_grid, pos_dim)
        self.pos_projection = nn.Linear(pos_dim, feature_dim, bias=False)
        self.query = nn.Linear(feature_dim, compression_dim, bias=False)
        self.key = nn.Linear(feature_dim, compression_dim, bias=False)
        self.channel_weight = nn.Parameter(torch.ones(compression_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Paper initialization: Wq=Wk ~ N(0, 1/d_c^2), theta=1.  Using the
        # same tensor values avoids a random asymmetric starting projection.
        std = 1.0 / float(self.compression_dim)
        nn.init.normal_(self.query.weight, std=std)
        with torch.no_grad():
            self.key.weight.copy_(self.query.weight)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_projection.weight, std=0.02)
        with torch.no_grad():
            self.channel_weight.fill_(1.0)

    @staticmethod
    def _grid_shape(grid: torch.Tensor, n: int) -> Tuple[int, int]:
        g = grid.reshape(-1, 3)[0].detach().cpu().tolist()
        t, h, w = (int(v) for v in g)
        if t != 1 or h * w != n:
            raise ValueError(f"Expected one image with grid product n, got {g}, n={n}")
        return h, w

    def _position(self, n: int, grid: torch.Tensor, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
        h, w = self._grid_shape(grid, n)
        if max(h, w) > self.max_grid:
            raise ValueError(f"Grid {h}x{w} exceeds position table {self.max_grid}")
        s=self.spatial_merge_size
        if h%s or w%s:raise ValueError(f"Grid {h}x{w} must be divisible by official block size {s}")
        # Match tower.get_pos_ids_by_grid exactly: the processor serializes
        # 2x2 blocks, even though the official merger does NOT reduce length.
        rows=torch.arange(h,device=device)[:,None].expand(h,w)
        cols=torch.arange(w,device=device)[None,:].expand(h,w)
        rows=rows.reshape(h//s,s,w//s,s).permute(0,2,1,3).reshape(-1)
        cols=cols.reshape(h//s,s,w//s,s).permute(0,2,1,3).reshape(-1)
        pos=self.row_embedding(rows)+self.col_embedding(cols)
        return self.pos_projection(pos).to(dtype=dtype)

    @staticmethod
    def pool(z: torch.Tensor, m: int) -> torch.Tensor:
        # PyTorch fractional_max_pool2d is also a 1-D pool with height=1.
        # Fix its offset, keeping P deterministic for a given image/model.
        # The paper main text says average pooling, but supplemental 8.2
        # explicitly says fractional max pooling, kernel=3; follow 8.2.
        kernel = min(3, z.shape[0] - m + 1)
        samples = z.new_full((1, z.shape[1], 2), 0.5)
        return F.fractional_max_pool2d(
            z.T[None, :, None, :], kernel_size=(1, kernel),
            output_size=(1, m), _random_samples=samples)[0, :, 0, :].T

    @staticmethod
    def output_length(n: int, keep_ratio: float) -> int:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
        return max(1, min(n, int(round(n * keep_ratio))))

    def forward(self, x: torch.Tensor, grid: torch.Tensor,
                keep_ratio: float = 0.1):
        if x.ndim != 2:
            raise ValueError(f"x must be [n,d], got {tuple(x.shape)}")
        n, d = x.shape
        if d != self.feature_dim:
            raise ValueError(f"feature dim mismatch: {d} != {self.feature_dim}")
        m = self.output_length(n, keep_ratio)
        pos = self._position(n, grid, x.device, x.dtype)
        z = x + pos
        q_input = self.pool(z, m)
        q = self.query(q_input)
        k = self.key(z)
        logits = ((q * self.channel_weight.to(dtype=q.dtype)) @ k.T) / math.sqrt(self.compression_dim)
        p = logits.softmax(dim=-1)
        compressed = p @ x
        return compressed, p


def meta_losses(p: torch.Tensor, teacher_logits: torch.Tensor,
                student_logits: torch.Tensor):
    """Return the paper's prediction, entropy, collapse, and total losses."""
    teacher_logp = teacher_logits.detach().float().log_softmax(-1)
    student_logp = student_logits.float().log_softmax(-1)
    pred = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
    entropy = -(p.float().clamp_min(1e-12).log() * p.float()).sum(-1).mean()
    # Exact paper form: maximum mass accumulated by any input token.
    collapse = p.float().sum(0).max()
    total = pred + entropy + collapse
    return {"prediction": pred, "entropy": entropy,
            "collapse": collapse, "total": total}
