"""Numerical repair of sparse-square normalization, preserving frozen v2 source.

Original v2 adds 1e-12 to a potentially tiny denominator. This makes surviving
rows sub-stochastic. Renormalizing the returned nonempty row exactly cancels
that additive term; empty rows already use the original safe fixed-cell fallback.
Only sparse_relu uses this class. The other 23 schemes keep their original code.
"""
from vqa.compression_generators_v2 import SpatialGenerator
class SparseRowStochastic(SpatialGenerator):
    def forward(self,x,grid,keep_ratio=.1,raw=None,question=None):
        _,p=super().forward(x,grid,keep_ratio,raw=raw,question=question)
        p=p/p.sum(-1,keepdim=True).clamp_min(1e-20)
        return p@x.float(),p
