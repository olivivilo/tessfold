"""Dense joint positional parameter initialization; unchanged forward/module."""
import torch
from vqa.meta_compress_initialization import initialize as original_initialize

def initialize(config):
    module=original_initialize({**config,'initialization':'fourier-position'})
    # A dense orthonormal embedding lets featurewise max pooling encode joint
    # row/column corner configurations; coordinate axes lose that information.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']+16)
        basis=torch.linalg.qr(torch.randn(module.feature_dim,64),mode='reduced').Q
    with torch.no_grad():
        module.pos_projection.weight.copy_(basis*config['position_strength'])
        module.query.weight.copy_(basis.T*config['projection_gain'])
        module.key.weight.copy_(module.query.weight)
    return module
