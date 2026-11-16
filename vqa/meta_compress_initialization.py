"""Parameter initialization only: the MetaCompress forward and loss stay intact."""
import math
import torch
from vqa.meta_compress import MetaCompress


def initialize(config):
    # Construct on CPU, then move, for device-independent seed/initial weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        module = MetaCompress(feature_dim=2048 if config['insertion']=='post-merger' else 768)
        kind = config.get('initialization', 'paper')
        with torch.no_grad():
            if kind == 'paper':
                module.query.weight.mul_(config.get('projection_gain', 1.))
                module.key.weight.copy_(module.query.weight)
            elif kind == 'fourier-position':
                # Fill EXISTING learnable absolute row/column tables. No mask,
                # additive logit bias, temperature, or extra forward operation.
                # A random coordinate embedding puts position in feature space;
                # the tied Q/K initially read it back. All parameters stay free.
                d, c = module.feature_dim, module.compression_dim
                quarter = c//4
                coords = torch.arange(module.max_grid).float()[:, None]
                wavelengths = torch.logspace(math.log10(8), math.log10(256), quarter)
                phase = coords * (2*math.pi/wavelengths[None])
                waves = torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
                module.row_embedding.weight.zero_(); module.col_embedding.weight.zero_()
                module.row_embedding.weight[:, :c//2].copy_(waves)
                module.col_embedding.weight[:, c//2:].copy_(waves)
                # Signed coordinate permutation: orthonormal columns, no QR
                # ambiguity across BLAS implementations. No content fitting.
                axes=torch.randperm(d)[:c]
                basis=torch.zeros(d,c)
                basis[axes,torch.arange(c)] = torch.where(torch.rand(c)>.5,1.,-1.)
                strength=config.get('position_strength',2.)
                gain=config.get('projection_gain',1.)
                module.pos_projection.weight.copy_(basis*strength)
                module.query.weight.copy_(basis.T*gain)
                module.key.weight.copy_(module.query.weight)
            else:
                raise ValueError(kind)
    return module
