"""Include pixel-parity scale in existing learned absolute position tables."""
import math
import torch
from vqa.meta_compress_dense_position_initialization import initialize as dense_initialize

def initialize(config):
    module=dense_initialize(config)
    quarter=module.compression_dim//4
    coords=torch.arange(module.max_grid).float()[:,None]
    waves=torch.logspace(math.log10(config['wavelength_min']),math.log10(config['wavelength_max']),quarter)
    phase=coords*2*math.pi/waves[None]
    encoding=torch.stack((phase.sin(),phase.cos()),-1).flatten(1)
    with torch.no_grad():
        module.row_embedding.weight.zero_();module.col_embedding.weight.zero_()
        module.row_embedding.weight[:,:32].copy_(encoding)
        module.col_embedding.weight[:,32:].copy_(encoding)
    return module
