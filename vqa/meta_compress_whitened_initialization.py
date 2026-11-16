"""Another initializer for the unchanged MetaCompress, using training covariance."""
from pathlib import Path
import hashlib
import torch
from vqa.meta_compress_initialization import initialize as original_initialize

def initialize(config):
    module=original_initialize({**config,'initialization':'paper','projection_gain':1.})
    root=Path(__file__).resolve().parents[1]
    path=root/'artifacts/meta_compress_strict/calibration'/f'{config["insertion"]}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==config['calibration_basis_sha256']
    item=torch.load(path,
                    map_location='cpu',weights_only=True)
    with torch.no_grad():
        module.query.weight.copy_(item['weight']*config['projection_gain'])
        module.key.weight.copy_(module.query.weight)
    return module
