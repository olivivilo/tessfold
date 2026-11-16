"""Initialize existing tied Q/K; no query-centering operation in the forward."""
import hashlib
from pathlib import Path
import torch
from vqa.meta_compress_initialization import initialize as original_initialize

def initialize(config):
    module=original_initialize({**config,'initialization':'paper','projection_gain':1.})
    path=Path(__file__).resolve().parents[1]/'artifacts/meta_compress_strict/pool_null_calibration'/f'{config["insertion"]}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==config['calibration_basis_sha256']
    weight=torch.load(path,map_location='cpu',weights_only=True)['weight']
    with torch.no_grad():
        module.query.weight.copy_(weight*config['projection_gain']);module.key.weight.copy_(module.query.weight)
    return module
