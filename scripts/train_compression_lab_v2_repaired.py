"""Dispatch the explicit sparse numeric repair without changing signed legacy runs."""
from scripts import train_compression_lab_v2 as lab
from vqa.compression_generators_v2_repair import SparseRowStochastic
ORIGINAL_BUILD=lab.build
ORIGINAL_SOURCES=list(lab.SOURCES)
REPAIR_SOURCES=['vqa/compression_generators_v2_repair.py','scripts/train_compression_lab_v2_repaired.py']
def build(cfg):
    if cfg['kind']=='sparse_relu':return SparseRowStochastic(cfg['kind'])
    return ORIGINAL_BUILD(cfg)
lab.build=build
ORIGINAL_TRAIN=lab.train
def train(engine,data,cfg):
    lab.SOURCES=ORIGINAL_SOURCES+REPAIR_SOURCES if cfg['kind']=='sparse_relu' else ORIGINAL_SOURCES
    return ORIGINAL_TRAIN(engine,data,cfg)
lab.train=train
if __name__=='__main__':lab.main()
