"""Manual/CI Gloo smoke, invoked with torchrun and a temporary fixture config."""
import argparse
import torch.distributed as dist
from tessfold.config import load_config
from tessfold.runtime import initialize
from tessfold.training import run_train,run_evaluate
from tests.test_training import TinyEngine


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=load_config(a.config);runtime=initialize('cpu',2)
    try:
        run_train(cfg,runtime,engine_factory=TinyEngine)
        run_train(cfg,runtime,resume=True,engine_factory=TinyEngine)
        run_evaluate(cfg,runtime,engine_factory=TinyEngine)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
