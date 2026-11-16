"""Public entry point. Validation and help work without model imports."""
from __future__ import annotations
import argparse
import json
import os


def main(argv=None):
    os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    parser=argparse.ArgumentParser(description='TessFold frozen-backbone reproduction')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare-cohorts',help='Rebuild exact splits from locally converted public data')
    p.add_argument('--source-root',default='.')
    p.add_argument('--output-root',default='.')
    p=commands.add_parser('download-data',help='Explicitly download pinned public dataset sources')
    p.add_argument('--endpoint',default='https://huggingface.co')
    for name in ('validate','pilot','train','evaluate','download-model'):
        p=commands.add_parser(name)
        p.add_argument('--config',required=True,help='JSON config; relative paths resolve from current directory')
        if name in {'pilot','train','evaluate'}:
            p.add_argument('--device',choices=['auto','cpu','cuda','npu'],default='auto')
        if name=='validate':p.add_argument('--skip-images',action='store_true',help='Schema/ID checks only; cannot verify image content')
        if name in {'pilot','train'}:p.add_argument('--resume',action='store_true')
        if name=='evaluate':p.add_argument('--stage',choices=['pilot','full'],default='full')
    a=parser.parse_args(argv)
    if a.command=='prepare-cohorts':
        from .data import prepare_cohorts
        prepare_cohorts(a.source_root,a.output_root)
        return
    if a.command=='download-data':
        from .data import download_sources
        download_sources(a.endpoint.rstrip('/'))
        return
    from .config import load_config,validate
    cfg=load_config(a.config)
    if a.command=='validate':
        _,summary=validate(cfg,check_images=not a.skip_images)
        print(json.dumps({k:v for k,v in summary.items() if k!='image_content_sha256'},ensure_ascii=False,indent=2))
    elif a.command=='download-model':
        from .download import download
        download(cfg)
    else:
        from .runtime import initialize
        from .training import run_train,run_evaluate
        runtime=initialize(a.device,cfg['threads'])
        try:
            if a.command=='evaluate':run_evaluate(cfg,runtime,a.stage)
            else:run_train(cfg,runtime,pilot=a.command=='pilot',resume=a.resume)
        finally:
            import torch.distributed as dist
            if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
