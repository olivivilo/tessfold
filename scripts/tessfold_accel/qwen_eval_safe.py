"""Immutable concurrent feature-cache IO guard for formal/timing evaluation."""
import json,os
from pathlib import Path
import torch
from scripts.compression_lab_v2_io import identical
from scripts.tessfold_accel.run_qwen_safe import visible_load
from scripts.tessfold_accel import qwen_training,qwen_formal


def main():
    original=qwen_training.atomic_save;torch.load=visible_load
    def guarded(path,obj):
        try:return original(path,obj)
        except FileExistsError:
            path=Path(path)
            if path.parent.name!='features' or path.parent.parent.name!='qwen_cache':raise
            winner=visible_load(path,map_location='cpu',weights_only=True)
            if not identical(obj,winner):raise RuntimeError(f'Concurrent formal feature differs: {path}')
            path.with_name(path.name+f'.{os.getpid()}.tmp').unlink(missing_ok=True)
            print(json.dumps(dict(identical_concurrent_feature=str(path),rank=os.environ.get('RANK'))),flush=True)
    qwen_training.atomic_save=guarded;qwen_formal.main()

if __name__=='__main__':main()
