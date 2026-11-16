"""Same strict trainer with a generic initializer/placement and eight ranks.

Global batch stays eight. More ranks replace accumulation, without changing
the generator, per-example loss, optimizer, or cohort.
"""
import json
import sys
from scripts import train_meta_compress_strict as runner
from scripts.benchmark_meta_compress_strict import build

config_path = sys.argv[sys.argv.index('--config')+1]
config = json.loads((runner.ROOT/config_path).read_text())
if config['insertion'] in ['post-norm', 'post-gelu']:
    from scripts.train_meta_compress_strict_inner import student
    runner.student = student
runner.initialize = build
runner.SOURCES = list(dict.fromkeys(runner.SOURCES+[
    'scripts/train_meta_compress_strict_full.py',
    'scripts/benchmark_meta_compress_strict.py',
    'scripts/train_meta_compress_strict_inner.py',
    'vqa/meta_compress_dense_position_initialization.py',
    'vqa/meta_compress_multiscale_position_initialization.py',
    'vqa/meta_compress_whitened_initialization.py',
    'vqa/meta_compress_pool_null_initialization.py',
]))

if __name__ == '__main__':
    runner.main()
