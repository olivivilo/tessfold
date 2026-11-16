"""Versioned initializer dispatch; leaves previous running source hashes intact."""
from scripts import train_meta_compress_strict as runner
from vqa.meta_compress_whitened_initialization import initialize
runner.initialize=initialize
runner.SOURCES=runner.SOURCES+['scripts/train_meta_compress_strict_whitened.py',
    'vqa/meta_compress_whitened_initialization.py','scripts/calibrate_meta_compress_strict.py']

if __name__=='__main__':runner.main()
