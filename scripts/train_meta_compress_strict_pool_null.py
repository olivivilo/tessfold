from scripts import train_meta_compress_strict as runner
from vqa.meta_compress_pool_null_initialization import initialize
runner.initialize=initialize
runner.SOURCES=runner.SOURCES+['scripts/train_meta_compress_strict_pool_null.py',
    'vqa/meta_compress_pool_null_initialization.py','scripts/calibrate_meta_compress_strict_pool_null.py']
if __name__=='__main__':runner.main()
