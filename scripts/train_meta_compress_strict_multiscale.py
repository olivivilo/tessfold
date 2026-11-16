from scripts import train_meta_compress_strict as runner
from vqa.meta_compress_multiscale_position_initialization import initialize
runner.initialize=initialize
runner.SOURCES=runner.SOURCES+['scripts/train_meta_compress_strict_multiscale.py',
    'vqa/meta_compress_multiscale_position_initialization.py','vqa/meta_compress_dense_position_initialization.py']
if __name__=='__main__':runner.main()
