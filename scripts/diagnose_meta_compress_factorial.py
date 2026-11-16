"""Complete the clipping x objective check, without changing frozen first suite.

Also checks whether full regularization works with the local-bias initialization.
"""
import argparse
import json
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import scripts.diagnose_meta_compress as base
from scripts.train_meta_compress import atomic_json,sha

VARIANTS=[
    dict(name='kl_sgd_clip001',kind='generator',objective='prediction',optimizer='sgd',lr=.001,clip=.01,ratio=.1),
    dict(name='local_bias_all_adam',kind='local',objective='total',optimizer='adam',lr=.0003,clip=1.,ratio=.1),
]


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--variant',type=int)
    a=p.parse_args();parent=base.OUT;out=parent/'factorial'
    if a.prepare:
        assert not (out/'protocol.json').exists()
        out.mkdir(exist_ok=True)
        protocol=json.loads((parent/'protocol.json').read_text());protocol['variants']=VARIANTS
        protocol['extension_sha256']=sha(__file__)
        (out/'samples.jsonl').write_bytes((parent/'samples.jsonl').read_bytes())
        atomic_json(out/'protocol.json',protocol)
    else:
        assert sha(__file__)==json.loads((out/'protocol.json').read_text())['extension_sha256']
        base.OUT=out;base.VARIANTS=VARIANTS
        base.run(a.variant,50,16)


if __name__=='__main__':main()
