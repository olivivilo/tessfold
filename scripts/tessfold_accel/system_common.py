"""Independent new exact compiled system measurement, immutable signatures."""
from pathlib import Path
from scripts.tessfold_repair.common import ROOT,load,lines,save,sha,now,load_module
BASE=ROOT/'artifacts/tessfold_acceleration_20261115/system_exact'
REG=BASE/'registration.json'
SOURCES=['vqa/tessfold_fast_runtime.py','vqa/tessfold_native_partition.py','vqa/tessfold_partition_native.cpp',
    'scripts/tessfold_accel/system_common.py','scripts/tessfold_accel/system_benchmark.py',
    'scripts/tessfold_accel/system.sh','scripts/tessfold_accel/system_report.py']

def prepare():
    from scripts.tessfold_repair.common import verify as parent_verify,REG as PARENT
    import shutil
    parent_verify();BASE.mkdir(exist_ok=True)
    if REG.exists():return verify()
    for name in ['timing_rows.jsonl','multi_pages.json']:
        shutil.copy2(PARENT.parent/name,BASE/name)
    reg=dict(time=now(),source_sha256={s:sha(ROOT/s) for s in SOURCES},parent_registration_sha256=sha(PARENT),
        data_sha256={s:sha(BASE/s) for s in ['timing_rows.jsonl','multi_pages.json']},
        strict_gate='448real cases exact before promotion; compiled FP32 NumPy reference sums and identical dense FP32 readout',
        no_training=True,main='25percent,50percent secondary;90fixedQA,15configs,3repeats naturalEOS and fixed64; separate1350memory requests;9450records',
        graph='Measured independently; capture/setup and persistent plan memory excluded from main eager-cold implementation.',
        hard='True linear representative gather numerical variant, not same as soft, report separately')
    save(REG,reg)
    for s in SOURCES:
        dest=BASE/'sources'/s;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/s,dest)
    return reg

def verify():
    from scripts.tessfold_repair.common import verify as parent_verify,REG as PARENT
    parent_verify();r=load(REG);assert r
    assert sha(PARENT)==r['parent_registration_sha256']
    for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in r['data_sha256'].items():assert sha(BASE/p)==h,p
    return r

if __name__=='__main__':
    prepare();print(str(REG))
