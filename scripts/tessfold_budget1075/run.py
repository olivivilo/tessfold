"""Reuse frozen prior implementations; redirect only new experiment outputs."""
import sys
from scripts.tessfold_budget1075.common import BASE, ROOT, save, sha, now, verify


def main():
    action, *rest = sys.argv[1:]
    verify()
    import os
    save(BASE / 'runtime' / f'{action}.{os.environ.get("SLURM_JOB_ID")}.rank{os.environ.get("RANK", "0")}.json',
         dict(time=now(), action=action, wrapper_sha256=sha(__file__), args=rest,
              note='Only redirect output BASE for eval/memory/audit; train labels are budget1075/...; cache and math unchanged.'))
    sys.argv = [sys.argv[0], *rest]
    if action == 'train':
        from scripts.tessfold_accel.run_qwen_safe import main as execute
    elif action == 'eval':
        from scripts.tessfold_accel import qwen_formal
        qwen_formal.BASE = BASE
        from scripts.tessfold_accel.qwen_eval_safe import main as execute
    elif action == 'memory':
        from scripts.tessfold_accel import qwen_memory
        qwen_memory.BASE = BASE
        execute = qwen_memory.main
    elif action == 'audit':
        from scripts.tessfold_accel import training_audit_npu
        training_audit_npu.BASE = BASE
        execute = training_audit_npu.main
    else:
        raise ValueError(action)
    execute()


if __name__ == '__main__':
    main()
