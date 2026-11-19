# Entry Points

## Recommended interfaces: no historical outputs required

| Command | Purpose |
|---|---|
| `python -m tessfold --help` | List available interfaces |
| `python -m tessfold download-model --config CONFIG` | Download the selected backbone from a pinned source |
| `python -m tessfold download-data` | Explicitly download public datasets; never invoked during installation |
| `python -m tessfold prepare-cohorts` | Reconstruct three splits by indexing pinned public-source conversions |
| `python -m tessfold validate --config CONFIG` | Validate schema, image contents, and split disjointness |
| `python -m tessfold pilot --config CONFIG --device DEVICE` | Run a small multi-image pilot and development-set checks |
| `python -m tessfold train --config CONFIG --device DEVICE` | After a successful pilot, train from a fresh initialization with the same seed |
| `python -m tessfold evaluate --config CONFIG --device DEVICE` | Load the locked checkpoint and generate independent teacher/student answers |
| `python -m pytest` | Run regression tests without downloading backbone weights |
| `python scripts/audit_source_repo.py` | Check the Git upload surface for results, weights, and other generated files |

`DEVICE` is `cpu`, `cuda`, or `npu`. `auto` selects CUDA or CPU. `pilot` and `train` support `--resume`. `evaluate --stage pilot` evaluates the pilot-selected model; formal paper experiments should use a full-training checkpoint locked in advance.

## Retained original experiment interfaces

These sources are retained to explain and reconstruct historical experiments. **They are not ready-to-run controllers for a fresh machine.** Output paths, Slurm partitions, CANN paths, and registered experiment directories reflect the historical protocols. Registration functions create runtime files; their presence in the source does not imply that results or weights are included.

| Subsystem | Original interface |
|---|---|
| Native 8B training | `scripts/tessfold_accel/qwen_training.py` |
| Native StableColumn training | `scripts/tessfold_accel/qwen_training_stable.py` |
| Shared-cache concurrency repair | `scripts/tessfold_accel/run_qwen_safe.py` |
| Native 8B formal evaluation and timing | `scripts/tessfold_accel/qwen_formal.py`, `qwen_eval_safe.py` |
| Additional 8B 10%/75% experiments | `scripts/tessfold_budget1075/{common,run,controller}.py` |
| UND spatial compression training | `scripts/train_compression_lab_v2{,_distributed,_repaired}.py` |
| UND formal benchmark | `scripts/benchmark_compression_lab_v2.py` |
| Strict system inference and interface repair | `scripts/tessfold_repair/` |
| Equivalence, timing, memory, same-page multi-question, and real-batch tests | Corresponding modules in `scripts/tessfold_accel/` |
| MetaCompress loss, generator, and baselines | `vqa/meta_compress.py`, `scripts/train_meta_compress*.py` |
| First- and second-round generator searches | `scripts/train_compression_lab*.py` and `vqa/compression_generators*.py` |
| Earlier official UND evaluation and LoRA | `scripts/run_official_und.sh` and related scripts; install ms-swift separately |

Fused-vision, segmented, hard, packed, and prefix-cache candidates are retained as separate inference variants, not implicit defaults of the portable interface. Each requires its own numerical and output-equivalence validation. The existence of an implementation does not establish bit-exactness or a speed advantage over the strict path.

Report-generation scripts are programs that produce reports; their previously generated reports and scores are not included. Historical controllers may submit Slurm jobs automatically. Review their actions and cluster settings before running them.
