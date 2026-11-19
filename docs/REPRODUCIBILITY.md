# Source Fidelity and Reproducibility Scope

## Retained method implementations

The implementations in `vqa/` were copied from the current working version. Native `NativeQwen` additionally accepts `model_path` and `TESSFOLD_NATIVE_MODEL`; the official UND wrapper accepts `TESSFOLD_UND_MODEL` as its default model-path override. These changes do not alter the network architecture, partitions, softmax, P@X, four-branch compression, or MRoPE formulas.

Pre-adjustment source checksums are recorded in `EXTRACTED_SOURCE_MANIFEST.json`. This is source provenance metadata, not experimental output. A small number of historical entry points use relative paths or `DATASET_ROOT` in place of personal absolute paths; their method computations are unchanged.

`scripts/` retains original training, evaluation, diagnostic, and execution programs together with their local dependencies. Archive-packaging and migration scripts, and illustration-generation scripts unrelated to training, are excluded. Original data and model implementations are obtained from public sources.

## Portable interface

`tessfold/` is a separate execution layer. It loads models and data from explicit configuration, without requiring historical checkpoints, predictions, `selected_runs.json`, or completion markers to exist beforehand.

Its training protocol includes:

- Full-vocabulary KL under a teacher-generated greedy response prefix, without gold cross-entropy or row entropy.
- A frozen backbone and trainable generator, AdamW, warmup/cosine scheduling, and pre-clipping gradient-norm logging.
- An equally weighted macro average of native task metrics; DeepForm/KLC use DUE corpus F1.
- Fresh full-training initialization after a successful pilot, fixed development-set eligibility criteria, and no test-based selection.
- Per-epoch shuffling with a fixed seed. Distributed short batches use zero-weight padding so real examples have equal weight.
- Checkpoints containing optimizer state and global update count. Data order is reconstructed from seed and epoch; learning rate is reconstructed from the update count. Current routers have no dropout and use deterministic partition computations.
- Strict resume checks on code, configuration, actual model bytes, manifests, image hashes, and world size.
- Cache namespaces derived from model and implementation hashes; teacher keys derived from the question and image content.

By default, `cache_logits=false` avoids storing full-vocabulary probabilities and recomputes teacher forward passes. Setting it to true trades disk space for compute. Feature caching can also be disabled. These explicit execution choices do not change the KL definition.

## Differences from historical scripts

The portable runner is not a drop-in replacement for every historical protocol:

1. Earlier full UND experiments used fixed-epoch selection and field-level exact-match statistics. The portable runner uses development-set eligibility and native DUE F1; its numbers must not be presented as results of the earlier protocol.
2. Historical native 8B training could split one allocation into HCCL subgroups for multiple configurations. The portable runner trains one configuration at a time, on one device or with DDP, and explicitly corrects short-batch sample weighting.
3. Original native scripts could reuse existing short-answer prediction caches. The portable runner generates teachers afresh and does not depend on old results.
4. The formal forward path records diagnostic durations after a shared visual frontend. **These are not fair full-deployment speed measurements.** To reproduce TTFT, end-to-end latency, independent memory measurements, fixed-64-token generation, or real batching, use the original timing/memory scripts and regenerate their registrations and fixed request sets. Do not use the portable diagnostics to support paper-level speedup claims.
5. Historical registrations, caches, and weights referenced by controllers are not distributed with the source. Their absence must not be interpreted as successful reconstruction.

## Validation performed during source preparation

Regression tests cover reference partition agreement, all four retention fractions and the identity case, convexity, gradients, distributed short-batch weighting, image-content isolation, development-set eligibility, and a pilot/full/resume/evaluation cycle using a synthetic frozen decoder. Synthetic outputs are written to temporary test directories and are not VQA results.

A two-process Gloo training/resume/evaluation check was also performed. A recovery test deliberately interrupts training after committing an epoch update and verifies every recovered parameter against uninterrupted training.

Local validation used a separate venv with read-only access to an existing CPU tensor environment, and an editable installation of this project. The CI configuration specifies a fresh CPU installation from public dependencies. GitHub CI was not run during this preparation, and real full backbones were not downloaded or retrained.

`pip check` identified a missing `av` video dependency declared by qwen-vl-utils. Installing it timed out on PyPI, and the fallback mirror was also unreachable. Image-routing tests and the 26 checks above do not require video decoding and passed; this is not validation of a complete installation from scratch. A normal complete installation resolves `av` through qwen-vl-utils. The local test venv and build outputs were removed after preparation and are not distributed.

Model loading uses upstream implementations, including `trust_remote_code=True` for UND; obtain these from the pinned sources. Unavailable models or datasets, different GPU/NPU kernels, and floating-point reduction order can affect reproducibility. A successful source build and passing unit tests do not establish identical paper metrics.
