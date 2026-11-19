# Installation

Use Python 3.10 and `venv`. `pyproject.toml` pins the core model-interface versions, including PyTorch 2.8.0, Transformers 4.57.1, tokenizers 0.22.2, and qwen-vl-utils 0.0.14. The C++ partitioner requires `g++`. On first use, it compiles into the Git-ignored `artifacts/` directory with fast-math and floating-point contraction disabled.

## CPU

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test,analysis]'
.venv/bin/python -m pytest
```

## CUDA

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
# Select PyTorch 2.8.0 / torchvision 0.23.0 CUDA wheels compatible with your driver.
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0
.venv/bin/python -m pip install -e '.[test,analysis]'
.venv/bin/python -m tessfold pilot --config configs/local/run.json --device cuda
```

## Ascend

The historical Ascend environment used Python 3.10 on aarch64, CANN 8.5, and torch-npu 2.8.0.post5. Install compatible host drivers, CANN, and vendor wheels first; these system components are not distributed in this Python repository.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-npu.txt
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m tessfold pilot --config configs/local/run.json --device npu
```

Drivers and wheel sources vary between hosts. The original machine's `.venv-und`, `.pth` files, and installation paths do not constitute a portable environment. `auto` selects CUDA or CPU only; specify `npu` explicitly for Ascend.

`requirements-legacy.txt` lists additional dependencies for historical data conversion, statistics, baselines, and optional LoRA scripts. Official ms-swift fine-tuning requires a separate installation following the upstream MonkeyOCRv2 instructions. Third-party checkouts are excluded.

## Building a wheel

```bash
.venv/bin/python -m pip install build
.venv/bin/python -m build --wheel
```

The wheel includes the C++ sources and source-reconstruction recipes. Models, data, and runtime outputs must be prepared separately.
