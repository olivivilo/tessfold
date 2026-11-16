# 环境构建

使用 Python 3.10 与 venv。`pyproject.toml` 固定了核心模型接口版本，特别是 PyTorch 2.8.0、Transformers 4.57.1、tokenizers 0.22.2 和 qwen-vl-utils 0.0.14。C++ 分区器需要 `g++`，首次执行时编译到被忽略的 `artifacts/`；编译参数禁止 fast-math/FMA 重结合。

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
# 在 PyTorch 官方安装页面选择与驱动匹配的 2.8.0 / 0.23.0 CUDA wheel。
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0
.venv/bin/python -m pip install -e '.[test,analysis]'
.venv/bin/python -m tessfold pilot --config configs/local/run.json --device cuda
```

## Ascend

历史硬件路线为 Ascend，Python 3.10/aarch64、CANN 8.5、torch-npu 2.8.0.post5。
先由宿主机安装匹配的驱动、CANN 和 vendor wheels；它们不由 Python 仓库打包。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-npu.txt
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m tessfold pilot --config configs/local/run.json --device npu
```

不同机器的驱动和 wheel 来源不同，不能把原机器的 `.venv-und`、`.pth` 或本地安装路径当作可发布环境。`auto` 仅自动选择 CUDA 或 CPU；Ascend 请显式指定 `npu`。

`requirements-legacy.txt` 是历史数据转换、统计、基线/可选 LoRA 脚本的额外依赖。
官方 ms-swift 微调还需按上游 MonkeyOCRv2 安装；本仓库不附第三方 checkout。

构建 wheel：

```bash
.venv/bin/python -m pip install build
.venv/bin/python -m build --wheel
```

wheel 包含 C++ 源码和来源重建配方。运行时的模型、数据和输出仍需独立准备。
