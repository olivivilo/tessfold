#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d vendor/MonkeyOCRv2/.git ]]; then
  git clone --filter=blob:none https://github.com/Yuliang-Liu/MonkeyOCRv2.git vendor/MonkeyOCRv2
  git -C vendor/MonkeyOCRv2 checkout 6cc0c0bb4864f2541fa6bc3a8f976cffaac8f359
fi
.venv/bin/python - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();repo=root/'vendor/MonkeyOCRv2'
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert commit=='6cc0c0bb4864f2541fa6bc3a8f976cffaac8f359', 'Review an upstream version change before applying the recorded fix'
swift=repo/'parsing/train/ms-swift'
readme=swift/'README.md'
if not readme.exists():readme.write_text('# Author-vendored ms-swift\nPackaging README omitted from the pinned upstream tree.\n')
registry=swift/'swift/llm/dataset/data/dataset_info.json'
registry.parent.mkdir(parents=True,exist_ok=True)
if not registry.exists():registry.write_text('[]\n')
template=swift/'swift/llm/template/template/monkeyocrv2.py'
text=template.read_text();start=text.index('class MonkeyOCRv2UndTemplate(')
prefix,und=text[:start],text[start:]
if 'encoded = Template._encode(self, inputs)' not in und:
 assert und.count('encoded = super()._encode(inputs)')==1
 und=und.replace('encoded = super()._encode(inputs)','encoded = Template._encode(self, inputs)',1)
 template.write_text(prefix+und)
out=root/'artifacts/official_und';out.mkdir(parents=True,exist_ok=True)
(out/'upstream.json').write_text(json.dumps({'repository':'https://github.com/Yuliang-Liu/MonkeyOCRv2','commit':commit,
 'adaptations':['Packaging README and empty optional dataset registry missing upstream',
                'Und template expands vision placeholders once, skipping Parsing expansion',
                'Runtime choice of author eager vision attention on Ascend; Qwen SDPA'],
 'local_template_sha256':hashlib.sha256(template.read_bytes()).hexdigest()},indent=2))
PY
if [[ ! -x .venv-und/bin/python ]]; then .venv/bin/python -m venv .venv-und; fi
.venv-und/bin/python - <<'PY'
import sysconfig
from pathlib import Path
root=Path.cwd()
Path(sysconfig.get_paths()['purelib'],'project_runtime.pth').write_text(str(root/'.venv-npu/lib/python3.10/site-packages')+'\n'+str(root/'.venv/lib/python3.10/site-packages')+'\n')
PY
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
.venv-und/bin/python -m pip install -r requirements-official-und.txt
.venv-und/bin/python -m pip freeze > artifacts/requirements-official-und.lock.txt
.venv/bin/python -m scripts.download_official_und
