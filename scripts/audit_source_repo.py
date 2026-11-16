"""Check the Git upload surface, local imports, and excluded payload categories."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pt','.pth','.ckpt','.safetensors','.onnx','.bin','.zip','.parquet','.arrow',
           '.jsonl','.csv','.npz','.npy','.xlsx','.log','.pyc','.so','.pdf','.png','.jpg'}
GENERATED={'artifacts','models','data','datasets','weights','outputs','results','cache','runs',
           'logs','.venv','.venv-und','.venv-npu','vendor','build','dist'}


def main():
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
    paths=sorted(set(p for p in names if p));errors=[];total=0
    for name in paths:
        path=ROOT/name;parts=Path(name).parts
        if parts[0] in GENERATED or path.suffix in FORBIDDEN:errors.append(f'Payload in upload surface: {name}')
        if path.is_symlink():errors.append(f'Symlink in source repo: {name}')
        if not path.is_file():continue
        total+=path.stat().st_size
        if path.stat().st_size>5*1024**2:errors.append(f'Unexpected >5 MiB file: {name}')
        if path.suffix=='.py':
            tree=ast.parse(path.read_text(),filename=name)
            for node in ast.walk(tree):
                modules=[]
                if isinstance(node,ast.Import):modules=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom) and not node.level and node.module:
                    modules=[node.module]
                    if node.module in {'scripts','vqa','tessfold'}:
                        modules += [node.module+'.'+a.name for a in node.names if a.name!='*']
                for module in modules:
                    if module.split('.')[0] not in {'scripts','vqa','tessfold'}:continue
                    local=ROOT.joinpath(*module.split('.'))
                    if not local.with_suffix('.py').exists() and not local.is_dir():errors.append(f'Missing {module} from {name}')
        if path.suffix in {'.py','.sh','.md','.json','.toml','.yml','.example'}:
            text=path.read_text()
            if re.search(r'(?:hf_[a-zA-Z0-9]{25,}|sk-[a-zA-Z0-9]{25,}|AKIA[0-9A-Z]{16})',text):errors.append(f'Possible embedded credential: {name}')
    result=dict(source_files=len(paths),bytes=total,errors=errors)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
