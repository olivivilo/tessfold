import json
from pathlib import Path
import pytest
from tessfold.config import load_config,validate


def fixture_config(tmp_path):
    cfg=dict(backend='native8b',model=str(tmp_path/'model'),kind='Tree',
             insertion='post-merger',ratio=.5,output=str(tmp_path/'output'),data_root=str(tmp_path))
    for i,split in enumerate(('train','dev','eval')):
        image=tmp_path/f'{split}.img';image.write_bytes(bytes([i,1,2]))
        row=dict(id=split,document_id=split,image=image.name,question='What?',answers=['yes'],task='DocVQA')
        manifest=tmp_path/f'{split}.jsonl';manifest.write_text(json.dumps(row)+'\n')
        cfg[f'{split}_rows']=str(manifest)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    return path,load_config(path)


def test_clean_splits_and_duplicate_content(tmp_path):
    _,cfg=fixture_config(tmp_path)
    splits,summary=validate(cfg)
    assert summary['train']['questions']==1
    Path(splits['eval'][0]['image']).write_bytes(Path(splits['train'][0]['image']).read_bytes())
    with pytest.raises(ValueError,match='identical image bytes'):validate(cfg)


def test_invalid_native_insertion(tmp_path):
    p,cfg=fixture_config(tmp_path);cfg['insertion']='pre-merger';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='post-merger'):load_config(p)


def test_schema_and_question_overlap(tmp_path):
    _,cfg=fixture_config(tmp_path)
    p=Path(cfg['dev_rows']);row=json.loads(p.read_text());row['id']='train';p.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='overlap in id'):validate(cfg)
