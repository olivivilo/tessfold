"""Check that all selected train image grids fit the learned position tables."""
from concurrent.futures import ThreadPoolExecutor
import json
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from scripts.train_meta_compress import choose_rows, read_rows
from vqa.official_und import ROOT
rows=choose_rows(read_rows('train'),20000)
docs={r['document_id']:r for r in rows}
def check(row):
    with Image.open(ROOT/row['image']) as im:w,h=im.size
    nh,nw=smart_resize(h,w,factor=28,min_pixels=3136,max_pixels=1003520)
    return {'document_id':row['document_id'],'height':h,'width':w,'grid_h':nh//14,'grid_w':nw//14}
with ThreadPoolExecutor(12) as pool:results=list(pool.map(check,docs.values()))
summary={'images':len(results),'max_grid_h':max(r['grid_h'] for r in results),'max_grid_w':max(r['grid_w'] for r in results),
         'out_of_range':[r for r in results if max(r['grid_h'],r['grid_w'])>512]}
(ROOT/'artifacts/meta_compress/image_size_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
assert not summary['out_of_range']
