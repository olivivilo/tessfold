"""Export a separate formal-benchmark answer browser for each completed budget.

This only reads the locked benchmark; it never changes model selection.
"""
import json
import os
from urllib.parse import quote
from scripts.report_compression_lab_v2 import ROOT, BASE, load, refresh
from scripts.train_compression_lab_v2 import read


def main():
    source = {r['id']: r for file in ['data/full/val.jsonl', 'data/full/test.jsonl']
              for r in read(ROOT / file)}
    for cfgpath in load(BASE / 'full_configs.json'):
        cfg = load(ROOT / cfgpath)
        directory = ROOT / cfg['output'] / 'benchmark'
        refresh(directory)
        if not (directory / 'complete.json').exists():
            continue
        initial, trained = {}, {}
        for path in sorted(directory.glob('predictions.rank*.jsonl')):
            for row in read(path):
                if row['variant'] == 'initial':
                    initial[row['id']] = row
                elif row['variant'] == 'trained':
                    trained[row['id']] = row
        assert initial.keys() == trained.keys() and len(trained) == 23565
        items = []
        for key, row in trained.items():
            before = initial[key]
            assert before['teacher_prediction'] == row['teacher_prediction']
            items.append(dict(id=key, task=row['task'], question=row['question'],
                answers=row['answers'], teacher=row['teacher_prediction'],
                initial=before['student_prediction'], trained=row['student_prediction'],
                teacher_score=row['teacher_score'], initial_score=before['student_score'],
                trained_score=row['student_score'],
                gain=row['student_score']-before['student_score'],
                tokens=row['tokens'], compressed_tokens=row['compressed_tokens'],
                hit_token_limit=row['hit_token_limit'],
                image=quote(os.path.relpath(ROOT/source[key]['image'], directory))))
        payload = json.dumps(items, ensure_ascii=False).replace('<', '\\u003c')
        html = TEMPLATE.replace('BUDGET', f'{cfg["ratio"]:.0%}').replace('PAYLOAD', payload)
        target = directory / 'review.html'
        tmp = target.with_name(f'review.{os.getpid()}.tmp')
        tmp.write_text(html)
        tmp.replace(target)
        print(target, len(items))


TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>UND formal benchmark: keep BUDGET</title>
<style>body{font:15px/1.55 system-ui;background:#f4f6f8;color:#172b3d;margin:20px}
select,input,button{font:inherit;padding:6px;margin:4px}main{display:grid;grid-template-columns:40% 60%;gap:14px}
article,aside{padding:16px;background:white;border-radius:8px}article{max-height:82vh;overflow:auto}
img{max-width:100%;max-height:75vh;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}
button.row{display:block;text-align:left;width:98%;background:white;border:1px solid #d4dde4}
@media(max-width:900px){main{display:block}}</style>
<h1>Formal per-question comparison: retain BUDGET tokens</h1>
<p>23,565 questions, cap=256, fixed epoch 4; this page does not select models.
Task-native scores; DeepForm/KLC use field exact match. Display answers verbatim.</p>
<select id="task"><option value="">All tasks</option></select>
<select id="change"><option value="">Before/after training: all</option><option value="up">Improved after training</option>
<option value="down">Worse after training</option><option value="same">Unchanged after training</option>
<option value="teacher_gap">Teacher higher than trained</option></select>
<input id="search" placeholder="Search questions, gold answers, or predictions" size="35"><p id="count"></p>
<main><article><div id="rows"></div><button id="prev">Previous page</button><button id="next">Next page</button></article>
<aside><div id="detail"></div><img id="image" loading="lazy"></aside></main>
<script type="application/json" id="data">PAYLOAD</script><script>
const data=JSON.parse(document.querySelector('#data').textContent);let found=[],page=0;
const task=document.querySelector('#task'),change=document.querySelector('#change'),search=document.querySelector('#search');
for(const t of [...new Set(data.map(x=>x.task))].sort())task.add(new Option(t,t));
function line(d,label,v){const p=document.createElement('pre');p.textContent=label+v;d.append(p)}
function detail(x){const d=document.querySelector('#detail');d.replaceChildren();line(d,'Task / ID: ',x.task+' / '+x.id);
line(d,'Question: ',x.question);line(d,'Gold: ',JSON.stringify(x.answers));line(d,'Teacher: ',x.teacher);
line(d,'Initial: ',x.initial);line(d,'Trained: ',x.trained);
line(d,'Score teacher / initial / trained: ',[x.teacher_score,x.initial_score,x.trained_score].map(v=>v.toFixed(4)).join(' / '));
line(d,'Visualtoken: ',x.tokens+' → '+x.compressed_tokens);line(d,'trainedAchieved256 tokenCap: ',x.hit_token_limit);
document.querySelector('#image').src=x.image}
function render(){const q=search.value.toLowerCase();found=data.filter(x=>(!task.value||x.task===task.value)&&
(!change.value||change.value==='up'&&x.gain>1e-12||change.value==='down'&&x.gain< -1e-12||
change.value==='same'&&Math.abs(x.gain)<=1e-12||change.value==='teacher_gap'&&x.teacher_score>x.trained_score)&&
(!q||[x.id,x.question,JSON.stringify(x.answers),x.teacher,x.initial,x.trained].join(' ').toLowerCase().includes(q)));
page=Math.max(0,Math.min(page,Math.ceil(found.length/25)-1));const rows=document.querySelector('#rows');rows.replaceChildren();
document.querySelector('#count').textContent=found.length+' questions; page '+(page+1)+'; 25 questions per page.';
for(const x of found.slice(page*25,(page+1)*25)){const b=document.createElement('button');b.className='row';
b.textContent=x.task+' | '+x.question+' | '+x.initial_score.toFixed(2)+' → '+x.trained_score.toFixed(2);
b.onclick=()=>detail(x);rows.append(b)}if(found.length)detail(found[page*25])}
for(const el of [task,change,search])el.oninput=()=>{page=0;render()};
document.querySelector('#prev').onclick=()=>{page--;render()};document.querySelector('#next').onclick=()=>{page++;render()};render();
</script></html>'''

if __name__ == '__main__':
    main()
