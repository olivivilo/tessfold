"""Standalone browser review of actual gold/teacher/student answers, with image links."""
import json,os
from urllib.parse import quote
from scripts.report_compression_lab_v2 import ROOT,BASE,load
from scripts.train_compression_lab_v2 import read
def main():
 configs=[load(ROOT/p) for name in ['screen_configs.json','expanded_configs.json','full_configs.json'] for p in load(BASE/name)]
 source={r['id']:r for name in ['validation','confirmation'] for r in read(BASE/f'{name}.jsonl')}
 items=[]
 for cfg in configs:
  out=ROOT/cfg['output']
  for pattern in ['validation.*.jsonl','confirmation.*.jsonl']:
   for path in sorted(out.glob(pattern)):
    for row in read(path):
     if row.get('student_score') is None:continue
     item={k:row[k] for k in ['id','task','cohort','stage','question','answers','teacher_prediction','student_prediction','teacher_score','student_score','kl']}
     item.update(run=cfg['name'],ratio=cfg['ratio'],position=cfg['insertion'],final=row['stage']==f'epoch{cfg["epochs"]}',
      image=quote(os.path.relpath(ROOT/source[row['id']]['image'],BASE)))
     items.append(item)
 payload=json.dumps(items,ensure_ascii=False).replace('<','\\u003c')
 template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UND 新压缩生成器：逐题对照</title>
<style>body{font:15px/1.55 system-ui;margin:24px;color:#182531;background:#f5f7fa}select,input,button{font:inherit;padding:7px;margin:4px}main{display:grid;grid-template-columns:42% 58%;gap:16px}article,aside{background:white;padding:18px;border-radius:8px}article{max-height:85vh;overflow:auto}img{max-width:100%;max-height:78vh;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.meta{color:#536575;font-size:13px}.answer{border-left:3px solid #638eb5;padding-left:12px}button.row{display:block;text-align:left;width:98%;background:white;border:1px solid #d8e0e5;margin:8px 0}.bad{color:#a32b2b}.good{color:#187243}@media(max-width:900px){main{display:block}}</style>
<h1>冻结 UND：第二轮逐题作答对照</h1><p>所有数据均来自实际独立生成。得分为各任务原生指标；字段exact不等于文档级F1。本页展示筛选集和独立确认集，均为64 token生成上限；23565QA正式评测使用256上限，其逐题结果另见各full run的benchmark/answers.csv。</p>
<div id="filters"></div><div><input id="search" placeholder="搜索问题、gold或作答" size="36"><label><input id="final" type="checkbox" checked>仅固定末轮</label><label><input id="drop" type="checkbox">仅teacher得分更高</label></div>
<p id="count"></p><main><article><div id="rows"></div><button id="prev">上一页</button><button id="next">下一页</button></article><aside><div id="detail"></div><img id="page" loading="lazy"></aside></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.querySelector('#data').textContent),filters=document.querySelector('#filters'),selects={};let page=0,found=[];
for(const [key,label] of [['cohort','集合'],['ratio','保留率'],['position','位置'],['task','任务'],['run','配置']]){let s=document.createElement('select');s.add(new Option(label+'：全部',''));for(const v of [...new Set(data.map(x=>x[key]))].sort())s.add(new Option(v,v));filters.append(s);selects[key]=s;s.onchange=()=>{page=0;render()}}
function line(parent,label,value,cls=''){const p=document.createElement('pre');p.className=cls;p.textContent=label+value;parent.append(p)}
function detail(x){const d=document.querySelector('#detail');d.replaceChildren();line(d,'配置：',x.run,'meta');line(d,'集合 / 阶段：',x.cohort+' / '+x.stage,'meta');line(d,'问题：',x.question);line(d,'Gold：',JSON.stringify(x.answers),'answer');line(d,'Teacher：',x.teacher_prediction,'answer');line(d,'Student：',x.student_prediction,'answer');line(d,'得分teacher→student：',x.teacher_score.toFixed(4)+' → '+x.student_score.toFixed(4));line(d,'KL：',x.kl.toFixed(6));document.querySelector('#page').src=x.image}
function render(){const query=document.querySelector('#search').value.toLowerCase();found=data.filter(x=>Object.entries(selects).every(([k,s])=>!s.value||String(x[k])===s.value)&&(!document.querySelector('#final').checked||x.final)&&(!document.querySelector('#drop').checked||x.teacher_score>x.student_score)&&(!query||[x.question,JSON.stringify(x.answers),x.teacher_prediction,x.student_prediction].join(' ').toLowerCase().includes(query)));page=Math.max(0,Math.min(page,Math.ceil(found.length/25)-1));const r=document.querySelector('#rows');r.replaceChildren();document.querySelector('#count').textContent=found.length+'条记录；第'+(page+1)+'页；每页25条。';for(const x of found.slice(page*25,(page+1)*25)){const b=document.createElement('button');b.className='row';b.textContent=x.task+' | '+x.run+' | '+x.question+' | '+x.teacher_score.toFixed(2)+' → '+x.student_score.toFixed(2);b.onclick=()=>detail(x);r.append(b)}if(found.length)detail(found[page*25]);}
for(const id of ['search','final','drop'])document.getElementById(id).oninput=()=>{page=0;render()};document.querySelector('#prev').onclick=()=>{page--;render()};document.querySelector('#next').onclick=()=>{page++;render()};render();
</script></html>'''
 target=BASE/'review.html';tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
 tmp.write_text(template.replace('PAYLOAD',payload));tmp.replace(target);print(target,len(items))
if __name__=='__main__':main()
