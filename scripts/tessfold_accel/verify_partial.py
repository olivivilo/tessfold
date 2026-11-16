"""Safe partial CPU audit of finished evidence and active signed train sources."""
import collections,json
from scripts.tessfold_repair.common import ROOT,load,save,sha,now
from scripts.tessfold_accel.system_common import verify
from scripts.tessfold_accel.final_audit import strict_rows
BASE=ROOT/'artifacts/tessfold_acceleration_20261115'


def main():
    verify();out={};source_count=0
    for label,count in [('formal_equivalence',94260),('interleaved',2520),('multi_deployment',864),('batched_serving',144)]:
        if not (BASE/label/'complete.json').exists():continue
        rows=strict_rows(BASE/label);assert len(rows)==count
        for p in (BASE/label).glob('manifest.rank*.json'):
            m=load(p);sources=m.get('source_sha256',m.get('sources',{}))
            if isinstance(sources,str):sources={f'scripts/tessfold_accel/{label}.py':sources}
            for path,digest in sources.items():assert sha(ROOT/path)==digest;source_count+=1
        out[label]=count
    qa=strict_rows(ROOT/'artifacts/compression_lab_v2/column_tree_full_20261108','formal_benchmark.jsonl');ids={r['id'] for r in qa}
    if 'formal_equivalence' in out:
        rr=strict_rows(BASE/'formal_equivalence');models={r['model'] for r in rr};assert len(models)==4
        assert {(r['id'],r['model']) for r in rr}=={(i,m) for i in ids for m in models}
        assert all(r['p_exact'] and r['feature_exact'] and r['generation_exact'] for r in rr)
    if 'multi_deployment' in out:
        rr=strict_rows(BASE/'multi_deployment');pages=load(ROOT/'artifacts/tessfold_system_repair_20261115/multi_pages.json');lookup={p['document_id']:p for p in pages}
        index={(r['page'],r['method'],r['scenario'],r['repeat']):r for r in rr}
        expected={(p['document_id'],m,s,k) for p in pages for m in ['REF','Column','Tree'] for s in ['cold','feature_processor','feature_text_only','prefix_kv'] for k in range(3)}
        assert set(index)==expected and len(index)==864
        from vqa.benchmark_metrics import native_score
        for r in rr:
            page=lookup[r['page']];assert [q['id'] for q in r['queries']]==[q['id'] for q in page['questions']]
            cold=index[(r['page'],r['method'],'cold',r['repeat'])]
            for q,source,ref in zip(r['queries'],page['questions'],cold['queries']):
                assert abs(q['score']-native_score(q['prediction'],source))<1e-10
                if r['scenario'] in ['feature_processor','feature_text_only']:assert q['generated_ids']==ref['generated_ids']
            for k in [1,2,5,10]:assert abs(r['amortization'][str(k)]-(r['ingest_ms']+sum(q['milliseconds'] for q in r['queries'][:k]))/k)<1e-8
    for label in ['qwen_full_pass1','qwen_full_tree25','qwen_full_stable50']:
        for p in (BASE/label).glob('*_r*/manifest.json'):
            m=load(p);assert m['backbone_trainable']==0
            for path,digest in m['source_sha256'].items():assert sha(ROOT/path)==digest
    save(BASE/'partial_audit.json',dict(time=now(),completed_records=out,checked_source_references=source_count,completed_coverage_and_multi_metrics=True,active_training_sources_unchanged=True,not_final_audit=True))
    print(json.dumps(load(BASE/'partial_audit.json')))

if __name__=='__main__':main()
