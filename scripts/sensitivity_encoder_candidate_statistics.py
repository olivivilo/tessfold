"""Report paired sensitivity to OT convergence, never select or fit a proxy."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'
def main():
    rows=[]
    with (OUT/'trials.csv').open() as h:
        for r in csv.DictReader(h):
            if r['kind']=='random':rows.append(r)
    rows.sort(key=lambda r:(r['document_id'],float(r['ratio']),r['family'],int(r['replicate'])))
    x=np.array([float(r['loss_sinkhorn_full_question_excess']) for r in rows]).reshape(3601,12,2)
    old=np.array([float(r['loss_V1_rec']) for r in rows]).reshape(3601,12,2)
    y=np.array([float(r['score_drop']) for r in rows]).reshape(3601,12,2)
    err={r['id']:r['max_marginal_error'] for p in (OUT/'exact_ot').glob('*.json') for r in json.loads(p.read_text())['trials']}
    e=np.array([err[r['id']] for r in rows]).reshape(3601,12,2)
    dx=x[:,:,1]-x[:,:,0];do=old[:,:,1]-old[:,:,0];dy=y[:,:,1]-y[:,:,0]
    good=(dx*dy>0);oldgood=(do*dy>0);nontie=(dy!=0)&(dx!=0)&(do!=0)
    rng=np.random.default_rng(20261022);draw=rng.integers(3601,size=(1000,3601));result={}
    for name,mask in [('all',np.ones_like(nontie)),('both_trial_max_marginal_error_le_1e_6',e.max(2)<=1e-6)]:
        use=nontie&mask;n=use.sum(1);a=(good&use).sum(1);b=(oldgood&use).sum(1);nn=n[draw].sum(1)
        result[name]={'nontie_pairs':int(n.sum()),'OT_order_accuracy':float(a.sum()/n.sum()),'OT_accuracy_image_bootstrap_ci95':list(map(float,np.quantile(a[draw].sum(1)/nn,[.025,.975]))),'old_rec_order_accuracy_same_pairs':float(b.sum()/n.sum()),'OT_minus_old_rec_accuracy':float((a.sum()-b.sum())/n.sum()),'paired_image_bootstrap_difference_ci95':list(map(float,np.quantile((a-b)[draw].sum(1)/nn,[.025,.975])))}
    result['scope']='supplementary sensitivity, no model or loss selection; exploratory unadjusted intervals among multiple fixed variants'
    (OUT/'statistical_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
