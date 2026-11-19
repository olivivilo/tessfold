"""Strict report for the completed isolated diagnostics; refuses partial tables."""
import csv
import json
import math
import statistics as st
from pathlib import Path
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from scripts.diagnose_meta_compress import ROOT,OUT,VARIANTS
from scripts.diagnose_meta_compress_factorial import VARIANTS as FACTORIAL
from scripts.train_meta_compress import sha,atomic_json


def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def table(headers,rows):
    def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in rows])
def f(v):return f'{v:.5f}'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'samples.jsonl')==protocol['samples_sha256']
    assert sha(ROOT/'artifacts/meta_compress/initial.pt')==protocol['original_initial_sha256']
    assert sha(ROOT/'artifacts/meta_compress/meta_compress.pt')==protocol['original_final_sha256']
    samples=read(OUT/'samples.jsonl');ids={r['id'] for r in samples};fit=next(r for r in samples if r['role']=='fit')
    results=[];records=[];traces={};grad_rows=[]
    variants=VARIANTS+FACTORIAL
    for cfg in variants:
        out=(OUT if cfg in VARIANTS else OUT/'factorial')/cfg['name'];complete=json.loads((out/'complete.json').read_text())
        assert complete['steps']==50 and complete['base_trainable_parameters']==0
        ev=read(out/'evaluations.jsonl');trace=read(out/'training.jsonl');traces[cfg['name']]=trace
        assert [r['step'] for r in trace]==list(range(1,51))
        expected=ids if cfg['kind']!='free' else {fit['id']}
        assert {(r['id'],r['stage']) for r in ev}=={(i,s) for i in expected for s in ('initial','final')}
        assert len(ev)==len(expected)*2
        for r in ev:r['variant']=cfg['name'];r['ratio']=cfg['ratio'];records.append(r)
        a=next(r for r in ev if r['role']=='fit' and r['stage']=='initial')
        b=next(r for r in ev if r['role']=='fit' and r['stage']=='final')
        held={s:[r for r in ev if r['role']=='holdout' and r['stage']==s] for s in ['initial','final']}
        result={'variant':cfg['name'],'config':cfg,'fit_initial':a,'fit_final':b,
            'holdout':{s:{'n':len(v),'score_percent':100*st.mean(r['student_score'] for r in v) if v else None,
                         'kl':st.mean(r['prediction'] for r in v) if v else None} for s,v in held.items()},
            'update_path_length':sum(r['update_norm'] for r in trace),
            'median_clip_multiplier':st.median(r['clip_multiplier'] for r in trace),
            'parameter_changes':json.loads((out/'parameter_changes.json').read_text())}
        results.append(result)
        ga=json.loads((out/'gradients.initial.json').read_text());gb=json.loads((out/'gradients.final.json').read_text())
        for stage,g in [('initial',ga),('final',gb)]:
            grad_rows.append([cfg['name'],stage,*[f(g['norms'][k]) for k in ['prediction','entropy','collapse']],
                *[f(g['cosines'][k]) for k in ['prediction__entropy','prediction__collapse','entropy__collapse']]])
    controls=read(OUT/'controls/evaluations.jsonl');cc=json.loads((OUT/'controls/complete.json').read_text())
    assert cc['evaluations']==48 and len(controls)==48
    assert {(r['id'],r['method'],r['ratio']) for r in controls}=={(i,m,q) for i in ids for m in ('mean','sample') for q in (.1,.25,.5)}
    for r in controls:r['variant']='fixed_'+r['method'];r['stage']='fixed';records.append(r)
    full_budget=read(OUT/'full_budget/records.jsonl')
    assert json.loads((OUT/'full_budget/complete.json').read_text())['records']==3 and len(full_budget)==3
    for r in full_budget:
        assert r['tokens']==r['compressed_tokens'] and r['prompt_and_position_length_unchanged']
        r['variant']='dense_full_budget';r['stage']='fixed';r['ratio']=1.;records.append(r)
    fieldnames=list(dict.fromkeys(k for r in records for k in r))
    with (OUT/'answers.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fieldnames);w.writeheader();w.writerows(records)
    interface=json.loads((OUT/'interface_audit.json').read_text())
    precision=json.loads((OUT/'controls/feature_precision_audit.json').read_text())
    directions=json.loads((OUT/'precision/complete.json').read_text())
    matrix_manifest=json.loads((OUT/'matrices/complete.json').read_text())
    assert len(matrix_manifest['matrices'])==5
    assert all(r['code_sha256']==sha(ROOT/'scripts/visualize_meta_compress_diagnosis.py') for r in matrix_manifest['matrices'])
    task_grads=read(OUT/'task_gradients/records.jsonl')
    assert json.loads((OUT/'task_gradients/complete.json').read_text())['records']==16 and len(task_grads)==16
    assert {(r['id'],r['state']) for r in task_grads}=={(i,s) for i in ids for s in ['original_initial','original_20k_final']}
    extension=json.loads((OUT/'factorial/protocol.json').read_text())
    summary={'protocol':protocol,'extension_protocol':extension,'interface':interface,'results':results,'fixed_controls':controls,'full_budget':full_budget,'precision':precision,'finite_differences':directions,'task_gradients':task_grads}
    atomic_json(OUT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for color_index,cfg in enumerate(variants):
        tr=traces[cfg['name']]
        for ax,key in zip(axes.flat,['prediction','entropy','collapse','update_norm']):
            ax.plot([r['step'] for r in tr],[r[key] for r in tr],label=cfg['name'],lw=1.25,
                color=plt.get_cmap('tab20')(color_index))
    for ax,label in zip(axes.flat,['Teacher-prefix KL (nats)','Row entropy (nats)','Collapse (max column mass)','Actual update L2 norm']):
        ax.set_ylabel(label);ax.set_xlabel('Single-image update');ax.grid(alpha=.2)
    axes[0,0].set_yscale('log');axes[1,0].set_yscale('log');axes[1,1].set_yscale('log')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(OUT/'ablation_curves.png',dpi=160);fig.savefig(OUT/'ablation_curves.pdf');plt.close(fig)
    main_rows=[]
    for r in results:
        a,b=r['fit_initial'],r['fit_final'];h=r['holdout']
        hp='N/A' if h['final']['n']==0 else f"{h['initial']['score_percent']:.2f} → {h['final']['score_percent']:.2f}"
        hk='N/A' if h['final']['n']==0 else f"{h['initial']['kl']:.4f} → {h['final']['kl']:.4f}"
        main_rows.append([r['variant'],r['config']['ratio'],f(a['prediction'])+' → '+f(b['prediction']),
            f"{100*a['student_score']:.2f} → {100*b['student_score']:.2f}",hp,hk])
    fixed_rows=[]
    for ratio in [.1,.25,.5]:
        for method in ['mean','sample']:
            subset=[r for r in controls if r['method']==method and r['ratio']==ratio]
            train=next(r for r in subset if r['role']=='fit');hold=[r for r in subset if r['role']=='holdout']
            fixed_rows.append([method,ratio,f(train['prediction']),f"{100*train['student_score']:.2f}",
                f(st.mean(r['prediction'] for r in hold)),f"{100*st.mean(r['student_score'] for r in hold):.2f}"])
    sample_rows=[[r['role'],r['task'],r['id'],r['diagnostic_tokens'],r['diagnostic_teacher_prediction']] for r in samples]
    configs=[[v['name'],v['kind'],v['objective'],v['optimizer'],v['lr'],v['clip'],v['ratio']] for v in variants]
    update_rows=[[r['variant'],f"{r['median_clip_multiplier']:.7g}",f"{r['update_path_length']:.7g}",
        f(r['fit_final']['entropy_fraction']),f(r['fit_final']['effective_support']),f(r['fit_final']['collapse'])] for r in results]
    pred_rows=[[r['variant'],r['fit_initial']['student_prediction'],r['fit_final']['student_prediction'],fit['diagnostic_teacher_prediction']] for r in results]
    text='''# MetaCompress Failure Diagnosis: Input Audit and Twelve Interventions

This report documents completed isolated diagnostics; original 20k/2-epoch training and checkpoints remain. Twelve runs each complete 50 single-image updates, alongside 48 fixed compression controls, 3 full-budget controls, and BF16/FP32 directional finite differences. This is not a new formal benchmark or test-based checkpoint selection.

## Entry Points and Raw Records

- Experiment script: [`scripts/diagnose_meta_compress.py`](../scripts/diagnose_meta_compress.py).
- Slurm entry point: [`scripts/slurm_meta_compress_diagnosis.sh`](../scripts/slurm_meta_compress_diagnosis.sh).
- Fixed controls: [`scripts/diagnose_meta_compress_controls.py`](../scripts/diagnose_meta_compress_controls.py).
- Additional KL-only clipping controls and local full-loss controls: [`scripts/diagnose_meta_compress_factorial.py`](../scripts/diagnose_meta_compress_factorial.py), first  `--prepare`, then run separately  `--variant 0` / `--variant 1`. The first frozen source version and protocol remain unchanged.
- Precision audit: [`scripts/diagnose_meta_compress_precision.py`](../scripts/diagnose_meta_compress_precision.py).
- Cross-task gradient audit: [`scripts/diagnose_meta_compress_task_gradients.py`](../scripts/diagnose_meta_compress_task_gradients.py).
- Full-token-budget controls: [`scripts/diagnose_meta_compress_full_budget.py`](../scripts/diagnose_meta_compress_full_budget.py).
- Matrix exports / visualizations: [`scripts/visualize_meta_compress_diagnosis.py`](../scripts/visualize_meta_compress_diagnosis.py), `.pt`  contains P, grid, configuration, and checkpoint SHA-256.
- Unified status/report entry point: `bash scripts/run_meta_compress_diagnosis.sh status|report`.
- Report entry: [`scripts/report_meta_compress_diagnosis.py`](../scripts/report_meta_compress_diagnosis.py).
- Original answers per question: [`answers.csv`](../artifacts/meta_compress_diagnosis/answers.csv); complete values: [`summary.json`](../artifacts/meta_compress_diagnosis/summary.json). Each run also retains 50-step logs, initial/final component gradients, parameter changes, and final checkpoints.

```bash
# Prepare the fixed cohort once; when  cohort; Existing protocol exists, refuse overwrite
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.diagnose_meta_compress --prepare
sbatch scripts/slurm_meta_compress_diagnosis.sh
# Fixed controls should also run on a  compute node execution
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 .venv/bin/python -m scripts.diagnose_meta_compress_controls
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.report_meta_compress_diagnosis
```

## 1. Fixed Protocol and Scope of Conclusions

The model is official MonkeyOCRv2-B-Und, with visual encoder, official merger, and Qwen3-1.7B frozen. Retain the original post-merger insertion, BF16 backbone, and FP32 generator. Train with full-vocabulary forward KL under a shared teacher-generated answer prefix, including EOS; gold never enters the loss.

The official merger belongs to **MonkeyOCRv2**, aligning 768-dimensional encoder tokens to 2,048-dimensional Qwen inputs. It is not MetaCompress and does not reduce token count in this instance. The following two Qwen forwards share weights but have independent state:

```mermaid
flowchart LR
    V[frozen official encoder] --> M[frozen official merger]
    M --> X[original token X]
    X --> T[frozen Qwen teacher]
    X --> G[Compression-matrix generator]
    G --> P[row-wise softmax yields P]
    P --> C[compressed token PX]
    X --> C
    C --> S[frozen Qwen student]
    T --> K[shared teacher prefix  KL]
    S --> K
    P --> R[Entropy and Collapse]
    K --> L[Loss ablation]
    R --> L
    L -. update generator only .-> G
```

Eight distinct images are hash-selected from the original 20k **training records**: one DocVQA image for single-image overfitting and seven independent training-source held-out images spanning all 7 tasks with training data. OCRBench has none and is excluded. Images do not overlap official val/test. Selection requires teacher score=1, a nonempty/non-None answer of at most 12 tokens, and generally at most 4,096 image tokens (5,200 for KLC/DeepForm). **Thus teacher scores are 100% by construction, not a new performance result. One held-out image per task cannot establish generalization or significance.**

All generator variants share one initialization. Repeat the same image/question 50 times at batch=1; gradient statistics differ from original batch=36, so step counts and hyperparameter effects do not directly extrapolate. Free P has different parameter count and optimizer and serves only as an optimizability oracle, not an isolated generator-architecture ablation. Step 50 is fixed as final, with no gold/held-out checkpoint selection.

'''+table(['Role','Task','Question ID','Original token count','Teacher / gold reference'],sample_rows)+'''

## 2. Input, Cache, and Causal-Shift Audit

On the fitting image, native official and cached paths generate identically; maximum fresh-versus-cached encoder-feature absolute error  '''+f(interface['native_cached_feature_max_abs'])+''', raw difference  '''+f(interface['native_cached_raw_max_abs'])+'''. Maximum cached-versus-recomputed teacher-logit difference:  '''+f(interface['cached_logits_max_abs'])+'''; explicit P=I through the same multiply/cast/placeholder path gives KL='''+f(interface['identity_KL'])+'''. Separately recomputed first/last response logits match the corresponding full teacher-forcing logits. Trainable backbone parameters: 0.

These checks support the current data path and causal shift and do not support explanations involving the wrong encoder cache, images omitted from the decoder, or shared teacher/student KV. Each generation starts with independent state. Numerical equivalence is audited on one fitting image only, so not all possible data issues are ruled out.

## 3. Twelve Evaluated Configurations

'''+table(['Name','Architecture','Optimization objective','Optimizer','lr','Global norm clip','Retention fraction'],configs)+'''

`generator` is the original MetaCompress generator.`free` Use log of the fitting-image generator P as independent trainable logits, then row softmax. This is not the paper’s Section-4 initialization/SGD setup and cannot infer on independent images.`local` is an explicit modification: add fixed local bias to original logits, 0 within each balanced contiguous token group and log(1e-6) outside, then softmax. It remains convex but groups contiguous native 2×2-block **sequence** positions, not full 2D neighborhoods, and differs from the paper. Adam is an additional diagnostic; the paper-style primary experiment retains SGD. Full loss keeps KL + H + max-column-mass, all weighted 1.

Log/softmax round trips need not reproduce P bitwise; BF16 may amplify tiny differences. Recompute every variant’s initial KL rather than substituting generator initial loss for free P.

## 4. Initial/Final Outcomes on Fitting and Held-Out Images

Scores below are the existing single-question scorers ×100: ANLS for DocVQA/InfoVQA, relaxed accuracy for ChartQA, max(containment accuracy, ANLS) for DT-VQA, denotation for WTQ, and single-field exact match for DeepForm/KLC, **not official DUE corpus F1**. The held-out seven-question mean is descriptive across mixed metrics, not an official multitask score. All KL values use the same teacher responses.

'''+table(['Experiment','Retention fraction','Fit KL: initial → final','Fit score: initial → final','Holdout score: initial → final','Holdout KL: initial → final'],main_rows)+'''

Training questions: **'''+fit['question']+'''**; gold / teacher: **'''+fit['diagnostic_teacher_prediction']+'''**.

'''+table(['Experiment','Initial generation','Step-50 generation','teacher'],pred_rows)+'''

## 5. Gradient Magnitudes, Conflicts, and Actual Updates

Differentiate KL, H, and collapse separately on the same fitting question. Report concatenated trainable-parameter L2 norms and direction cosines per variant. Magnitudes do not establish full-dataset behavior; cosine<0 indicates local first-order conflict.

'''+table(['Experiment','Stage','||g_KL||','||g_H||','||g_C||','cos(KL,H)','cos(KL,C)','cos(H,C)'],grad_rows)+'''

'''+table(['Experiment','Median clipping multiplier','Actual update path length over 50 steps','Final row entropy / log(n)','Final exp(H)','Final collapse'],update_rows)+'''

Free P and generator parameter spaces differ, so update-path lengths are not directly comparable. Unclipped runs use clip=inf to measure norms without finite-threshold rescaling. BF16 can round away forward effects of small FP32 updates. Per-image fractions of BF16 compressed-feature elements changed by the original formal checkpoint versus initial are in  `controls/feature_precision_audit.json`, not complete failure of every gradient.

![Training trajectory](../artifacts/meta_compress_diagnosis/ablation_curves.png)

Each step’s logged loss is **before** its update. Initial/final tables independently recompute before the first and **after** the 50th update.

## 6. Training-Free Retention and Local-Compression Controls

Fixed `mean`: balanced m groups in native order, taking each group’s mean;`sample`: select the middle original token/group, giving one-hot row-stochastic P. Question/answer/decoder attention do not select tokens. These distinguish reduced count from dense mixing; higher-budget gains are not same-budget method gains.

'''+table(['Fixed method','Retention fraction','fit KL','Fit score','holdout KL','Holdout score'],fixed_rows)+'''

## 7. Auditability and Limitations

Generate this report only after checks pass for all 12 runs, the order of 50 updates in each, uniqueness per image/stage, and coverage of 48 fixed controls. Original initial/final checkpoint SHA-256 values must match the protocol. All questions, gold, and teacher/student answers are retained in answers.csv. The original benchmark is not used for checkpoint selection.

This is single-seed/single-image overfitting with tiny held-out mechanism diagnostics, not task-balanced multi-image training, repeated seeds, or full independent evaluation. Local bias, Adam, and larger budgets are separate interventions.
'''
    drows=[]
    for r in directions['records']:
        if r['step_norm']==0:
            drows.append([r['precision'],0,f(r['kl']),f(r['repeat_delta']),'N/A','N/A'])
        else:
            drows.append([r['precision'],r['step_norm'],f(r['kl']),f(r['actual_delta']),
                f(r['first_order_predicted_delta']),f(r['compressed_elements_changed_fraction'])])
    text+='''
## 8. Numerical Precision and Small Updates

Starting from the original generator initialization, move a specified **total-parameter L2 distance** along the unit negative KL gradient at each precision. Reset between trials; this is not sequential training. BF16 cache source and teacher target are fixed. FP32 mode converts the same BF16 checkpoint values to FP32 for student computation, rather than loading separate high-precision pretrained weights. Deltas use each mode's own baseline; absolute KL across modes does not establish model quality.

'''+table(['Student accuracy','Parameter-step L2 norm','KL','Measured delta (step 0: repeated-forward difference)','First-order predicted delta','Fraction of changed compressed-feature elements'],drows)+'''

BF16 casts make the parameter-to-forward-loss map piecewise and nonsmooth; finite differences and first-order backpropagation predictions can disagree at very small steps. This audits one image and local scales, not whether FP32 is optimal for full training.

Precision controls retain native BF16 model’s FP32 RoPE buffers. The early version applied to the entire model  `.to(bfloat16)`also changes these buffers, so that version is excluded from conclusions; records:  `precision_invalid_buffer_cast/`. This was repaired in the precision-audit script only; original formal training never used that additional dtype cast.
'''
    grows=[]
    for r in task_grads:
        g=r['norms'];c=r['cosines']
        dot=g['prediction']**2+g['prediction']*g['entropy']*c['prediction__entropy']+g['prediction']*g['collapse']*c['prediction__collapse']
        grows.append([r['task'],r['id'],r['state'],f(g['prediction']),f(g['entropy']),f(g['collapse']),
            f(g['collapse']/max(g['prediction'],1e-30)),f(c['prediction__entropy']),f(c['prediction__collapse']),f(dot)])
    text+='''
## 9. Cross-Task Gradient Audit

A fit-image gradient cannot be generalized to the entire dataset. The original generator is therefore audited at initial and original-20k-final states on 8 images across 7 tasks. No training or prior averaging of image gradients is performed. Reported per-image norms/cosines are not the exact global gradient of the original batch=36 training.

'''+table(['Task','Question ID','Status','||g_KL||','||g_H||','||g_C||','C/KL gradient-norm ratio','cos(KL,H)','cos(KL,C)','dot(KL,total)'],grows)+'''

Final column reconstructed from three norms/cosines: dot(g_KL,g_total)=||g_KL||²+||g_KL||||g_H||cos(KL,H)+||g_KL||||g_C||cos(KL,C). Negative values mean that first-order descent along total increases KL; a positive global clipping multiplier cannot change that sign. BF16 discrete small updates need not exactly follow this first-order approximation.
'''
    byname={r['variant']:r for r in results}
    factorial_rows=[]
    for objective,clip_name,free_name in [('Full loss','original_all_sgd_clip001','all_sgd_no_clip'),('KL-only','kl_sgd_clip001','kl_sgd_no_clip')]:
        a=byname[clip_name]['fit_final'];b=byname[free_name]['fit_final']
        factorial_rows.append([objective,f(a['prediction']),a['student_prediction'],f(b['prediction']),b['student_prediction']])
    k=byname['kl_sgd_no_clip'];free=byname['free_dense_kl_adam'];local=byname['local_bias_all_adam'];dense=byname['kl_adam_keep025']
    initial_grads=[r for r in task_grads if r['state']=='original_initial']
    ratios=[r['norms']['collapse']/max(r['norms']['prediction'],1e-30) for r in initial_grads]
    conflicts=sum(r['cosines']['prediction__collapse']<0 for r in initial_grads)
    def total_dot(r):
        g=r['norms'];c=r['cosines']
        return g['prediction']**2+g['prediction']*g['entropy']*c['prediction__entropy']+g['prediction']*g['collapse']*c['prediction__collapse']
    total_conflicts=sum(total_dot(r)<0 for r in initial_grads)
    text+='''
## 10. Updated Evidence: Supported and Unsupported Explanations

**The main issue is the combination of compressor optimization and structural constraints, not evidence against the encoder or Qwen3-1.7B capacity. Removing clipping alone, changing to Adam alone, or increasing the token budget alone is insufficient in these tests.**

### 10.1 A 2×2 Comparison with the Same SGD and Initialization

'''+table(['Optimization objective','clip=.01: final KL','clip=.01: generation','Unclipped: final KL','Unclipped: generation'],factorial_rows)+'''

All four use LR=.001, the same training image, 50 updates, and 10% retention. Unclipped KL-only changes KL from  '''+f(k['fit_initial']['prediction'])+' decreases to  '+f(k['fit_final']['prediction'])+''', with independently generated correct answers, whereas unclipped full loss still fails. Update magnitude and objective composition jointly matter; increasing SGD LR alone is insufficient. Original clipping/update evidence across 1,112 steps remains in failure_audit.

### 10.2 Regularizer Direction Interacts with Initialization

Original-initialization median collapse/KL gradient-norm ratio over 8 images: **'''+f(st.median(ratios))+'**, range **'+f(min(ratios))+'～'+f(max(ratios))+'**; of which **'+str(conflicts)+'/8** images have negative collapse/KL gradient cosine; **'+str(total_conflicts)+'''/8** images have a negative dot product between full-total and KL gradients: first-order total descent increases prediction KL. This is measured component-gradient evidence, not an inference from loss magnitudes. It applies to these 8 conditionally selected images, not the full dataset.

Final local-bias + full-loss KL:  '''+f(local['fit_final']['prediction'])+', collapse='+f(local['fit_final']['collapse'])+''', fitting the current question. Entropy/collapse are not universally ineffective; they interact with initial support, feature scale, optimizer, and clipping. Low-collapse solutions favored by current dense initialization may not preserve evidence.

### 10.3 High Entropy Does Not Prove Failure; Low Entropy Does Not Guarantee Success

Final free-P KL:  '''+f(free['fit_final']['prediction'])+', row entropy  '+f(free['fit_final']['entropy'])+''', also fits one question; final KL-only-SGD row entropy remains  '''+f(k['fit_final']['entropy'])+'''. The claim that high row entropy inevitably prevents correct answers is therefore too strong. Conversely, at 25% the final row entropy of dense KL-only Adam is only  '''+f(dense['fit_final']['entropy'])+', but maximum column mass is  '+f(dense['fit_final']['collapse'])+' / '+str(dense['fit_final']['compressed_tokens'])+''', with many rows collapsing onto one source token and answers still incorrect. Entropy or single-question KL alone cannot establish general compression quality.

Free P has different degrees of freedom/optimizer. Success proves a usable solution exists for that image/budget/decoder, not an isolated generator-capacity defect. KL-only SGD also fits, ruling out complete generator untrainability.

### 10.4 The 10% Budget Is Difficult, but Does Not Explain Every Failure

See Section 6 for fixed-selection/mean controls. Higher-retention local preservation helps some images; more global dense rows do not automatically restore text. Single-image 10% fitting disproves impossibility for that image only, not guaranteed recovery of arbitrary document QA.

### 10.5 Multi-Image Generalization Remains Unresolved

Learning to answer 38 on one image may preserve only the local signal for that question, or exploit a shortcut specific to that teacher response. It does not show preservation of the full page or all possible questions. Inspect held-out initial/final answers and KL separately from fitting results; perfect training accuracy does not establish a repaired method. No formal model is selected or replaced on this evidence.

Dense KL-only Adam at 25% and 50% each improves held-out mean **14.29→28.57**; both gains come from the same WTQ question, whose answer changes from  `1` changes to the correct answer `10` , with the other six scores unchanged. Retain these gains rather than claiming that no setting improves anything. However, held-out mean KL worsens from **1.5030→1.9077** and **1.5189→1.8846**, with single-column concentration and an incorrect training answer. One additional correct answer among seven does not establish stable generalization or justify ignoring matrix degeneration.

Next, run task-balanced multi-image experiments with image-disjoint training-source splits and multiple questions per image to discourage memorizing one answer. Compare dense KL-only SGD against local-bias + full loss, monitoring generation scores, KL, collapse/m, gradient norms, and BF16 forward changes. Test 50%/25%/10% separately, select on independent development generation, and only then run the full benchmark. Current evidence does not require unfreezing the LLM or justify repeating the original 20k recipe unchanged.
'''
    full_failures=sum(r['student_score']==0 for r in full_budget)
    lead='''
Main finding: input/cache/causal-shift audits pass; **without token reduction or prompt shortening, original dense-P mixing alone makes '''+str(full_failures)+'''/3 questions answered correctly by the teacher fall to zero**. With the same SGD, removing strong clipping enables single-image fitting under KL-only, whereas removing clipping alone still fails under the full loss. A local bias can also enable fitting with the full loss. Diagnostics directly support gradient conflict, dense-matrix degeneration, and BF16 rounding at small updates. **Single-image fitting does not establish recovery on independent images; see Sections 10–11.**
'''
    text=text.replace('\n## Entry Points and Raw Records',lead+'\n## Entry Points and Raw Records',1)
    fbrows=[[r['task'],r['id'],r['tokens'],r['teacher_prediction'],r['student_prediction'],f(r['prediction']),f"{100*r['student_score']:.2f}",f(r['entropy_fraction'])] for r in full_budget]
    text+='''
## 11. Full-Budget Control: Mixing Without Token Reduction

For two DocVQA and one ChartQA images, set the original generator to retention=1: m=n, with identical image placeholders, question prompt, and contiguous position length to teacher. Its pooling kernel is then 1, but softmax(QKᵀ/sqrt(dc)) **is not identity**. Compare same-budget original X (teacher) and mixed PX without training the generator.

'''+table(['Task','Question ID','n=m','Original teacher','Generation after mixing','KL','Score after mixing','Row entropy / log(n)'],fbrows)+'''

This tests whether shortening alone explains failure, but does not rule out extra positional/budget effects. keep=1/kernel=1 and keep=.1/fractional-max kernel=3 are not identical conditions.
'''
    text+='''
The original teacher answers all three correctly; after mixing,  '''+str(full_failures)+'''  questions score zero. On these images, initialization-induced mixing alone already breaks answers; degradation cannot be attributed entirely to 90% reduction or shorter text positions. The paper's shared Wq=Wk theoretical initialization does not guarantee token preservation under actual UND feature statistics and current position/pooling parameterization. This is evidence of a transfer difference, not evidence that the paper's original model has the same problem.
'''
    text+='''
## 12. Compression Matrices for the Same Image

![Compression matrix](../artifacts/meta_compress_diagnosis/matrices/matrix_comparison.png)

Color is log10(P), with a shared range [−6,0]. For visualization, take the maximum over each 3 source tokens so single-column peaks survive resizing. **This downsamples the display only; saved P and all losses are unchanged.** Horizontal axis: native 2×2-block sequence positions; vertical: output tokens. Panels show their own m; 25% retention has a different row count. Original matrices: [`matrices/`](../artifacts/meta_compress_diagnosis/matrices/complete.json). Directly use  `torch.load(path, weights_only=True)['P']` readout, without loading a decoder.
'''
    (ROOT/'docs/METACOMPRESS_INTERVENTION_RESULTS.md').write_text(text)
    print(json.dumps({'variants':len(results),'updates':600,'records':len(records),'report':'docs/METACOMPRESS_INTERVENTION_RESULTS.md'},ensure_ascii=False))


if __name__=='__main__':main()
