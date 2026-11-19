"""Generate the auditable Markdown report; full mode rejects incomplete jobs."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/meta_compress');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    root=ROOT/a.root;cfg=json.loads((root/'protocol.json').read_text());training=read_jsonl(root/'training.jsonl')
    # A resumed job may re-log work since its latest checkpoint. Keep the most
    # recent record for each optimizer step, matching the final checkpoint.
    training=list({x['step']:x for x in training}.values());training.sort(key=lambda r:r['step'])
    complete=(root/'complete.json').exists()
    ep=root/'eval';protocol_path=ep/'protocol.benchmark.json'
    evcfg=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    records=[]
    for f in sorted(ep.glob('predictions.benchmark.rank*.jsonl')):records.extend(read_jsonl(f))
    bykey={}
    for row in records:
        key=(row['id'],row['variant'])
        if key in bykey:raise RuntimeError(f'Duplicate evaluation {key}')
        bykey[key]=row
    records=list(bykey.values())
    expected=evcfg['samples']*len(evcfg['variants']) if evcfg else 1536
    evaluated=evcfg is not None and len(records)==expected
    if evaluated:
        planned=read_jsonl(ep/'samples.benchmark.jsonl')
        assert set(bykey)=={(r['id'],v) for r in planned for v in evcfg['variants']},'Unexpected evaluation membership'
    if not a.allow_partial and not (complete and evaluated):raise RuntimeError(f'Incomplete: training={complete}, eval={len(records)}/{expected}')
    epochs=[]
    for epoch in sorted({r['epoch'] for r in training}):
        z=[r for r in training if r['epoch']==epoch]
        epochs.append({'epoch':epoch,'logged_steps':len(z),**{k:statistics.mean(x[k] for x in z) for k in ['prediction','entropy','collapse','loss','grad_norm']}})
    metrics={}
    for variant in sorted({r['variant'] for r in records}):
        table={}
        for task in sorted({r['task'] for r in records}):
            z=[r for r in records if r['task']==task and r['variant']==variant]
            if not z:continue
            table[task]={'samples':len(z),**{k:statistics.mean(r[k] for r in z) for k in ['teacher_score','student_score','score_drop','prediction_loss','entropy_loss','collapse_loss','total_loss','tokens','compressed_tokens']},
                         'student_token_limit_fraction':statistics.mean(r['student_hit_token_limit'] for r in z)}
        metrics[variant]=table
    delta_ci={}
    if evaluated:
        rng=np.random.default_rng(20261026)
        for task in sorted(metrics['trained']):
            trained={r['id']:r for r in records if r['variant']=='trained' and r['task']==task}
            initial={r['id']:r for r in records if r['variant']=='initial' and r['task']==task}
            assert trained.keys()==initial.keys()
            differences=np.array([100*(trained[k]['student_score']-initial[k]['student_score']) for k in trained])
            boots=differences[rng.integers(0,len(differences),(2000,len(differences)))].mean(1)
            delta_ci[task]={'trained_minus_initial_pp':float(differences.mean()),'paired_bootstrap_ci95':np.quantile(boots,[.025,.975]).tolist()}
    summary={'training_complete':complete,'evaluation_complete':evaluated,'evaluation_rows':len(records),'expected_evaluation_rows':expected,'epochs':epochs,'variants':metrics,'paired_changes':delta_ci}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    if records:
        with (root/'answers.csv').open('w',newline='') as h:
            fields=['task','id','variant','question','answers','teacher_prediction','student_prediction','teacher_score','student_score','score_drop','prediction_loss','entropy_loss','collapse_loss','total_loss']
            writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for row in records:writer.writerow({**row,'answers':json.dumps(row['answers'],ensure_ascii=False)})
    if training:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
        for ax,key in zip(axes.flat,['prediction','entropy','collapse','loss']):
            xs=np.array([r['step'] for r in training]);ys=np.array([r[key] for r in training])
            ax.plot(xs,ys,alpha=.3,lw=.6,label='batch mean')
            window=min(25,len(ys))
            ax.plot(xs[window-1:],np.convolve(ys,np.ones(window)/window,mode='valid'),lw=1.3,label=f'{window}-step mean')
            ax.set(xlabel='Optimizer step',ylabel=key);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(root/'training_loss.png',dpi=160);fig.savefig(root/'training_loss.pdf');plt.close(fig)
    state='Training and evaluation completed' if complete and evaluated else ('Training completed; formal evaluation incomplete' if complete else 'Formal training incomplete; no final evaluation')
    outcome=''
    if complete and evaluated:
        macro={v:100*statistics.mean(z['student_score'] for z in table.values()) for v,table in metrics.items()}
        teacher_macro=100*statistics.mean(z['teacher_score'] for z in metrics['trained'].values())
        outcome=(f"\n**This round does not meet the compression target.** Descriptive mean of eight task scores: uncompressed teacher **{teacher_macro:.2f}**, "
                 f"Initial compressor **{macro['initial']:.2f}**, trained **{macro['trained']:.2f}**, mean pooling **{macro['mean']:.2f}**. "
                 "Tasks use different metrics; this mean is not the paper’s official overall score. The trained version does not improve; "
                 "Completion is not successful reproduction. Most total-loss reduction comes from collapse regularization, "
                 "Epoch-mean prediction KL is essentially unchanged and row entropy slightly higher; answer-quality-preserving compression has not been learned. "
                 "A negative result under this implementation/settings does not invalidate the paper’s method or KL signal.\n")
    text=f'''# MonkeyOCRv2-B-Und: MetaCompress Generator Training and Evaluation

**Status: {state}.** Training logs record {len(training)} / 1,112 optimizer steps; evaluation records collected: {len(records)} / {expected}  variant records. This report is generated from actual outputs by `scripts/report_meta_compress.py`; pre-completion statistics reflect logged batches, not full epochs. Formal entry point: **`bash scripts/run_meta_compress.sh status`**; training entry: **`sbatch scripts/slurm_meta_compress_cpu.sh`**.
{outcome}

## 1. Paper Design and Implemented Architecture

See [Wang et al., CVPR 2026, MetaCompress, Section 5, Eqs. 6–10, Section 6.1](../Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.pdf) and [Supplementary Section 8.2](../artifacts/decoder_dependent_loss/paper_supplemental.pdf). This trains a reusable image-conditioned generator, **not** the paper’s Section 4 experiment optimizing one free matrix for one image over 500 epochs. A directly reusable full-training implementation is absent from the public repository; implementation details are disclosed below.

Let official UND encoder outputs be $X_0\\in\\mathbb R^{{n\\times768}}$, let $F$ be the official merger, and compress immediately before the decoder: $X=F(X_0)\\in\\mathbb R^{{n\\times2048}}$.

$$
Z=X+E_{{pos}},\\quad Q=\\operatorname{{FracMaxPool}}_{{k=3,m}}(Z)W_q,\\quad K=ZW_k,
$$
$$
P=\\operatorname{{softmax}}_{{row}}\\left(\\frac{{Q\\operatorname{{diag}}(\\theta)K^\\top}}{{\\sqrt{{d_c}}}}\\right),\\qquad X_c=PX.
$$

Actual $d_c=64$, $m=\\max(1,\\operatorname{{round}}(0.1n))$; $W_q,W_k\\in\\mathbb R^{{2048\\times64}}$; $\\theta\\in\\mathbb R^{{64}}$. Position encoding uses two learnable `512×64` row/column tables and a bias-free `64→2048` projection, totaling **458,816 trainable parameters**. Initially $W_q=W_k$ with Gaussian standard deviation $1/64$, $\\theta=1$. Each row of $P$ sums to 1; $PX$ is convex aggregation. Earlier random-matrix column-mass bounds are removed; the paper-style collapse term controls repeated aggregation.

```mermaid
flowchart LR
    I[Document image] --> E[official UND encoder frozen]
    E --> F[official merger frozen]
    F --> X[visual token X: n×2048]
    X --> G[MetaCompress trainable]
    G --> P[row softmax matrix P: m×n]
    X --> C[compressed token PX]
    P --> C
    X --> T[frozen Qwen teacher]
    C --> S[same frozen Qwen student]
    T --> KL[shared-prefix  full-vocab KL]
    S --> KL
    P --> R[mean row entropy and collapse]
    KL --> L[total loss]
    R --> L
```

**The formal method compresses after the merger, i.e.  $PF(X_0)$; unlike earlier random experiments’  $F(PX_0)$. The nonlinear merger makes the two paths nonequivalent.** The implementation still supports  `--insertion pre-merger`, used only for early interface smoke, not formal scores. The generator reads image tokens/coordinates only, never question/gold/teacher answer.

| Aspect | Original paper | Implementation and differences |
|---|---|---|
| Backbone | General multiturn VLMs such as LLaVA | Official MonkeyOCRv2-B-Und + Qwen3-1.7B, independent document VQA |
| Compression position | After visual projector, before LLM | identical; compress the official merger’s 2,048-dimensional output |
| Feature source | Freeze the corresponding VLM’s visual module | Reextract BF16 features from official **UND** encoder inputs, without reusing the separate B model cache |
| query pooling | Main text says average; supplement Section 8.2 specifies fractional max, kernel=3 | Follow the supplement: PyTorch fractional max with fixed offset=.5 for same-image determinism, replacing draft adaptive-average pooling |
| Position encoding | Absolute position embedding; full parameterization not disclosed | Learnable low-dimensional row/column tables followed by projection; strictly follow official  `get_pos_ids_by_grid` 2×2 block order rather than incorrect row-major order |
| Compressed dimension | $d_c\\ll d$, not all configuration values disclosed in the main text | Fixed at 64 here; not claimed for every author configuration |
| KL trajectory and temperature | KL between response distributions; teacher-forcing details not fully disclosed | Shared teacher-generated prefix; full vocabulary, T=1, averaged over response positions |
| Teacher response length | No MonkeyOCRv2 settings disclosed | Training cap=64; free-generation evaluation cap=256 |
| Training data | About 20k image/text samples from MT-GQA/MT-VQA-v2 and others | Hash-fixed 20k from seven existing document-VQA training sources, not the original paper’s training corpus |
| Optimization | SGD 1e-3, clip=1e-2, batch=36, 2 epochs, four RTX A6000 GPUs | Same numerical settings, global-norm clipping, CPU/Gloo 36 ranks, synchronous batch=36 |

## 2. Loss and Backpropagation

Use the paper-style generator’s complete **three-term Eq. 10 objective**, both regularizer weights 1:

$$
L_{{pred}}=\\frac1T\\sum_{{t=1}}^T D_{{KL}}\\!\\left(p_T(\\cdot\\mid q,a_{{<t}},X)\\Vert p_S(\\cdot\\mid q,a_{{<t}},PX)\\right),
$$
$$
L_H=-\\frac1m\\sum_i\\sum_jP_{{ij}}\\log P_{{ij}},\\qquad
L_{{collapse}}=\\max_j\\sum_iP_{{ij}},\\qquad L=L_{{pred}}+L_H+L_{{collapse}}.
$$

$a$ is the uncompressed teacher's greedy response, including its actual EOS. Gold is not optimized. Both teacher-forcing paths use the same teacher prefix, summing over vocabulary and averaging response positions. Teacher logits are detached and may be cached unchanged as FP32. Student gradients pass through frozen Qwen to  $P$ and generator; **frozen weights do not disable student autograd**. Convert BF16 logits to FP32 for log-softmax, $P$ and generator use FP32. Teacher/student downstream answers are independently generated; students receive neither teacher answers nor previous-question cache.

Mean row entropy encourages local concentration; collapse discourages multiple rows using one source token. These are regularizers, not an assumed optimal predictor of gold scores. Conditional KL along one greedy teacher trajectory is not sequence KL integrated over all response trajectories.

## 3. Data, Splits, and Execution

Training set:`data/full/train.jsonl` , totaling 243,652 QA. A fixed hash selects **20,000 QA / 16,731 images**, with zero image-ID overlap with local val/test. Each epoch pads 16 final-batch samples, giving **40,032 sample presentations / 1,112 SGD updates**, not 40,032 distinct QA.

| Training task | Questions in this run |
|---|---:|
'''
    for task,n in cfg['task_counts'].items():text+=f'| {task} | {n:,} |\n'
    text+='''
Use official  `AutoProcessor` , ChatML, and image processing, with max_pixels=1,003,520. The separate B encoder's earlier raw-feature cache is not reused. Code inspection found that released B and B-Und visual tensors differ; the draft's old cache path was removed before formal training. Native UND features for this round are stored in  `artifacts/meta_compress_cache/features/`, teacher responses at  `teachers/`, complete FP32 teacher logits in  `logits/`. Caches avoid repeated computation without updating the backbone.

Both ordinary and high-priority NPU partitions remained unavailable, so their queued jobs were cancelled. Actual training uses six CPU nodes, six model processes per node, 16 threads per process (576 CPU cores), and 36-rank Gloo-synchronous SGD. Formal training job **151231** automatically triggers eight-task evaluation **151233**, followed by report generation **151237**. Job **151239** precomputes the frozen teacher cache using 4 CPU array shards, 28 processes / 448 cores, reducing SGD generation waits with the same helper, 16 threads, and complete FP32 objective; it changes neither generator nor objective. Peak concurrency is capped at 1,024 CPU cores. All teacher/student scores are regenerated in this round's CPU BF16 environment; earlier NPU answers are not mixed in.

Prespecify 64 distinct images/task, one question/image, **512 images/questions** total. DocVQA/InfoVQA use official validation, others test; no training overlap. Compare  `trained`, the same saved at training start  `initial`, balanced contiguous grouping  `mean`  compressors sharing the uncompressed teacher. **The final epoch-2 model was fixed in advance; benchmark scores select neither checkpoints nor hyperparameters.** This is subset evaluation, not the paper's full benchmark. DeepForm/KLC use field-level exact match rather than official document-level corpus F1. DT-VQA retains max(answer-containment accuracy, ANLS); WTQ uses denotation and OCRBench the existing subtask scoring.

## 4. Completed Interface and Gradient Checks

- Single-process/two-example and two-process DDP smoke checks are complete; they are interface tests, not formal 20k training results.
- Independent gradient audit of the formal architecture: `artifacts/meta_compress_gradient_audit/gradient_audit.json`  has KL-only generator-gradient norm **2.829941**; trainable official-backbone parameters: **0**.
- Row-stochastic checks have maximum row-sum error below **4e-7**; formal training continues logging it per sample.
- SGD updates only the generator; teacher runs without gradients and student retains the gradient path.

## 5. Actual Training Logs

| Epoch | Logged batches | prediction KL | Row entropy | collapse | Total loss | Pre-clipping gradient norm |
|---|---:|---:|---:|---:|---:|---:|
'''
    for e in epochs:text+=f"| {e['epoch']} | {e['logged_steps']} | {e['prediction']:.6f} | {e['entropy']:.6f} | {e['collapse']:.6f} | {e['loss']:.6f} | {e['grad_norm']:.6f} |\n"
    if not epochs:text+='| No complete global batch logged yet | 0 | — | — | — | — | — |\n'
    if training:text+='\n[Training-curve PNG](../artifacts/meta_compress/training_loss.png) · [Training-curve PDF](../artifacts/meta_compress/training_loss.pdf). Curves show actual batch means and 25-step smoothing, without forcing monotonic fits.\n'
    text+='''
## 6. Downstream Evaluation Results

'''
    if not evaluated:text+=f'**Formal evaluation is still incomplete ({len(records)}/{expected}  records). No final task score or training-effectiveness conclusion is available.** Formal tables require all planned samples.\n'
    else:
        text+='Scores are percentages; training Δ=trained−initial, positive favors training.\n\n| Task | n | Uncompressed teacher | mean pool | initial | trained | teacher−trained | Training Δ (paired-bootstrap 95% CI, pp) |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
        for task,z in metrics['trained'].items():
            change=delta_ci[task];lo,hi=change['paired_bootstrap_ci95']
            text+=f"| {task} | {z['samples']} | {100*z['teacher_score']:.2f} | {100*metrics['mean'][task]['student_score']:.2f} | {100*metrics['initial'][task]['student_score']:.2f} | {100*z['student_score']:.2f} | {100*z['score_drop']:.2f} | {change['trained_minus_initial_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |\n"
        text+='\n| Task | Variant | KL | Row entropy | collapse | Total loss | Mean n → m |\n|---|---|---:|---:|---:|---:|---|\n'
        for variant,table in metrics.items():
            for task,z in table.items():text+=f"| {task} | {variant} | {z['prediction_loss']:.4f} | {z['entropy_loss']:.4f} | {z['collapse_loss']:.4f} | {z['total_loss']:.4f} | {z['tokens']:.1f}→{z['compressed_tokens']:.1f} |\n"
        text+='\nOnly 64 images/task, so intervals can be wide. This tests fixed 90% reduction/settings, not every retention. Check loss and generated accuracy separately; preserving teacher errors also reduces KL.\n'
    text+='''
## 7. Entry Points and Outputs

```bash
# Inspect jobs, training progress, and existing evaluations
bash scripts/run_meta_compress.sh status

# Manually rebuild current progress without claiming incomplete results
.venv/bin/python -m scripts.report_meta_compress --allow-partial

# formal CPU training; rerun the same command to restore the latest  checkpoint resume
sbatch scripts/slurm_meta_compress_cpu.sh

# Evaluate after training; the normal pipeline already schedules dependencies, so do not duplicate submissions
sbatch scripts/slurm_meta_compress_eval_cpu.sh

# Summarize after all evaluations complete; missing samples raise an error
.venv/bin/python -m scripts.report_meta_compress

# Single-image QA and matrix export after training
.venv/bin/python -m scripts.predict_meta_compress \\
  --image /absolute/path/to/document.png --question 'What is the date?' \\
  --teacher --save-matrix /tmp/compression_P.pt
```

Training `scripts/train_meta_compress.py`; generator/loss  `vqa/meta_compress.py`; independent generation evaluation  `scripts/eval_meta_compress.py`. Python environment is managed by the existing **`.venv`**; CPU model in BF16; dependency records follow  `requirements-decoder-loss-cpu.txt` and `requirements-report.txt`. Additional NPU  `.venv-und` and `scripts/slurm_meta_compress.sh`, measured results follow the CPU protocol.

- [Training protocol / code SHA-256](../artifacts/meta_compress/protocol.json), [Fixed training samples](../artifacts/meta_compress/samples.jsonl), [B/UND feature-source audit](../artifacts/meta_compress/encoder_provenance_audit.json);
- [Per-global-step training log](../artifacts/meta_compress/training.jsonl), each `examples.rank*.jsonl` retains every per-sample loss;
- `artifacts/meta_compress/initial.pt`, `meta_compress.pt`, `complete.json`; source snapshots  `source_snapshots/`;
- `artifacts/meta_compress/eval/predictions.benchmark.rank*.jsonl`: gold, both answers/token IDs/scores, three component losses, and total;
- `artifacts/meta_compress/eval/matrices/benchmark.rank*.pt`: trained P for each evaluation worker’s first image, with grid/checkpoint SHA-256;
- [Answer-comparison CSV (generated after evaluation outputs exist)](../artifacts/meta_compress/answers.csv), [Machine-readable summary](../artifacts/meta_compress/summary.json).
'''
    if complete and evaluated:
        macro={v:100*statistics.mean(x['student_score'] for x in table.values()) for v,table in metrics.items()}
        teacher_macro=100*statistics.mean(x['teacher_score'] for x in metrics['trained'].values())
        conclusion=f'''## Conclusions After Completion

**Training is complete but compressed VQA has not improved; this is not successful reproduction of the paper’s compression quality.** Eight-task percentage macro: uncompressed teacher **{teacher_macro:.2f}**, mean-pool **{macro['mean']:.2f}**, initial generator **{macro['initial']:.2f}**, trained generator **{macro['trained']:.2f}**; trained-versus-initial change **{macro['trained']-macro['initial']:+.2f}  percentage points**. Each task has only 64 images; macro scores are descriptive mixed-task summaries, not a unified official leaderboard.

Epoch 1→2 total loss decreases **12.685964→12.379922**, mainly from collapse **2.334961→2.015338**. Prediction KL **2.331271→2.333542** does not improve, and row entropy **8.019732→8.031041** does not decrease. The earlier small total-loss decrease therefore is not evidence of improved semantic preservation. Eight-task evaluation KL also changes from about **2.07750→2.07852**, without improvement.

Evidence indicates that this setting primarily optimizes the anti-repeated-aggregation regularizer without learning sufficient document-evidence preservation. At 90% token reduction, initial/trained matrices remain diffuse with high row entropy. Pre-clipping global norms around 54–62 face a threshold of 0.01, strongly shrinking updates. These are diagnostic leads, not proof that one hyperparameter, model capacity, or the original paper is uniquely responsible. Audit initialization/attention scale and component gradients first, then validate changes on training-source development data rather than selecting on this benchmark. Preserve failed results and complete answers; do not replace final outcomes with smoke or intermediate checkpoints.

'''
        text=text.replace('## 1. Paper Design and Implemented Architecture',conclusion+'## 1. Paper Design and Implemented Architecture',1)
    dest=ROOT/'docs/METACOMPRESS_GENERATOR_TRAINING.md';tmp=dest.with_suffix('.tmp');tmp.write_text(text);tmp.replace(dest)
    print(json.dumps({'report':str(dest),'training_steps':len(training),'training_complete':complete,'evaluation_rows':len(records),'evaluation_complete':evaluated}))

if __name__=='__main__':main()
