"""Render the completed encoder-only experiment, without selecting a winning loss."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/token_proxy/encoder_candidates_v1'

def main():
    a=json.loads((OUT/'analysis.json').read_text());assert a['has_full_token_OT'] and a['has_line_probe']
    m=a['metrics'];p=json.loads((OUT/'probe_metrics.json').read_text());l=json.loads((OUT/'line_probe_metrics.json').read_text());d=a['full_token_transport_diagnostics'];sp=json.loads((OUT/'subspace_audit.json').read_text())['summary']
    def f(x):return '—' if x is None else f'{x:.4f}'
    def ci(x,percent=False):return '['+', '.join(f'{v*100:.2f}%' if percent else f'{v:.4f}' for v in x)+']'
    names=[('Question-weighted coverage + position','coverage_pos_question'),('Full-token Sinkhorn / uniform','sinkhorn_full_uniform'),('Full-token Sinkhorn / question-weighted excess','sinkhorn_full_question_excess'),('Text + numeric + coarse adjacency evidence','evidence_combined_question'),('Text + numeric + OCR-line relations','evidence_combined_line_question'),('Question-conditioned subspace reconstruction','subspace_question')]
    mainrows='\n'.join(f"| {label} | {f(m[n]['pooled_spearman'])} | {f(m[n]['within_question_ratio_family'])} {ci(m[n]['within_question_ratio_family_ci95'])} | {m[n]['paired_seed_ordering']['accuracy']*100:.2f}% {ci(m[n]['paired_seed_ordering']['image_bootstrap_ci95'],True)} |" for label,n in names)
    taskrows=[]
    for t,v in a['per_task'].items():
        taskrows.append(f"| {t} | {v['images']} | {24*v['images']:,} | {v['teacher_mean']*100:.2f} | "+' | '.join(f(m[n]['per_task'][t]['pooled_spearman']) for n in ['coverage_pos_question','sinkhorn_full_question_excess','evidence_combined_line_question','subspace_question'])+' |')
    conditionrows='\n'.join('| '+t+' | '+' | '.join(f(m[n]['per_task'][t]['within_question_config']) for n in ['coverage_pos_question','sinkhorn_full_question_excess','evidence_combined_line_question','subspace_question'])+' |' for t in a['per_task'])
    allrows='\n'.join(f"| `{n}` | {f(v['pooled_spearman'])} | {f(v['within_question'])} | {f(v['within_question_ratio_family'])} | {v['paired_seed_ordering']['accuracy']*100:.2f}% |" for n,v in m.items())
    subsetrows='\n'.join(f"| {label} | {f(m[n]['teacher_positive_spearman'])} | {f(m[n]['teacher_perfect_spearman'])} | {f(m[n]['high_retention']['pooled_spearman'])} | {f(m[n]['high_retention']['within_question_config'])} |" for label,n in names)
    configrows='\n'.join('| '+key+' | '+' | '.join(f(m[n]['per_ratio_family'][key]['pooled_spearman']) for n in ['coverage_pos_question','sinkhorn_full_question_excess','evidence_combined_line_question','subspace_question'])+' |' for key in m['coverage_pos_question']['per_ratio_family'])
    apr='\n'.join(f"| {key} | {value:.4f} |" for key,value in p['type_AP'].items())
    pair=m['coverage_pos_question']['paired_seed_ordering'];zero=a['student_zero_scores']
    counter=json.loads((OUT/'counterexamples.json').read_text())
    def escape(value):return str(value).replace('|','\\|').replace('\n',' ')[:180]
    examples=[]
    for name,pairs in counter.items():
        if not pairs:continue
        ex=pairs[0];lo=ex['low_loss'];hi=ex['high_loss']
        examples.append(f"**`{name}`**: {escape(lo['sample_id'])}, {lo['family']}, retention  {lo['ratio']}. Question: {escape(lo['question'])}. Gold: {escape(lo['answers'])}; teacher: {escape(lo['teacher_prediction'])}({lo['teacher_score']:.4f}). \n\n| Trial | loss | Student answer (at most 180 characters) | Student score | Score drop |\n|---|---:|---|---:|---:|\n"+'\n'.join(f"| {'Low loss' if r is lo else 'High loss'} / seed {r['replicate']} | {r['loss'][name]:.7g} | {escape(r['student_prediction'])} | {r['student_score']:.4f} | {r['score_drop']:.4f} |" for r in [lo,hi]))
    ot=m['sinkhorn_full_question_excess'];ln=m['evidence_combined_line_question']
    sensitivity=json.loads((OUT/'statistical_sensitivity.json').read_text())
    sens=sensitivity['both_trial_max_marginal_error_le_1e_6'];sd=sensitivity['all']
    text=rf'''# Strict Encoder-Only Loss Validation for Document VQA Compression

**Status: completed.** Official MonkeyOCRv2-B-Und; across eight tasks: **3,601 distinct images, 3,601 questions, 86,424 random compressions, and 7,202 mean-pooling controls**. Original answers, gold references, teacher/student scores, and all new losses are retained separately for every compression.

**Main finding: no new loss reliably identifies the better random matrix for the same question and compression budget.** Pooled correlations across questions/configurations can be positive, but conditional correlations and paired seeds must also be examined. Full-token question-weighted Sinkhorn excess: pooled Spearman **{ot['pooled_spearman']:.4f}**, within-question/retention/family correlation **{ot['within_question_ratio_family']:.4f}**, non-tied ordering accuracy **{ot['paired_seed_ordering']['accuracy']*100:.2f}%**. The text + numeric + OCR-line combination gives **{ln['pooled_spearman']:.4f} / {ln['within_question_ratio_family']:.4f} / {ln['paired_seed_ordering']['accuracy']*100:.2f}%**. CIs appear in the primary table; a slightly higher pooled correlation does not select a new training objective.

**Full question-weighted OT shows a weak positive signal; the new candidates are not all indistinguishable from random ordering.** Its unadjusted 95% ordering CI is slightly above 50%, but the advantage is only about 1.63 percentage points and does not demonstrate superiority to the previous reconstruction baseline. Strong correlation or a practical guarantee of score preservation has not been established. No intervals are adjusted for multiple comparisons, and the 24 fixed fields are not independent tests.

**Entry point:** [`bash scripts/run_encoder_candidates.sh status`](../scripts/run_encoder_candidates.sh); [Per-trial CSV](../artifacts/token_proxy/encoder_candidates_v1/trials.csv), [Complete statistics JSON](../artifacts/token_proxy/encoder_candidates_v1/analysis.json), [Reverse-ordering answer examples](../artifacts/token_proxy/encoder_candidates_v1/counterexamples.json), [Filterable original per-trial answers](../artifacts/token_proxy/random_v1_20x/review.html).

## 1. Work Performed and Excluded Variants

| Literature proposal | Implementation evaluated here | Compliance with the strict constraint |
|---|---|---|
| Question-evidence-weighted coverage / Sinkhorn (A / A′) | Independent OCR character/bigram head bridges questions; sampled raw-encoder coverage and full-token Sinkhorn divergence | Retain A; exclude A′ variants in official projector/Qwen embedding space |
| Non-generative text/numeric/relation evidence preservation (A) | JS on character presence, hashed bigrams, and numeric/format types; coarse text–number adjacency; an additional same-OCR-line probe | Read raw X or PX only; no exact-string, arithmetic, or header–value semantic recognizer |
| Question-conditioned subspace reconstruction (A′ baseline) | Use ridge subspace reconstruction on all 768 raw-encoder dimensions, weighting teacher tokens with an independent lexical bridge | Modified to avoid projector/Qwen embeddings; evaluated normally, but not a formula-by-formula reproduction in the original interface space |

Explicitly excluded: (1) distances on official merger/alignment-MLP outputs; (2) question queries built from Qwen token embeddings; (3) evidence heads supervised by LLM attention, logits, NLL, or VQA answer scores. These depend on the excluded decoder/interface and violate this experiment's constraints. **A method being described as training-free does not make it decoder-agnostic.**

Computational graph:

```text
frozen official Und encoder → X[N,768] ──────→ official merger/LLM → teacher answer (cached)
                         ├─ P → C[M,768] → official merger/LLM → student answer (cached)
                         └─ X, C, U, PU, question → independent OCR/line probe + encoder-only mathematical distances → loss
```

The loss branch does not read official merger/LLM tensors on the right. Before extracting auxiliary-probe features, replace the official  `vision_tower.merger` with Identity before loading remaining vision weights, without loading Qwen. [Implementation](../scripts/cache_encoder_probe.py). The separate 768→192→120 evidence MLP and 578→64→1 line MLP **are small OCR-supervised probes, not the official encoder–LLM alignment layer**. Their parameters/outputs never enter the downstream LLM.

Computationally decoder-free and independent of interface weights, but validated on one UND decoder only; **cross-decoder statistical predictive ability is not established**. Training a VQA-improving compressor is outside this study.

## 2. Matching Data, Random Matrices, and Per-Trial Outcomes

Following the [20×-scale experiment](TOKEN_PROXY_20X_RESULTS.md) uses identical images, questions, preprocessing, features, and matrices. **Teacher/student answers/scores reuse prior caches; only candidate losses are recomputed, not 93,626 generations.** Matched intervention replay avoids generation noise.

- Model: `models/MonkeyOCRv2-B-Und`, official complete weights without LoRA. Raw encoder outputs are BF16, 768-dimensional, max_pixels=1,003,520.
- One question/image; retention $r=M/N\in\{{0.125,0.25,0.375,0.5,0.625,0.875\}}$. Two families, local/global, with two seeds each give 24 random trials/image. Two mean-pooling controls/image at 0.25/0.5 are excluded from primary correlation statistics.
- Unmodified [`vqa/token_proxy.py`](../vqa/token_proxy.py). Each token belongs to one balanced support; support sizes differ by at most 1. Apply row softmax to uniform[-2,2] logits, mix with 30% uniform weights, and add a $10^{{-6}}$  global  floor. All entries are strictly positive with row sums  1, $C=PX$ is convex aggregation; equivalently $P=\operatorname{{softmax}}(\log P)$. This is not unconstrained dense Gaussian mixing.
- Each trial independently retains  `matrix_seed / matrix_file / teacher_prediction / student_prediction / answers / teacher_score / student_score / score_drop`. **Seed-level ANLS or other downstream scores are not averaged before correlation.**
- New workers verify all matrix and raw-X SHA-256 values, and every  $PX$ after BF16 conversion matches the actual downstream compressed tensor. Regenerate each image’s first random P independently and verify identical groups/weights. Loss uses FP32  $PX$; downstream uses its BF16 version.
- Existing interface audits show equal native/cached/identity logits and generation, and teacher agreement with official baseline on all 3,601/3,601 questions. [Existing audit](../artifacts/token_proxy/random_v1_20x/audit.json). Original scorers remain: DeepForm/KLC normalized per-question exact match, not corpus F1. All scores lie in [0,1]; pooled cross-task correlation is diagnostic only.

| Task | Distinct images / questions | Random trials | Subset teacher score ×100 | coverage ρ | Full OT excess ρ | Evidence + line relations ρ | subspace ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(taskrows)}

Teacher values describe this subset, not reevaluation of the paper’s full benchmark.

## 3. Implemented Loss Definitions

### 3.1 Decoder-Independent Question Bridge and Feature Coordinates

Train independent non-autoregressive probes using InfoVQA **training-set** OCR WORD boxes/text (confidence≥70): 720 training images and 240 image-disjoint validation images, with zero image-hash overlap with project VQA val/test. No VQA gold, answer scores, or official LLM representations supervise the probes. Train for 20 fixed epochs and use the last one, without checkpoint selection by downstream correlation.

$H(X)$ outputs multilabel probabilities for lowercase character presence, 64 hashed bigrams, and 8 types (text/digit/decimal/percent/currency/negative/date/alphabet). Remove question stopwords and map into the same explicit character/bigram coordinates, cosine max-similarity to $H(X_i)$ gives $s_i(q)$, with fixed temperature 0.1:

$$a_i(q)=0.3/N+0.7\operatorname{{softmax}}_i(s_i(q)/0.1).$$

This is a **lexical bridge**, not complete semantic question understanding; answer evidence may share no question text. OCR validation on 240 images checks probes, not VQA evidence grounding.

Distances use a fixed random orthogonal projection $R\in\mathbb{{R}}^{{768\times64}}$ (seed=20261022), $h(x)=xR/(s\sqrt{{64}})$, $s={p['rms']:.6f}$ is the training-set scalar RMS. Teacher and student use the same scale; **the student is not normalized independently**. The projection is independent of official alignment weights. $U$ contains normalized 2D patch coordinates; compressed coordinates are $PU$.

### 3.2 Question-Weighted Coverage

Uniformly sample at most 256 fixed teacher anchors/image; find nearest neighbors among all student tokens:

$$L_{{cov}}=\sum_{{i\in A}}\tilde a_i\min_j\big[\|h(X_i)-h(C_j)\|^2+0.25\|U_i-(PU)_j\|^2\big].$$

Retain four fixed uniform/question variants with/without positions. $\tilde a$ is renormalized within sampled anchors. **This is sampled, not exact full-teacher coverage**. Concentrated question weights can miss evidence, so it is not an unlimited-precision version of the proposal.

### 3.3 Full-Token Sinkhorn

Full $N\times M$ token costs follow the same formula; target mass is fixed  $b_j=1/M$. Entropy regularizer  $\epsilon=0.05$, objective $\langle\pi,D\rangle+\epsilon\mathrm{{KL}}(\pi\|ab^T)$; use debiased Sinkhorn divergence:

$$S_\epsilon(\mu,\nu)=OT_\epsilon(\mu,\nu)-\tfrac12 OT_\epsilon(\mu,\mu)-\tfrac12 OT_\epsilon(\nu,\nu).$$

Two source masses: uniform or  $a(q)$. **Question-weighted source and uniform target marginals differ, even if  $P=I$, raw divergence may also be nonzero.** Retain raw values and report separately:

$$L_{{OT,q,excess}}=S_\epsilon((X,a(q)),(C,\mathrm{{unif}}))-S_\epsilon((X,a(q)),(X,\mathrm{{unif}})).$$

Excess over the uncompressed reference can be negative, **not a guaranteed nonnegative distance**. Subtracting a questionwise constant preserves within-question matrix ordering but can change pooled correlation.

Final primary results use **full-token** transport without spatial-centroid merging; the 64-dimensional feature projection remains. The 512-centroid approximation is retained only as a computational ablation, not a substitute for final OT conclusions. NPU FP32 uses stable scaling with log-domain fallback, 150–1,500 iterations, and target marginal tolerance $10^{{-6}}$; actual errors are reported in Section 6.

### 3.4 Non-Generative Text, Numeric, and Relational Evidence Preservation

At a fixed 16×8 spatial query grid, independently read the nearest 8 tokens from X/U and C/PU with Gaussian bandwidth=.04. Far-from-token queries fall back toward  $H(0)$. Compare the two probe outputs using Bernoulli JS divergence:

- `evidence_text_*`: character+bigram probability differences.
- `evidence_number_*`: probability differences for digit, decimal, percent, currency, negative, and date.
- `evidence_relation_question`: four text/number binding-state differences in neighboring query regions; **coarse adjacency**, not learned table semantics.
- `evidence_combined_question`: fixed equal-weight sum of the three terms.

Additional structural probe: on the same 720/240-image split, train a 37,121-parameter same-line head using WORD→LINE annotations. Inputs are two OCR-word ROIs' 192-dimensional raw-X features from the frozen evidence head's first layer, plus absolute feature and 2D position differences. At validation, predict same-line probabilities for neighboring regions of one fixed spatial grid and compare teacher/student JS, obtaining  `evidence_line_uniform/question`. `evidence_combined_line_question` is the equal-weight text+number+line sum. **This is an added experiment, not claimed fully preregistered. VQA scores select neither weights nor final candidates.**

Limitations: character-presence/bigram hashing **cannot reconstruct character order or exact numerical strings**. No header–value semantic relations, arithmetic relations, or non-autoregressive exact OCR head are implemented. Negative results apply only to **these specific probes** on this distribution; they do not invalidate every exact-text/numeric/relation-preservation method. OCR word/token supervision also differs from compressed mixed tokens and fixed-grid readout at evaluation.

### 3.5 Question-Conditioned Subspace Reconstruction

Use the raw encoder’s full **768-dimensional** space, without 64-dimensional feature reduction for subspace reconstruction. Let  $\lambda=0.001s^2$, use Cholesky for the ridge-projection residual:

$$E_i=\lambda X_i(C^TC/M+\lambda I)^{{-1}},\qquad L_{{sub,q}}=\frac{{\sum_{{i\in A}}\tilde a_i\|E_i\|^2}}{{\sum_{{i\in A}}\tilde a_i\|X_i\|^2}}.$$

Risk evaluation retains the same at most 256 teacher anchors; covariance uses all C. Retain uniform, question-weighted, and excess terms subtracting the uncompressed ridge reference. This baseline needs no LLM forward, but 768×768 Cholesky still has a computational cost.

It asks whether student span can reconstruct teacher, with coefficients neither required to equal P nor constrained to be convex. Mixed C can preserve span while breaking local reading/decoder usability. Low residual therefore does not establish sufficient VQA information.

Additional [FP64 spectral/coefficient audit on 64 compressions of one hash-selected image per task](../artifacts/token_proxy/encoder_candidates_v1/subspace_audit.json): rank range at relative threshold 1e-4  {sp['rank_at_relative_1e_4']['min']:.0f}–{sp['rank_at_relative_1e_4']['max']:.0f}; ridge-covariance condition-number range  {sp['ridge_cov_condition']['min']:.3g}–{sp['ridge_cov_condition']['max']:.3g}. Median fraction of negative reconstruction coefficients={sp['coefficient_negative_fraction']['median']*100:.2f}%; median across trials of the coefficient absolute-row-sum 95th percentile = {sp['coefficient_abs_row_sum_p95']['median']:.2f}, max={sp['coefficient_abs_row_sum_p95']['max']:.2f}. This differs from P’s nonnegative convexity constraint. Maximum FP32-versus-FP64 relative error for the uniform residual is  {sp['float32_uniform_relative_error']['max']*100:.4f}%, with no clear numerical distortion at these 64 audited points. This spot check does not establish precision for all 93,626 solves.

## 4. Do Independent Auxiliary Heads Learn Useful Signals?

| Independent validation metric | Numerical value |
|---|---:|
| text AP | {p['text_AP']:.4f} |
| digit AP | {p['digit_AP']:.4f} |
| Character macro AP (text tokens) | {p['character_macro_AP_text_tokens']:.4f} |
| bigram macro AP(text token) | {p['bigram_macro_AP_text_tokens']:.4f} |
| OCR word-query Top-10 hit rate | {p['OCR_word_query_localization']['top10_hit_rate']*100:.2f}% |
| Same-query uniform-approximation Top-10 hit rate | {p['OCR_word_query_localization']['uniform_approx_top10_hit_rate']*100:.2f}% |
| same-line AP / accuracy | {l['same_line_AP']:.4f} / {l['same_line_accuracy']:.4f} |
| **geometry-only same-line AP** | **{l['geometry_only_AP']:.4f}** |

OCR head: 170,808 parameters, 540,622 training tokens / 181,207 held-out tokens. Positive and negative tokens are sampled, so AP describes that sampling distribution, not natural full-image frequencies. OCR-query checks use 950 word queries on 80 held-out images, not VQA evidence-localization accuracy. The line head uses 167,820 training pairs / 56,620 held-out pairs. Geometric line rules have **higher** AP than the trained head; strong learned semantic relations have not been demonstrated.

| OCR type | AP |
|---|---:|
{apr}

[Per-epoch OCR training/validation metrics](../artifacts/token_proxy/encoder_candidates_v1/probe_metrics.json), [Training-curve PDF](../artifacts/token_proxy/encoder_candidates_v1/probe_training_loss.pdf), [Per-epoch line-head metrics](../artifacts/token_proxy/encoder_candidates_v1/line_probe_metrics.json), [OCR weights](../artifacts/token_proxy/encoder_candidates_v1/evidence_head.pt), [Line-head weights](../artifacts/token_proxy/encoder_candidates_v1/line_head.pt). Auxiliary training constructs the loss only; compressor, official projector, and LLM are not trained.

## 5. Correlation-Validation Results

Define $\Delta=score_{{teacher}}-score_{{student}}$, ; retain negative values. Higher loss should correspond to larger  loss increases, $\Delta$ increases. Pooled Spearman uses **86,424 individually retained trials**. Within-question/retention/family metrics rank and center each two-seed group before aggregation. They measure ordering at a fixed compression budget and must not be conflated with pooled correlation across configurations.

All confidence intervals use **1,000 image-cluster bootstrap resamples**, retaining each image's trials together and reranking pooled Spearman each time. Paired seeds are never averaged: only agreement between high/low loss and high/low score drop is checked. Conditional correlation uses within-group ranks; two seeds mainly test direction, not a finely estimated conditional relationship.

| Evaluated method | pooled Spearman | Within-question/retention/family correlation [95% CI] | Non-tied seed-ordering accuracy [95% CI] |
|---|---:|---|---|
{mainrows}

Total **{pair['pairs']:,}** paired two-seed groups, of which **{pair['outcome_ties']:,}** groups tie on downstream score; only **{pair['pairs']-pair['outcome_ties']:,}** groups are non-tied. Student zero-score rate among random trials: **{zero:,}/{a['random_trials']:,}({zero/a['random_trials']*100:.2f}%)**. This random family therefore has a severe floor effect: many trials cannot be ranked by answer score. A CI including 50% does not establish better-than-random ordering.

**Weak signals and numerical-error sensitivity:** retaining pairs for which both trials have maximum OT marginal error ≤1e-6 leaves  {sens['nontie_pairs']:,}  non-tied pairs; question-weighted OT ordering accuracy is  {sens['OT_order_accuracy']*100:.2f}%, 95% CI {ci(sens['OT_accuracy_image_bootstrap_ci95'],True)}. In all matched pairs, OT relative to the earlier  `V1_rec` accuracy difference is  {sd['OT_minus_old_rec_accuracy']*100:.3f}  percentage points; paired image-bootstrap 95% CI [{sd['paired_image_bootstrap_difference_ci95'][0]*100:.3f}, {sd['paired_image_bootstrap_difference_ci95'][1]*100:.3f}] percentage points. Improvement over the old reconstruction term is not established. [Complete supplementary statistics](../artifacts/token_proxy/encoder_candidates_v1/statistical_sensitivity.json). This sensitivity analysis neither replaces primary results nor selects candidates.

### Within-Question, Retention, and Family Correlations by Task

| Task | coverage | Full OT excess | Evidence + line relations | subspace |
|---|---:|---:|---:|---:|
{conditionrows}

### Teacher-Solvable Subsets and Higher Retention Fractions

| Method | teacher>0 pooled ρ | teacher=1 pooled ρ | r≥0.625 pooled ρ | Conditional correlation at r≥0.625 |
|---|---:|---:|---:|---:|
{subsetrows}

Teacher>0 includes **{a['teacher_positive_images']:,}** images. Filtering reduces the confound of an already incorrect teacher leaving no visible score drop, but does not solve same-question/same-budget random-matrix ordering.

### Counterexamples from Actual Answers

Examples are selected by reverse-order score drop to illustrate failure mechanisms, **not estimate their unbiased frequency**. Each pair shares image, question, retention, and matrix family; complete answers/seeds and more examples are in counterexamples.json.

{chr(10).join(chr(10)+ex+chr(10) for ex in examples)}

### Pooled Spearman at Fixed Retention and Matrix Family

Questions are still pooled here; this does not replace same-question conditional tests.

| r / family | coverage | Full OT excess | Evidence + line relations | subspace |
|---|---:|---:|---:|---:|
{configrows}

### All Fixed Variants and Historical Baselines

| Loss field | pooled ρ | Within-question correlation across configurations | Within-question, retention, and family correlation | Non-tied seed ordering |
|---|---:|---:|---:|---:|
{allrows}

`V1`, `V1_rec` is a same-trial comparison with the earlier loss. Reconstruction may show weak positive ordering; failure to validate new candidates does not make every baseline exactly random.`sinkhorn_*` None `full` prefixed variants use 512 spatial centroids as an ablation only. Mean-pooling correlations, two-factor rank residuals, per-task seed-ordering counts, and loss quantiles are in [analysis.json](../artifacts/token_proxy/encoder_candidates_v1/analysis.json); controls do not select new combinations.

![Per-trial loss versus answer-score drop](../artifacts/token_proxy/encoder_candidates_v1/correlation_scatter.png)

[Scatter PDF](../artifacts/token_proxy/encoder_candidates_v1/correlation_scatter.pdf). Scatter plots use a fixed random sample of at most 16,000 trials for visualization only; statistics use all random trials. Different x-axis scales prevent comparing losses by visual slope.

## 6. Numerical, Interface, and Precision Audits

- [Mathematical checks](../artifacts/token_proxy/encoder_candidates_v1/math_checks.json): objective discrepancy between tiny entropic OT and an independent SciPy constrained optimizer $7.76\times10^{{-9}}$; symmetry, identity, and joint C/PU row-permutation checks pass. Nonzero identity for question-raw OT follows from its marginal definition; excess identity is zero.
- Compare 512 / 2,048 spatial centroids against full-token OT on one hash-selected image per task and 64 fixed trials across 8 tasks: [Resolution audit](../artifacts/token_proxy/encoder_candidates_v1/transport_resolution_audit.json) . Correlations between 512-token and full-token uniform/question-excess orderings are about 0.851/0.861, but relative MAE is about 52%; even at 2,048 tokens it is about 36%–39%. Therefore **full-token OT was added for all 93,626 trials**, and the primary table does not use a coarse spatial approximation.
- [CPU float64 / NPU float32 cross-check](../artifacts/token_proxy/encoder_candidates_v1/npu_transport_check.json): maximum loss difference across the same 64 full-token trials $8.94\times10^{{-8}}$.
- Per-trial maximum marginal error for full-token OT: median={d['max_marginal_error']['median']:.3g}, max={d['max_marginal_error']['max']:.3g}; **{d['trials_above_1e_6']:,}/{d['trial_count']:,}** trials have maximum error above $10^{{-6}}$ (including the reused teacher-self solve in each trial). Not every solve is claimed to meet the target tolerance. Maximum iteration count  {d['max_iterations']['max']:.0f}. See JSON for full diagnostics.
- In the synthetic full-rank mixing example,  $X\approx XC^+C$ relative residual is about $1.23\times10^{{-15}}$, illustrating a structural blind spot of span reconstruction, not proving full rank for every actual C. Recorded  `student_m_ge_d` This is only a row-count condition, not measured rank.
- Set/span/fixed-spatial-readout losses are invariant to jointly permuting C and PU, unlike a causal decoder in general; these metrics cannot automatically guarantee fidelity for arbitrary decoders.
- Generator, samples, heads, line head, and key source SHA-256 values are in protocol/analysis; full outputs are additionally checked by [final_audit.json](../artifacts/token_proxy/encoder_candidates_v1/final_audit.json) Checks run per image/trial; source and original answer caches are never replaced according to favorable results.

## 7. Interpretation and Next Steps

Supported conclusion: **these specific strict encoder-only instances are not yet reliable per-trial downstream-quality proxies under the current balanced random convex-mixing distribution.** The results do not establish a defective encoder, impossibility of all decoder-free metrics, or inevitable failure of training with these losses. This is not a complete reproduction of the referenced algorithms and does not refute their original effectiveness.

Question weights rely on lexical overlap; the numeric head predicts types rather than exact values; line structure is not semantic relation understanding; subspace reconstruction permits arbitrary linear recovery unavailable to the LLM; random global mixing combines spatial evidence extensively. These explain varying loss despite near-zero downstream scores, or small loss despite lost answers. Around 80% zero scores and many ties further limit sensitivity to fine differences.

All VQA images come from an already analyzed evaluation batch: **exploratory replay, not unseen confirmatory holdout**. VQA scores fit neither formulas nor combinations. Added line/OT-precision work is labeled, and no retrospective “best” is declared validated.

A useful next step is to train and independently validate non-generative probes on separate OCR/table annotations, distinguishing values such as “12.5 / 125 / 12.6” and header–value bindings. Then test them on compression distributions that retain nonzero downstream performance and within-budget random variation. A changed compression family requires a new protocol rather than replacing this round's fixed-generator results. To claim cross-decoder generalization, freeze all loss parameters before testing another interface/LLM.

## 8. Execution and File Locations

```bash
cd /path/to/tessfold
# Existing venvs: .venv for CPU statistics/probes; .venv-und for Ascend encoder/OT.
bash scripts/run_encoder_candidates.sh status
# Recovery/rerun: reuse checkpoints and per-image caches; submit dependent Slurm jobs.
bash scripts/run_encoder_candidates.sh submit
# If all losses exist, recompute only statistics, numerical checks, and Markdown:
bash scripts/run_encoder_candidates.sh check
bash scripts/run_encoder_candidates.sh report
```

`submit` chains OCR protocol preparation → encoder-only OCR feature cache → evidence head (reuse if present) → CPU candidates / line probe / full-token NPU OT → mathematical checks and statistical reporting. **It requires the original  `random_v1_20x`  answer/feature/matrix caches**; to rebuild them from scratch, first follow the [20×-scale execution guide](TOKEN_PROXY_20X.md) instructions. Head/formula changes require a fresh experiment directory or removal of all derived losses, not new weights mixed with old cache. Defaults reuse existing experiments without destructive cleanup.

| File | Role |
|---|---|
| [protocol.json](../artifacts/token_proxy/encoder_candidates_v1/protocol.json) | Fixed inputs, hyperparameters, exclusions, and matrix/sample hashes |
| [encoder_evidence.py](../vqa/encoder_evidence.py) / [encoder_candidate_losses.py](../vqa/encoder_candidate_losses.py) | Lexical bridge, sampled coverage, evidence JS, ridge, spatially approximated OT |
| [exact_encoder_transport_npu.py](../scripts/exact_encoder_transport_npu.py) | Full-token OT, no spatial merging |
| [train_encoder_evidence.py](../scripts/train_encoder_evidence.py) / [train_encoder_relation.py](../scripts/train_encoder_relation.py) | Independent OCR / same-line-head training |
| [report_encoder_candidates.py](../scripts/report_encoder_candidates.py) | Unaveraged-trial statistics and complete CSV |
| [trials.csv](../artifacts/token_proxy/encoder_candidates_v1/trials.csv) | All 93,626 trials: gold, both answers/scores, and all new losses |
| [counterexamples.json](../artifacts/token_proxy/encoder_candidates_v1/counterexamples.json) | Actual answer pairs where lower loss gives greater drop at fixed question/retention/family |
| [analysis.json](../artifacts/token_proxy/encoder_candidates_v1/analysis.json) | Full pooled/stratified statistics, bootstrap CIs, controls, diagnostics, and source hashes |

in the CSV  `ratio` is **retention M/N**; `score_drop` Larger means worse; the proposed loss should also increase with degradation.`loss_sinkhorn_full_question_excess`, `loss_evidence_combined_line_question` are report fields; by  `sample_id / family / ratio` Filter the two seeds for a direct unaveraged comparison.
'''
    text=text.replace(chr(92)*2,chr(92))
    (ROOT/'docs/ENCODER_ONLY_LOSS_VALIDATION.md').write_text(text)
    print('Wrote docs/ENCODER_ONLY_LOSS_VALIDATION.md')
if __name__=='__main__':main()
