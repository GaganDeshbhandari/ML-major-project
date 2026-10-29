# Claim-Ranking Evaluation

This directory defines an offline, reproducible evaluation framework for
measuring whether Evidence Conflict Score (ECS) improves the existing
bias-aware claim-prioritization pipeline.

**Phase 7A status: evaluation framework complete.** Dataset validation,
Precision@K, Recall@K, NDCG@K, and per-objective baseline reporting are
implemented.

**Phase 7B status: opt-in RRF fusion implemented.** Production defaults are
unchanged. RRF is requested explicitly and combines only the single selected
O1, O2, or O3 baseline ranking with a ranking derived from already-computed
ECS values. FactLLaMA integration, prompts, and the frozen NLI benchmark are
untouched.

**Empirical status: pending annotation.** The repository does not contain a
manually labeled claim-level fact-checking priority dataset. No baseline,
ECS-only, or fused metric values are reported, and no labels were generated
from ECS, model outputs, or the NLI stance benchmark.

## Existing ranking behavior

The production prioritizer selects one objective at a time. It does **not**
produce a combined objective score, and this evaluation does not invent one.

| Objective | Implemented score | Returned claims | Ranking behavior |
|---|---|---|---|
| O1 | Predicted probability of `False` from the O1 random forest | All claims | Descending probability; ties preserve input order |
| O2 | Predicted probability of `High` TPB from the O2 random forest | All claims | Descending probability; ties preserve input order |
| O3 | Sample variance of PTL (`ddof=1`) | All rated claims | Descending disputability; undefined variance (one rating) last |

The public `PrioritizedClaim` output contains `claim_id`, `claim`, `objective`,
`rank` (1-based), and the objective-specific fields `predicted_truth`,
`predicted_tpb`, or `disputability`. There is no combined rank or combined
score in the output.

The pipeline-level `PrioritizationRetrievalResult` preserves those fields as
`initial_rank` and adds retrieved evidence, `retrieval_error`, and optional ECS
fields. Each result belongs to one claim:

- `ecs_result` is the claim-level `EvidenceConflictResult` when ECS is enabled
  and scoring succeeds.
- `ecs_result.ecs` is set only when `ecs_result.status` is `COMPUTED`; otherwise
  it is `None`, never `0.0` by convention for missing values.
- `ecs_error` records an expected NLI or scoring failure.
- `retrieval_error` records an expected retrieval failure. ECS is not computed
  for that claim.
- Successful retrieval with no evidence produces `NO_EVIDENCE`; all
  `NOT_ENOUGH_INFO` predictions produce `INSUFFICIENT_STANCE` with `ecs=None`.

Baseline evaluation must use the objective-specific ranking, not a fabricated
combined score.

## Opt-in RRF fusion

The default pipeline order is unchanged. Fusion is an explicit post-processing
step after prioritization, retrieval, and optional ECS scoring:

```python
from src.pipeline import fuse_prioritization_results, prioritize_and_retrieve

results = prioritize_and_retrieve(
    claims,
    ratings,
    "O1",
    ecs_config=ecs_config,
)
fused = fuse_prioritization_results(results, k=60)
```

`fuse_prioritization_results` returns `FusedPrioritizationResult` objects. Each
one retains the original `PrioritizationRetrievalResult` as `.result` and adds:

- `baseline_rank`: the selected objective's original `initial_rank`
- `ecs_rank`: the claim's 1-based rank among valid ECS values, or `None`
- `rrf_score`: the fused score
- `fused_rank`: the 1-based fused position

The original ranks, objective scores, evidence dictionaries, and ECS result
objects are not mutated. `prioritize_and_retrieve` keeps its existing signature
and default behavior.

### Formula and rank indexing

RRF uses 1-based ranks:

```text
RRF(claim) = 1 / (k + baseline_rank)
             + 1 / (k + ecs_rank)    # only when ecs_rank exists
```

`k` is a configurable positive finite smoothing constant and defaults to `60`.
The pure implementation is `src/ranking_fusion.py`; the pipeline adapter is
`src/pipeline/ranking_fusion.py`.

### ECS ranking validity

Only claims whose ECS status is `COMPUTED` and whose ECS value is finite receive
an ECS rank. Higher ECS ranks first. ECS ties preserve baseline order, defined
by `(baseline_rank, original input order)`. A computed ECS of `0.0` is valid.
`None`, non-finite values, and all non-computed statuses receive no ECS rank.
No ECS value is inferred from NLI confidence, stance counts, retrieval
relevance, or any other field.

### Missing ECS in fusion

Claims without a valid ECS remain in the fused ranking and receive only the
baseline contribution:

```text
RRF(claim) = 1 / (k + baseline_rank)
```

No fabricated ECS rank or score is assigned.

### Final ordering

Claims are sorted by descending RRF score. Exact RRF ties resolve by baseline
rank, then original input order. Every eligible input claim appears exactly
once.

## Why the NLI benchmark is not reused

`evaluation/evidence_conflict/data/nli_stance_benchmark_v1.json` contains
claim-evidence pairs labeled for three-class stance (`supports`, `refutes`,
`not_enough_info`). It does not contain human labels for claim-level
fact-checking priority, and its claims were selected for stance diagnostics,
not for ranking evaluation. It must not be treated as a claim-ranking
benchmark.

## Dataset readiness

The repository currently has no suitable claim-level ranking labels:

- `data/` contains only placeholders.
- The NLI benchmark labels stance, not ranking priority.
- The prioritization artifacts contain no human relevance or priority labels.
- There is no existing claim-ranking results file.

Therefore baseline, ECS-only, and fused metric computation is blocked on
annotation. The framework below is ready to validate and score a dataset once
one is created.

## Dataset schema

A dataset is one UTF-8 JSON object. The machine-readable schema is
`data/claim_ranking_schema_v1.json`. The expected populated dataset path is
`data/claim_ranking_dataset_v1.json`; that file is intentionally absent.

### Top-level fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `dataset_id` | non-empty string | yes | Stable dataset identifier |
| `version` | non-empty string | yes | Dataset version; freeze before scoring |
| `description` | non-empty string | yes | What the dataset covers and how it was selected |
| `labeling_guide_version` | non-empty string | yes | Version of the labeling guide used |
| `examples` | non-empty array | yes | Labeled claim records |
| `exclusions` | array | no | Claims considered but not evaluated, with reasons |

### Example fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `claim_id` | non-empty string | yes | Stable, unique claim identifier |
| `claim` | non-empty string | yes | Claim text shown to annotators |
| `relevance_grade` | integer 0-3 | yes | Human priority/relevance label |
| `rationale` | non-empty string | yes | Brief reason for the label |
| `baseline_ranks` | object | yes | Recorded O1/O2/O3 ranks, each positive integer or `null` |
| `ecs` | number in `[0,1]` or `null` | yes | Recorded ECS; `null` means missing, never `0.0` |
| `evidence` | array | no | Optional evidence records for reproducibility |
| `reproduction_notes` | string or `null` | no | Citation or access notes when evidence cannot be redistributed |

`baseline_ranks` must contain exactly `O1`, `O2`, and `O3`. A `null` rank means
the claim was not ranked for that objective (for example, missing ratings for
O3) and is excluded from that objective's baseline metrics with a reason. The
baseline ranks must be produced by the existing production pipeline on the same
claim set and recorded; they are not generated by ECS.

The current ECS formula in `src/evidence_conflict/scoring.py` yields values in
`[0,1]`, and the dataset schema validates that range. If the ECS scale changes,
create a new dataset-schema version rather than coercing or clipping recorded
values.

Each evidence record contains a required non-empty `content` and optional
`title`, `url`, `retrieved_at`, and `license`. If full text cannot be
redistributed, omit the evidence record entirely and keep the citation in
`reproduction_notes` rather than inventing evidence.

Dataset-level `exclusions` must not also appear as examples. They record claims
considered but omitted, with a non-empty reason. Per-objective exclusions for
missing baseline ranks are computed and reported separately.

### Dataset provenance and versioning

- Assign a stable `dataset_id` and increment `version` when labels, claim text,
  or baseline ranks change.
- Record the `labeling_guide_version`.
- Freeze the dataset file before computing or comparing results.
- The loader records the SHA-256 of the exact bytes read from disk.
- Do not edit a frozen dataset to improve results; create a new version.

## Labeling guide

The label is **priority for human fact-checking**, not whether the claim is
true or false. Labels must be assigned without looking at ECS, baseline ranks,
model predictions, or any candidate ranking.

### Relevance grades

| Grade | Label | Criteria |
|---:|---|---|
| 3 | Urgent / high priority | Checkable factual claim with potentially serious public harm if false or misleading; broad reach or high public relevance; time-sensitive or consequential; evidence may be conflicting or hard to interpret. |
| 2 | Important | Checkable and consequential, but lower urgency, reach, or potential harm than grade 3; still worth a fact-checker's attention. |
| 1 | Low priority | Checkable but niche, low-impact, already well established, or of limited public consequence. |
| 0 | Non-priority / non-relevant | Checkable claim that should remain in the evaluation set but is not a fact-checking priority. It counts as non-relevant for Precision@K and Recall@K and contributes zero gain to NDCG@K. |

A claim must be a factual assertion that can in principle be checked. Do not
reward a claim merely because ECS is high, evidence is conflicting, or a model
is uncertain. Do not use the claim's truth value as the priority label.

Opinion, prediction, vague or non-factual text, duplicates, claims missing
essential context, and claims that cannot be checked with available evidence are
**not grade 0 examples**. Omit them from `examples` and record them in the
dataset-level `exclusions` list with a reason.

### Annotation protocol

1. Give annotators the claim text, the codebook above, and any permitted
   evidence. Do not show ECS, baseline ranks, model scores, or other annotators'
   labels.
2. Label independently. Record the final grade and a short rationale for every
   claim.
3. Resolve disagreements by discussion and adjudication. Preserve the original
   independent labels if agreement statistics are reported.
4. Exclude claims that cannot be labeled consistently; record them in the
   dataset-level `exclusions` list with a reason.
5. Freeze labels before joining ECS values or baseline ranks.

A small diagnostic set can expose pipeline problems, but it is not a
statistically powered estimate of real-world performance. Report the dataset
size and selection process; do not describe synthetic or smoke-test labels as
real-world evidence.

## Metrics

The implementation is in `metrics.py`. All metrics are computed on an explicit
evaluation set of labeled claims.

### Precision@K

`P@K = (# claims with relevance_grade >= 1 in the top K) / K`

### Recall@K

`R@K = (# relevant claims in the top K) / (total relevant claims in the evaluation set)`

### NDCG@K

`NDCG@K = DCG@K / IDCG@K`

- Gain: `2**grade - 1`
- Discount at 1-based position `p`: `1 / log2(p + 1)`
- IDCG uses all labels in the evaluation set, sorted by descending grade and
  truncated to K.

NDCG uses the full graded labels. Precision and recall use binary relevance
(`grade >= 1`).

### K selection

K values are pre-registered before inspecting results. The default set is
`1, 3, 5, 10`. Use the same K values for every method being compared.

A K larger than the number of labeled claims in an objective's evaluation set
is reported as `n/a` (`None` in JSON), not computed on an undersized set. A
method that returns fewer than K claims on a sufficiently large evaluation set
is treated as missing the unreturned positions: Precision@K divides by K,
Recall@K misses relevant claims, and NDCG@K gets no credit for absent positions.

### Ties

A ranking is a total order. The metric functions preserve the supplied order
and never reorder ties silently.

- Baseline ranks are sorted by `(rank, dataset order)`.
- For score-based methods, `stable_rank_by_score` sorts by descending score,
  places `None` last, and preserves input order for ties. This mirrors the
  production prioritizer's stable-sort behavior and is evaluation-only.
- Apply the same tie policy to every method being compared and document it.

### Missing ECS values

`ecs: null` means missing and is never imputed as `0.0`. Baseline metrics do
not use ECS. A claim with a missing ECS remains in baseline metrics if it has a
human label and baseline rank.

The implemented ECS-sensitive policies are explicit:

- **Fused RRF ranking:** missing-ECS claims remain in the ranking and receive
  only the baseline contribution. They receive `ecs_rank=None`.
- **ECS-only evaluation:** missing-ECS claims are excluded from the ECS-only
  ranking and reported in `missing_ecs_claim_ids` and `exclusions`. They remain
  in the shared relevance set, so metric denominators are unchanged and
  unranked positions are treated as absent. No score is imputed.

### Duplicate claims

Duplicate `claim_id` values are rejected during dataset validation. Duplicate
claim IDs in a ranking are rejected by the metric functions. Each ID identifies
exactly one claim.

### Insufficient evidence

A claim may have `ecs: null` and/or no evidence. It is not automatically
excluded from baseline metrics if it has a human priority label and a baseline
rank. Insufficient evidence is not a relevance label and must not be converted
into one.

### Exclusions and comparisons

- Claims with a null baseline rank for an objective are excluded from that
  objective's metrics and listed with a reason.
- Dataset-level exclusions are reported separately.
- Labels and ranks are never imputed.
- Compare methods only on the same claim IDs, human labels, K values, relevance
  threshold, and tie policy. The three objectives may have different eligible
  sets because O3 can lack ratings; report each objective's set size and
  exclusions.

## Baseline report

No baseline metrics are reported because no real labels exist. The pending
report is generated by:

```powershell
.\venv\Scripts\python.exe -m evaluation.claim_ranking.baseline
```

It writes:

- `results/claim_ranking_baseline_pending.json`
- `results/claim_ranking_baseline_pending.md`

The report states the dataset identifier and version, K values, metric
definitions, number of claims evaluated (`0`), exclusions, and the reason
metrics are pending. Dataset path and SHA-256 are `null`; no metric values are
fabricated.

Once a frozen, manually labeled dataset exists, run:

```powershell
.\venv\Scripts\python.exe -m evaluation.claim_ranking.baseline `
  --dataset evaluation/claim_ranking/data/claim_ranking_dataset_v1.json `
  --k 1 3 5 10
```

This writes `claim_ranking_baseline.json` and `claim_ranking_baseline.md` with
the dataset path and SHA-256, per-objective eligible claim counts, exclusions,
and baseline metrics.

## Fusion comparison report

`evaluation/claim_ranking/fusion.py` compares three methods on the same labeled
claim set:

1. **Baseline:** the selected O1, O2, or O3 ranking.
2. **ECS-only:** valid ECS values in descending order, with missing-ECS claims
   excluded and reported.
3. **Fused:** baseline plus ECS using RRF, with missing-ECS claims receiving
   only the baseline contribution.

All methods use the same objective-eligible claim IDs, human labels,
pre-registered K values, relevance threshold, and tie policy. The report records
per-objective rankings, exclusions, metrics, RRF scores, ECS ranks, and the RRF
`k` value.

No fusion metrics are reported because no real labels exist. Generate the
pending report with:

```powershell
.\venv\Scripts\python.exe -m evaluation.claim_ranking.fusion
```

It writes:

- `results/claim_ranking_fusion_pending.json`
- `results/claim_ranking_fusion_pending.md`

Once a frozen, manually labeled dataset exists, run:

```powershell
.\venv\Scripts\python.exe -m evaluation.claim_ranking.fusion `
  --dataset evaluation/claim_ranking/data/claim_ranking_dataset_v1.json `
  --k 1 3 5 10 --rrf-k 60
```

This writes `claim_ranking_fusion.json` and `claim_ranking_fusion.md`.

## Tests

The evaluation and fusion tests are offline and use synthetic fixtures only:

```powershell
.\venv\Scripts\python.exe -m pytest tests\claim_ranking tests\pipeline\test_ranking_fusion.py -q
```

They do not download models, access external APIs, require a GPU, or run
production inference. The pure RRF core is imported directly by the evaluation
harness; no production model or retrieval code is invoked. Synthetic fixtures
are not saved as evaluation results.

## Scope boundaries

Phase 7B adds opt-in RRF fusion and a comparison report. It does not change the
default prioritization order, O1/O2/O3 scoring, retrieval, ECS scoring, or
FactLLaMA inputs or prompts. It does not modify the frozen NLI benchmark, add
dependencies, or commit or push changes. Empirical ranking-quality claims
remain blocked until independent claim-level annotations exist.
