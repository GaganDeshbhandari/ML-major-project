# Claim-Ranking Fusion

**Status: pending annotation.** No metric values are reported.

## Dataset

| Field | Value |
|---|---|
| Dataset ID | `claim-ranking-eval` |
| Dataset version | `1` |
| Schema version | `1.0` |
| Labeling guide version | `1` |
| Dataset path | not available |
| Dataset SHA-256 | not available |
| Expected dataset path | `evaluation/claim_ranking/data/claim_ranking_dataset_v1.json` |
| Claims evaluated | 0 |
| RRF k | 60.0 |
| ECS-only missing policy | `exclude_missing_ecs` |

## Why metrics are pending

No manually labeled claim-ranking dataset is present in the repository. The 90-example NLI stance benchmark is a claim-evidence stance benchmark, not a claim-level fact-checking priority benchmark. Baseline, ECS-only, and RRF-fused metrics must wait until independent claim-level annotations are collected and frozen.

## Missing data

- No manually labeled claim-ranking dataset.
- No baseline rankings or ECS values recorded for a real claim set.

## Methods

| Method | Status |
|---|---|
| Baseline O1/O2/O3 | pending annotation |
| ECS-only | pending annotation |
| RRF fused | pending annotation |

## Metric definitions

- **precision_at_k**: Number of claims with relevance_grade >= relevant_min_grade in the top K, divided by K. A K larger than the number of labeled claims in the objective's evaluation set is reported as null.
- **recall_at_k**: Number of claims with relevance_grade >= relevant_min_grade in the top K, divided by the total number of relevant claims in the objective's evaluation set. Null when there are no relevant claims or K exceeds the set size.
- **ndcg_at_k**: Discounted cumulative gain using exponential gain (2**grade - 1) and discount 1/log2(position + 1), divided by the ideal DCG over the same K and labels. Null when ideal DCG is zero or K exceeds the set size.
- **relevance_threshold**: Precision and recall use relevance_grade >= 1 as binary relevant. NDCG uses the full 0-3 grade.
- **k_selection**: K values are pre-registered before inspecting results. Default K values are 1, 3, 5, and 10. A K larger than the number of labeled claims in an objective's evaluation set is reported as null rather than computed on an undersized set.
- **baseline**: The selected objective's existing O1, O2, or O3 ranking. It is never replaced by a synthetic combined objective ranking.
- **ecs_only**: Claims with finite recorded ECS values, ordered by descending ECS. ECS ties use baseline order. Missing ECS claims are excluded and reported; they remain in the shared relevance set for metric denominators.
- **fused**: Reciprocal Rank Fusion of the selected baseline ranking and the ECS ranking: 1/(k + baseline_rank) + 1/(k + ecs_rank), using 1-based ranks. Missing ECS claims receive only the baseline contribution.
- **rrf_k**: Positive finite RRF smoothing constant. The default is 60.0, matching the standard RRF convention.
- **ties**: ECS ties preserve baseline order. Final RRF ties resolve by baseline rank, then original input order. All rankings are total orders.
- **missing_ecs**: null ECS is never imputed as 0.0. The fused ranking keeps missing-ECS claims with a baseline-only contribution. ECS-only evaluation excludes them and reports the affected claim IDs.

## Reproducibility

Regenerate this pending report from the project root:

```powershell
.\venv312\Scripts\python.exe -m evaluation.claim_ranking.fusion
```

Once a labeled dataset exists, run:

```powershell
.\venv312\Scripts\python.exe -m evaluation.claim_ranking.fusion --dataset <path-to-dataset.json> --k 1 3 5 10 --rrf-k 60
```
