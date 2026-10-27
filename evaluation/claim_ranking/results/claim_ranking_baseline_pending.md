# Claim-Ranking Baseline

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

## Why metrics are pending

No manually labeled claim-ranking dataset is present in the repository. The 90-example NLI stance benchmark is a claim-evidence stance benchmark, not a claim-level fact-checking priority benchmark, so it cannot be used to compute claim-ranking baseline metrics. Baseline metrics must wait until independent claim-level annotations are collected and frozen.

## Missing data

- No manually labeled claim-ranking dataset.
- No baseline rankings recorded for a real claim set.

## Baseline metrics

| Objective | Status |
|---|---|
| O1 | pending annotation |
| O2 | pending annotation |
| O3 | pending annotation |

## Metric definitions

- **precision_at_k**: Number of claims with relevance_grade >= relevant_min_grade in the top K, divided by K. A K larger than the number of labeled claims in the objective's evaluation set is reported as null.
- **recall_at_k**: Number of claims with relevance_grade >= relevant_min_grade in the top K, divided by the total number of relevant claims in the objective's evaluation set. Null when there are no relevant claims or K exceeds the set size.
- **ndcg_at_k**: Discounted cumulative gain using exponential gain (2**grade - 1) and discount 1/log2(position + 1), divided by the ideal DCG over the same K and labels. Null when ideal DCG is zero or K exceeds the set size.
- **relevance_threshold**: Precision and recall use relevance_grade >= 1 as binary relevant. NDCG uses the full 0-3 grade.
- **k_selection**: K values are pre-registered before inspecting results. Default K values are 1, 3, 5, and 10. A K larger than the number of labeled claims in an objective's evaluation set is reported as null rather than computed on an undersized set.
- **ties**: A ranking is a total order. Baseline ranks are sorted by (rank, dataset order). For score-based methods, stable_rank_by_score sorts by descending score, places null last, and preserves input order for ties. The evaluation module does not reorder a supplied ranking.
- **missing_ecs**: null ECS is preserved as missing and is never imputed as 0.0. Baseline metrics do not use ECS. Future ECS-sensitive methods must report how missing ECS values are ordered or excluded.
- **duplicate_claims**: Duplicate claim IDs are rejected in the dataset and in ranking inputs. Each claim ID must identify exactly one claim.
- **insufficient_evidence**: A claim may have ecs=null and/or no evidence. It is not automatically excluded from baseline metrics if it has a human priority label and a baseline rank. Future ECS methods must document their handling.
- **exclusions**: A claim with a null baseline rank for an objective is excluded from that objective's metrics and listed with a reason. Dataset-level exclusions are recorded separately. No labels or ranks are imputed.

## Reproducibility

Regenerate this pending report from the project root:

```powershell
.\venv312\Scripts\python.exe -m evaluation.claim_ranking.baseline
```

Once a labeled dataset exists, run:

```powershell
.\venv312\Scripts\python.exe -m evaluation.claim_ranking.baseline --dataset <path-to-dataset.json> --k 1 3 5 10
```
