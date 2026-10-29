# ClaimRank AI API

Minimal FastAPI backend.

## Run from the project root

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

With an activated virtual environment, use:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Health check: <http://127.0.0.1:8000/api/v1/health>

## Dataset endpoints

Both endpoints accept a multipart `file` upload and a `dataset_type` form field
(`claims` or `perception`).

- `POST /api/v1/datasets/preview` returns detected columns, row count, and up
  to five rows.
- `POST /api/v1/datasets/validate` returns `valid`, row count, columns, errors,
  and warnings. Required columns are `claim_id,claim` for claims and
  `claim_id,rating` for perception. Schema problems return HTTP 200 with
  `valid: false`; empty or malformed uploads return HTTP 400.

## Prioritization runs

- `POST /api/v1/prioritization-runs` accepts multipart `claims_file`,
  `perception_file`, and `objective`. It returns `202` with a `run_id` and
  queues processing in a background task.
- `GET /api/v1/prioritization-runs/{run_id}/status` returns status, stage,
  progress, claim counts, and concise error details.
- `GET /api/v1/prioritization-runs/{run_id}/results` returns the stored
  per-claim results, including original/final ranks, available scores, ECS,
  evidence, conflict details, and component errors.
- `GET /api/v1/prioritization-runs/{run_id}/summary` returns counts and
  descriptive statistics computed only from stored results. `evidence_coverage`
  is the fraction of ranked claims with at least one evidence record. The
  high-conflict count uses the existing UI band of ECS >= 0.7; the threshold is
  included in the response. Scores are not probabilities.
- `GET /api/v1/prioritization-runs/{run_id}/claims` lists ranked claims with
  `q` (or `search`), `offset`, `limit`, `sort`, `order`, `ranking`, `priority`,
  `conflict`, `conflict_status`, `has_evidence`, `min_ecs`, `max_ecs`,
  `min_disputability`, and `max_disputability` query parameters. `limit`
  defaults to 20 and is capped at 100. `sort` supports `final_rank`,
  `initial_rank`, `claim_id`, `ecs_score`, `final_score`, `disputability`, and
  `rank_change`. `ranking=initial` changes the default sort to the initial
  rank. `priority` accepts `True`/`False` for `remove_false_claims` and
  `High`/`Low` for `correct_user_misconceptions`; for
  `reduce_user_disagreement`, use the disputability bounds.
- `GET /api/v1/prioritization-runs/{run_id}/claims/{claim_id}` returns one
  claim by its stable stored `claim_id`, including rank change, evidence, and
  conflict details.
- `GET /api/v1/prioritization-runs/{run_id}/export` downloads completed stored
  results as UTF-8 CSV. Columns include claim ID/text, ranks, available labels
  and scores, ECS, conflict counts/classifications, final priority score, and
  evidence count/URLs plus JSON columns preserving nested evidence and stance
  predictions. Missing values are left blank.

Results, summary, claim, and export endpoints return `409` until a run is
completed and a consistent `409` error for failed runs, `404` for unknown runs
or claims, and `422` for invalid filters. Runs are stored only in process memory.
They disappear on restart and are not shared across multiple Uvicorn workers, so
status polling must reach the worker that created the run (use one worker or
sticky routing).

## Fact checks

FactLLaMA is invoked only when a fact-check job is explicitly requested for a
claim in a completed prioritization run.

- `POST /api/v1/prioritization-runs/{run_id}/claims/{claim_id}/fact-check`
  returns `202` with a unique `job_id`, the claim, and an initial `queued`
  status. It reuses the claim's stored evidence and never retrieves evidence
  again. The run must be completed and the claim must exist with usable stored
  evidence.
- `GET /api/v1/fact-check-jobs/{job_id}` returns the job status:
  `queued`, `processing`, `completed`, or `failed`. A completed job includes
  the actual FactLLaMA verdict (`true`, `half`, or `false`), `raw_output`,
  evidence references, and limitations. `explanation` is `null` because the
  existing FactLLaMA prompt requests only a single verdict and does not produce
  structured reasoning. A failed job includes a sanitized error type and
  message; stack traces are never returned.

Unknown runs, claims, or jobs return `404`. Incomplete or failed runs return
`409`. Claims with a retrieval error, no stored evidence, unusable evidence, or
empty claim text return `422`. Jobs are stored only in process memory and are
lost on restart. The FactLLaMA service is cached per process, the model is
loaded lazily on first inference, and inference is serialized so concurrent
requests do not load duplicate model instances. Real inference requires the
FactLLaMA dependencies, base model, and adapter described in
`docs/factllama.md`; tests use mocked inference and do not load weights.

## Model evaluation

These endpoints are read-only views of saved offline evaluation artifacts. They
never train models, run inference, retrieve evidence, or recompute metrics.

- `GET /api/v1/evaluation/summary` returns available metrics and experiment
  metadata from:
  - the newest `evaluation/factllama/results/rawfc_test_factllama_*.json`
  - `evaluation/evidence_conflict/results/nli_stance_benchmark.json`
  - `evaluation/claim_ranking/results/claim_ranking_fusion.json` (or the
    pending report when no computed report exists)
  - `evaluation/retrieval/results/retrieval_metrics.json` when present

  The response has top-level `metrics` keyed by experiment, an `experiments`
  list with dataset/model/configuration metadata, and an `unavailable` list with
  reasons. A `null` metric means unavailable, not zero. Metrics from different
  tasks and datasets are never combined into a composite score.
- `GET /api/v1/evaluation/prioritization-comparison` compares the saved
  bias-aware-only (`baseline`), ECS-only, and RRF-combined (`fused`) results.
  When no labeled dataset exists, it returns `available: false` with
  `status: pending_annotation`, null metrics, the saved dataset metadata, the
  K values, and the RRF configuration. Computed reports expose metrics per
  objective because each objective has its own eligible claim set.
- `GET /api/v1/evaluation/retrieval-metrics` reads
  `evaluation/retrieval/results/retrieval_metrics.json` if it exists. The
  artifact should contain a `metrics` object preserving names such as
  `Recall@K`, `Precision@K`, and `MRR`, plus optional `dataset`,
  `configuration`, and `generated_at` fields. Without the artifact, the
  response is `available: false`, `metrics: null`, and lists the expected
  metric names; no retrieval values are inferred from other evaluations.

A saved artifact that exists but is unreadable or malformed returns `503` with
a sanitized source/message; stack traces are not exposed. A missing artifact is
reported as unavailable with an explanatory note. The frontend Model Evaluation
page remains illustrative sample data; it is not changed by these endpoints.

## CORS

The frontend origin defaults to the Vite dev server at `http://localhost:5173`.
Override it with the `FRONTEND_ORIGIN` environment variable. Multiple origins
can be supplied as a comma-separated list.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/api
```
