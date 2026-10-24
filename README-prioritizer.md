# Bias-Aware News Prioritizer

This repository is an ML major project. **Phase 1 implements only the
bias-aware prioritization feature.** Evidence retrieval, FactLLaMA, the
Evidence Conflict Score, API routes, and a frontend are intentionally out of
scope for this phase.

## What the prioritization feature does

The feature accepts claims with stable claim IDs and user-perception ratings
associated with those IDs. Ratings are converted to **PTL (Perceived Truth
Level)** values, and claims are ranked according to one selected objective.

| Objective | Model/metric | Returned claims |
|---|---|---|
| **O1 — Removing False News** | `random_forest_model_o1.pkl` predicts `True`/`False` | Only claims predicted as `False`, in input claim order |
| **O2 — Correcting Misperceptions** | `random_forest_model.pkl` predicts `High`/`Low` TPB | Only claims predicted as `High`, in input claim order |
| **O3 — Decreasing Disagreement** | Sample variance of PTL (`ddof=1`) | All rated claims, sorted by descending `disputability` |

> Model predictions are classifier outputs, **not verified facts**. This
> feature does not fact-check claims. A successful unit-test run is not an
> accuracy evaluation.

O1 and O2 use the saved classifier's prediction directly. No confidence
thresholds, probability-based sorting, or incoming-data TPB median thresholds
are used. O3 does not load a model or scaler.

## Feature extraction

Ratings use the notebook's exact PTL mapping:

| Rating | PTL |
|---|---:|
| `True` | `1.0` |
| `Mostly True` | `0.5` |
| `Mixture` | `0.0` |
| `Mostly False` | `-0.5` |
| `False` | `-1.0` |

For O1 and O2, the four features are extracted per claim in exactly this order:

1. `mean`
2. `median`
3. `variance`
4. `skewness`

`variance` is sample variance (`ddof=1`), and `skewness` uses
`scipy.stats.skew` with its notebook-default behavior. Missing or undefined
feature values are replaced with `0` before scaling, matching the notebook.
The saved `StandardScaler` is used with `transform()` only; it is never fitted
or refitted on inference data. Claim text is not a feature.

For O3, `disputability` is the sample variance of PTL. A claim with only one
rating has undefined variance; its public `disputability` is `None` and it is
placed after claims with defined variance. Ties preserve input claim order.

The implementation never calculates GTL or TPB for incoming claims because
their ground truth is unknown.

## Repository layout

```text
src/prioritization/
├── __init__.py
├── config.py
├── errors.py
├── schemas.py
├── preprocessing/
│   ├── __init__.py
│   └── ratings.py
├── features/
│   ├── __init__.py
│   └── extraction.py
└── inference/
    ├── __init__.py
    ├── artifacts.py
    └── prioritizer.py
```

## Required model artifacts

Place these local files in `models/prioritizer/`:

| File | Purpose |
|---|---|
| `random_forest_model_o1.pkl` | O1 `True`/`False` classifier |
| `random_forest_model.pkl` | O2 `High`/`Low` TPB classifier |
| `scaler.pkl` | Shared four-feature `StandardScaler` |

The artifacts are already ignored by `.gitignore` (`models/**` and `*.pkl`).
They must remain untracked and must not be retrained or replaced by this
feature. Do not load pickle artifacts from untrusted sources.

The artifacts in this repository were serialized with **scikit-learn 1.6.1**.
O1 and O2 expect four input features; the scaler records the feature order
`mean`, `median`, `variance`, `skewness`. The loader validates artifact types,
feature counts, scaler feature order, and model class labels.

## Configuration

The default project-relative paths are used automatically. Optional
environment variables may override them:

| Variable | Purpose |
|---|---|
| `PRIORITIZATION_ARTIFACT_DIR` | Directory containing all three artifacts |
| `PRIORITIZATION_O1_MODEL_PATH` | Full path to the O1 model |
| `PRIORITIZATION_O2_MODEL_PATH` | Full path to the O2 model |
| `PRIORITIZATION_SCALER_PATH` | Full path to the scaler |

File-specific variables take precedence over `PRIORITIZATION_ARTIFACT_DIR`.
Relative values resolve against the repository root. URL-looking values are
rejected. The public `prioritize_claims()` function does not accept paths or
URLs; artifact configuration belongs to trusted deployment wiring.

`.env.example` documents the variable names, but the project does not
automatically load `.env` files. Set variables in the process environment.

## Installation

Requires Python 3.10 or newer. Tested with Python 3.12.10.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` allows newer scikit-learn releases for development. For
exact artifact parity with the serialized version, create a dedicated
environment and install scikit-learn 1.6.1:

```powershell
python -m venv .venv-sklearn161
.\.venv-sklearn161\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "scikit-learn==1.6.1" "numpy>=1.26,<3" "scipy>=1.11,<2" "joblib>=1.3,<2" "pytest>=8,<10"
```

The existing `venv312` environment uses scikit-learn 1.9.1. Loading artifacts
serialized by 1.6.1 under 1.9.1 emits `InconsistentVersionWarning`; successful
loading and smoke predictions do not by themselves prove full cross-version
prediction compatibility.

## Usage

```python
from src.prioritization import Claim, UserRating, prioritize_claims

claims = [
    Claim(claim_id="1", claim="Example claim"),
    Claim(claim_id="2", claim="Another claim"),
]

ratings = [
    UserRating(claim_id="1", rating="True"),
    UserRating(claim_id="1", rating="False"),
    UserRating(claim_id="2", rating="Mostly True"),
]

results = prioritize_claims(claims, ratings, objective="O3")

for result in results:
    print(result.rank, result.claim_id, result.disputability)
```

Ratings may be supplied in any order; they are associated with claims by exact
claim ID. Claim IDs may be strings or integers and are preserved. Ratings for
unknown IDs, duplicate claim IDs, unsupported rating labels, and claims without
perception records are rejected.

A model-backed objective:

```python
from src.prioritization import Claim, UserRating, prioritize_claims

claims = [Claim("1", "Example claim")]
ratings = [
    UserRating("1", "True"),
    UserRating("1", "Mostly True"),
]

results = prioritize_claims(claims, ratings, objective="O1")
```

### Output records

Each `PrioritizedClaim` contains:

- `claim_id`
- `claim`
- `objective`
- `rank` (1-based)
- `predicted_truth` for O1, otherwise `None`
- `predicted_tpb` for O2, otherwise `None`
- `disputability` for O3, otherwise `None`

There is no combined objective score.

## Tests

Run all prioritization tests (real-artifact tests run automatically when the
local artifacts are present):

```powershell
.\venv312\Scripts\python.exe -m pytest tests\prioritization -q
```

Run only the synthetic unit tests, without the local real-artifact smoke test:

```powershell
.\venv312\Scripts\python.exe -m pytest tests\prioritization -q --ignore=tests\prioritization\test_real_artifacts.py
```

Run only the real-artifact smoke test:

```powershell
.\venv312\Scripts\python.exe -m pytest tests\prioritization\test_real_artifacts.py -q
```

Compile-check the package and tests:

```powershell
.\venv312\Scripts\python.exe -m compileall -q src\prioritization tests\prioritization
```

The real-artifact smoke test uses the actual local model files and synthetic
ratings. It verifies loading, scaler shape/feature order, allowed class labels,
public API execution, and O3 variance ordering. It is not an accuracy test.

## Validation behavior

- Invalid objectives are rejected before artifact loading.
- Duplicate claim IDs are rejected.
- Ratings for unknown claim IDs are rejected.
- Unsupported or malformed rating labels are rejected.
- Claims without perception records are rejected with their IDs reported.
- Missing artifacts and incompatible artifact types/feature counts are rejected.
- Unexpected model output labels or prediction counts raise inference errors.
- Claim and rating IDs must match exactly; values are preserved.

## Known limitations

- No GTL or TPB is computed for incoming claims.
- No retrieval, FactLLaMA, Evidence Conflict Score, API routes, frontend, CSV
  upload, or deployment configuration is included in this phase.
- O1/O2 are classifiers trained on rating-derived features; their predictions
  are not fact verification.
- O3 cannot define variance for a single rating. It returns `None` for that
  field and ranks the claim after defined-variance claims.
- Cross-version scikit-learn artifact compatibility is not independently
  proven; use scikit-learn 1.6.1 for exact parity.
