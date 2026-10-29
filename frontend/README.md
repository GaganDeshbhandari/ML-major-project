# ClaimRank AI Frontend

React + Vite dashboard for the ClaimRank API.

## Development

Start the API from the project root:

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

The API base URL defaults to `http://localhost:8000/api/v1`. Override it with a
Vite environment variable when needed:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000/api/v1"
npm run dev
```

The API allows the Vite origin `http://localhost:5173` by default. Set
`FRONTEND_ORIGIN` on the API process for a different origin.

## Workflows

- The dashboard validates both CSVs with `/datasets/validate` and shows a
  preview from `/datasets/preview` before enabling a run.
- Creating a run posts to `/prioritization-runs`; the claims page polls the
  run status and then loads `/summary` and paginated `/claims` results.
- Search, evidence/ECS filters, final/initial ranking, and CSV export call the
  corresponding API endpoints. Claim detail loads one stored claim.
- FactLLaMA is requested only when the user clicks the fact-check button; the
  page then polls the in-memory job endpoint.
- Model evaluation loads `/evaluation/summary`,
  `/evaluation/prioritization-comparison`, and `/evaluation/retrieval-metrics`.
  Pending or missing metrics are shown as unavailable, never as zero.

Run state is in memory. If the API restarts, expired run and job IDs show an
explanatory error with a path back to a new run.

## Checks

```powershell
npm run build
```

There is no frontend test framework configured in this project.
