---
title: DATFID
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: main.py
hidden: false
---

# DATFID API - Hackathon Backend

This is a DATFID-style FastAPI backend for the Campus Founders AI Hackathon challenge: scaling CPU inference to concurrent users.

DATFID users upload CSV/XLSX panel data, fit an interpretable forecasting model, and generate forecasts.

## What Changed

The original demo path is preserved:

- `POST /modelfit/`
- `POST /modelforecast/`
- `POST /modelfit-file/`
- `POST /modelforecast-file/`

Those endpoints are synchronous and keep fitted model state in process-global variables. That is unsafe for concurrent users because a later fit can overwrite the model used by another user's forecast.

This prototype adds a new job-based path:

- `POST /jobs/fit-file`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/fit-result`
- `POST /jobs/{fit_job_id}/forecast-file`
- `GET /jobs/{forecast_job_id}/forecast-result`
- `GET /metrics`

The new path gives each workflow a `fit_job_id`, so forecasts use the intended fitted model instead of a global `stored_model`.

## Architecture

```text
client
  -> POST /jobs/fit-file or /jobs/{fit_job_id}/forecast-file
  -> FastAPI creates a job and internal DAG
  -> bounded job admission
  -> ready DAG node queue
  -> worker threads execute stages
  -> in-memory job/artifact stores
  -> polling/result endpoints
```

Internal DAG stages:

```text
fit_parse_file -> fit_parse_params -> fit_model -> fit_report
forecast_parse_file -> forecast_run
```

Repeated parsing, fitting, and forecasting are materialized with deterministic cache keys. This lets repeated workloads reuse artifacts instead of recomputing everything.

## Demo Result

On the DATFID M5 sample dataset:

```text
First run:
old sync fit avg:     ~5.91s
new job submit avg:   ~0.59s
new job complete avg: ~3.65s

Second run with warm DAG/materialization cache:
old sync fit avg:     ~5.58s
new job submit avg:   ~0.62s
new job complete avg: ~0.85s
```

Main takeaway:

```text
The API returns control much faster, repeated workflows complete faster, and forecasts are isolated by fit_job_id.
```

## Run Locally

```bash
cd datfid-api
DATFID_MAX_WORKERS=2 DATFID_MAX_QUEUE_SIZE=16 DATFID_MAX_STORED_JOBS=100 \
DATFID_DAG_CACHE_ENABLED=true DATFID_MAX_MATERIALIZED_ARTIFACTS=128 \
  uvicorn main:app --host 0.0.0.0 --port 7860
```

For the deployed hackathon backend:

```bash
export DATFID_BASE_URL=https://huseyinkocer-datfid-api.hf.space
```

The `datfid-master` Space still proxies only the original synchronous routes. Use the API Space directly for `/jobs/...`.

## Example

Submit a fit job:

```bash
curl -sS \
  -F file=@sample.csv \
  -F id_col=entity_id \
  -F time_col=date \
  -F y=y \
  -F current_features='["x1","x2"]' \
  ${DATFID_BASE_URL:-http://127.0.0.1:7860}/jobs/fit-file
```

Poll:

```bash
curl -sS ${DATFID_BASE_URL:-http://127.0.0.1:7860}/jobs/<fit_job_id>
```

Forecast from that fitted model:

```bash
curl -sS \
  -F df_forecast=@forecast.csv \
  ${DATFID_BASE_URL:-http://127.0.0.1:7860}/jobs/<fit_job_id>/forecast-file
```

Metrics:

```bash
curl -sS ${DATFID_BASE_URL:-http://127.0.0.1:7860}/metrics
```

Important metric fields:

- `queue_size`
- `jobs_succeeded`
- `jobs_failed`
- `cache_hits`
- `cache_misses`
- `materialized_artifacts`
- `dag_nodes_succeeded`
- `dag_nodes_failed`

## Benchmark

Synthetic:

```bash
python scripts/benchmark_concurrency.py \
  --base-url ${DATFID_BASE_URL:-http://127.0.0.1:7860} \
  --requests 8 \
  --concurrency 4 \
  --json-output benchmark.json
```

M5 sample:

```bash
mkdir -p ../tmp_samples
curl -L \
  -o ../tmp_samples/M5_Department.xlsx \
  https://raw.githubusercontent.com/datfid-valeriidashuk/sample-datasets/main/M5_Department.xlsx

python scripts/benchmark_concurrency.py \
  --base-url ${DATFID_BASE_URL:-http://127.0.0.1:7860} \
  --train-file ../tmp_samples/M5_Department.xlsx \
  --id-col agg_id \
  --time-col ds \
  --target Sales \
  --current-features all \
  --requests 4 \
  --concurrency 2 \
  --json-output benchmark_m5.json
```

Run the M5 benchmark twice without restarting the Space to show cold vs warm materialization.

## Config

```text
DATFID_MAX_WORKERS=2
DATFID_MAX_QUEUE_SIZE=16
DATFID_MAX_STORED_JOBS=100
DATFID_DAG_CACHE_ENABLED=true
DATFID_MAX_MATERIALIZED_ARTIFACTS=128
```

## Limitations

- In-memory only; state disappears on Space restart/sleep.
- Keep uvicorn at one process because state is process-local.
- Cache is global for the demo and not tenant-scoped.
- Worker threads bound execution and keep requests responsive, but they are not guaranteed CPU speedup for pure Python code.
- `datfid-master` does not proxy the new job routes yet.

## Production Path

A production version should use:

- Redis or managed queue for durable jobs.
- Postgres for job metadata.
- Object storage for files, artifacts, models, and results.
- ProcessPoolExecutor, Celery/RQ, separate worker containers, or Kubernetes jobs for CPU work.
- Tenant-scoped cache and job authorization.
- True work stealing after workers/artifacts are externalized.
