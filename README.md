# DATFID CPU Inference Orchestration

Solo hackathon solution for the **Campus Founders AI Hackathon** challenge:
**Scaling CPU inference to concurrent users for DATFID**.

## Impact

- Achieved **~90% faster API response** for model-fit requests by returning a `job_id` immediately instead of blocking until completion.
- Improved **cold end-to-end fit completion by ~38%** on the benchmark workload through bounded job orchestration.
- Improved **repeated-work completion by ~85%** using reusable intermediate artifacts/cache.
- Added controlled CPU execution so concurrent requests are admitted through a bounded worker system.
- Added job tracking so clients can poll `queued`, `running`, `succeeded`, or `failed` status.

## What I Built

- Preserved the original DATFID endpoints for compatibility.
- Added job-based fit and forecast endpoints in FastAPI.
- Split model fitting into internal stages: parse file, parse parameters, fit model, generate report.
- Tied forecasts to a specific `fit_job_id` so workflows are easier to track.
- Added `/metrics` for queue, job, stage, and cache visibility.

## Benchmark

```text
Cold run:
original blocking fit ~5.91s -> new job completion ~3.65s

Warm run:
original blocking fit ~5.58s -> new job completion ~0.85s

API response:
original blocking fit ~5.91s -> new job submit ~0.59s
```

## Tech / Architecture

- FastAPI backend with in-memory job and artifact stores.
- Bounded worker execution using Python thread workers.
- Internal DAG-style stages for fit and forecast workflows.
- Reusable cache/materialization layer for repeated work.
- Hackathon prototype; production would use durable storage, process/external workers, and a real queue.

## Links

- Runtime API: `https://huseyinkocer-datfid-api.hf.space`
- Backend Space: `https://huggingface.co/spaces/huseyinkocer/datfid-api`
- Metrics: `https://huseyinkocer-datfid-api.hf.space/metrics`

## Repository

- `datfid-api`: FastAPI backend with job orchestration, workers, stages, cache, and metrics.
- `datfid-master`: DATFID-style public proxy/demo layer.
- `pitch`: slides, speaker script, animations, GIFs, and demo commands.
