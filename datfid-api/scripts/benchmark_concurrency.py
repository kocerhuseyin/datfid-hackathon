#!/usr/bin/env python3
"""Compare legacy synchronous DATFID fit requests with job-based fit requests."""

import argparse
import csv
import io
import json
import math
import mimetypes
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import requests


FIT_FORM = {
    "id_col": "entity_id",
    "time_col": "date",
    "y": "y",
    "lag_y": "",
    "lagged_features": "",
    "current_features": '["x1","x2"]',
    "filter_by_significance": "false",
    "meanvar_test": "false",
    "signif": "0.05",
}


@dataclass
class UploadSpec:
    filename: str
    content: bytes
    content_type: str
    source: str


def generate_panel_csv(entities: int, periods: int) -> bytes:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["entity_id", "date", "y", "x1", "x2"])
    writer.writeheader()
    start = date(2024, 1, 1)
    for entity in range(entities):
        entity_bias = float(entity) * 2.0
        for period in range(periods):
            x1 = float(period) + entity * 0.1
            x2 = float((period % 7) - 3)
            y = 10.0 + entity_bias + 0.35 * period + 1.7 * x1 - 0.8 * x2
            writer.writerow({
                "entity_id": f"entity_{entity}",
                "date": (start + timedelta(days=period)).isoformat(),
                "y": f"{y:.6f}",
                "x1": f"{x1:.6f}",
                "x2": f"{x2:.6f}",
            })
    return out.getvalue().encode("utf-8")


def content_type_for_path(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "text/csv"
    if path.suffix.lower() == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if path.suffix.lower() == ".xls":
        return "application/vnd.ms-excel"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def build_upload_spec(args: argparse.Namespace) -> UploadSpec:
    if args.train_file:
        path = Path(args.train_file)
        if not path.exists():
            raise FileNotFoundError(f"Training file not found: {path}")
        return UploadSpec(
            filename=path.name,
            content=path.read_bytes(),
            content_type=args.file_content_type or content_type_for_path(path),
            source=str(path),
        )

    return UploadSpec(
        filename="synthetic.csv",
        content=generate_panel_csv(args.entities, args.periods),
        content_type="text/csv",
        source=f"synthetic:{args.entities}x{args.periods}",
    )


def build_fit_form(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "id_col": args.id_col,
        "time_col": args.time_col,
        "y": args.target,
        "lag_y": args.lag_y,
        "lagged_features": args.lagged_features,
        "current_features": args.current_features,
        "filter_by_significance": str(args.filter_by_significance).lower(),
        "meanvar_test": str(args.meanvar_test).lower(),
        "signif": args.signif,
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[index]


def summarize(name: str, latencies: List[float], successes: int, failures: int, wall_time: float) -> Dict[str, Any]:
    return {
        "scenario": name,
        "successes": successes,
        "failures": failures,
        "wall_time_seconds": round(wall_time, 4),
        "average_latency_seconds": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p50_latency_seconds": round(percentile(latencies, 50), 4),
        "p95_latency_seconds": round(percentile(latencies, 95), 4),
    }


def post_fit_sync(base_url: str, upload: UploadSpec, fit_form: Dict[str, str], timeout: float) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        with requests.Session() as session:
            response = session.post(
                f"{base_url}/modelfit-file/",
                files={"file": (upload.filename, upload.content, upload.content_type)},
                data=fit_form,
                timeout=timeout,
            )
        latency = time.perf_counter() - started
        return {"success": response.status_code == 200, "latency": latency, "status_code": response.status_code}
    except Exception as exc:
        return {"success": False, "latency": time.perf_counter() - started, "error": str(exc)}


def post_fit_job(
    base_url: str,
    upload: UploadSpec,
    fit_form: Dict[str, str],
    timeout: float,
    poll_interval: float,
    job_timeout: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    submit_started = time.perf_counter()
    try:
        with requests.Session() as session:
            response = session.post(
                f"{base_url}/jobs/fit-file",
                files={"file": (upload.filename, upload.content, upload.content_type)},
                data=fit_form,
                timeout=timeout,
            )
            submit_latency = time.perf_counter() - submit_started
            if response.status_code != 202:
                return {
                    "accepted": False,
                    "success": False,
                    "submit_latency": submit_latency,
                    "complete_latency": time.perf_counter() - started,
                    "status_code": response.status_code,
                    "error": response.text[:300],
                }

            job_id = response.json()["job_id"]
            deadline = time.perf_counter() + job_timeout
            while time.perf_counter() < deadline:
                status_response = session.get(f"{base_url}/jobs/{job_id}", timeout=timeout)
                if status_response.status_code != 200:
                    return {
                        "accepted": True,
                        "success": False,
                        "submit_latency": submit_latency,
                        "complete_latency": time.perf_counter() - started,
                        "status_code": status_response.status_code,
                        "error": status_response.text[:300],
                    }
                status_payload = status_response.json()
                if status_payload["status"] == "succeeded":
                    result_response = session.get(f"{base_url}/jobs/{job_id}/fit-result", timeout=timeout)
                    return {
                        "accepted": True,
                        "success": result_response.status_code == 200,
                        "submit_latency": submit_latency,
                        "complete_latency": time.perf_counter() - started,
                        "status_code": result_response.status_code,
                    }
                if status_payload["status"] == "failed":
                    return {
                        "accepted": True,
                        "success": False,
                        "submit_latency": submit_latency,
                        "complete_latency": time.perf_counter() - started,
                        "status_code": 500,
                        "error": str(status_payload.get("error", ""))[:300],
                    }
                time.sleep(poll_interval)

            return {
                "accepted": True,
                "success": False,
                "submit_latency": submit_latency,
                "complete_latency": time.perf_counter() - started,
                "error": "Timed out waiting for job completion.",
            }
    except Exception as exc:
        now = time.perf_counter()
        return {
            "accepted": False,
            "success": False,
            "submit_latency": now - submit_started,
            "complete_latency": now - started,
            "error": str(exc),
        }


def run_concurrent(count: int, concurrency: int, fn) -> Dict[str, Any]:
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(fn) for _ in range(count)]
        for future in as_completed(futures):
            results.append(future.result())
    return {"wall_time": time.perf_counter() - started, "results": results}


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "scenario",
        "successes",
        "failures",
        "wall_time_seconds",
        "average_latency_seconds",
        "p50_latency_seconds",
        "p95_latency_seconds",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("DATFID_BASE_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--train-file", default="", help="Optional CSV/XLSX file to upload instead of generated synthetic data.")
    parser.add_argument("--file-content-type", default="", help="Override uploaded file content type.")
    parser.add_argument("--id-col", default=FIT_FORM["id_col"])
    parser.add_argument("--time-col", default=FIT_FORM["time_col"])
    parser.add_argument("--target", default=FIT_FORM["y"])
    parser.add_argument("--lag-y", default=FIT_FORM["lag_y"])
    parser.add_argument("--lagged-features", default=FIT_FORM["lagged_features"])
    parser.add_argument("--current-features", default=FIT_FORM["current_features"])
    parser.add_argument("--filter-by-significance", default=FIT_FORM["filter_by_significance"])
    parser.add_argument("--meanvar-test", default=FIT_FORM["meanvar_test"])
    parser.add_argument("--signif", default=FIT_FORM["signif"])
    parser.add_argument("--entities", type=int, default=20)
    parser.add_argument("--periods", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--job-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    upload = build_upload_spec(args)
    fit_form = build_fit_form(args)

    sync_run = run_concurrent(
        args.requests,
        args.concurrency,
        lambda: post_fit_sync(base_url, upload, fit_form, args.timeout),
    )
    sync_results = sync_run["results"]
    sync_latencies = [item["latency"] for item in sync_results]
    rows = [
        summarize(
            "old_sync_fit",
            sync_latencies,
            sum(1 for item in sync_results if item["success"]),
            sum(1 for item in sync_results if not item["success"]),
            sync_run["wall_time"],
        )
    ]

    job_run = run_concurrent(
        args.requests,
        args.concurrency,
        lambda: post_fit_job(base_url, upload, fit_form, args.timeout, args.poll_interval, args.job_timeout),
    )
    job_results = job_run["results"]
    submit_latencies = [item["submit_latency"] for item in job_results]
    complete_latencies = [item["complete_latency"] for item in job_results]
    rows.append(
        summarize(
            "new_job_submit",
            submit_latencies,
            sum(1 for item in job_results if item.get("accepted")),
            sum(1 for item in job_results if not item.get("accepted")),
            job_run["wall_time"],
        )
    )
    rows.append(
        summarize(
            "new_job_complete",
            complete_latencies,
            sum(1 for item in job_results if item["success"]),
            sum(1 for item in job_results if not item["success"]),
            job_run["wall_time"],
        )
    )

    print_table(rows)

    if args.json_output:
        payload = {
            "base_url": base_url,
            "upload_source": upload.source,
            "upload_filename": upload.filename,
            "upload_content_type": upload.content_type,
            "fit_form": fit_form,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "entities": args.entities,
            "periods": args.periods,
            "rows": rows,
            "sync_errors": [item.get("error") for item in sync_results if item.get("error")],
            "job_errors": [item.get("error") for item in job_results if item.get("error")],
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
