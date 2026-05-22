# main.py
import hashlib, json, os, queue, threading, traceback, uuid
import pandas as pd
import numpy as np
import io, tempfile, textwrap
import statsmodels.api as sm

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, Response, status, Request
from fastapi.responses import FileResponse, PlainTextResponse, ORJSONResponse
from fastapi.concurrency import run_in_threadpool
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Predefined globals
stored_model = None
stored_result_join = None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DATFID_MAX_WORKERS = _env_int("DATFID_MAX_WORKERS", 2)
DATFID_MAX_QUEUE_SIZE = _env_int("DATFID_MAX_QUEUE_SIZE", 16)
DATFID_MAX_STORED_JOBS = _env_int("DATFID_MAX_STORED_JOBS", 100)
DATFID_DAG_CACHE_ENABLED = _env_bool("DATFID_DAG_CACHE_ENABLED", True)
DATFID_MAX_MATERIALIZED_ARTIFACTS = _env_int("DATFID_MAX_MATERIALIZED_ARTIFACTS", 128)

DATFID_PARSE_VERSION = "parse-v1"
DATFID_MODEL_VERSION = "demo-ols-v1"
DATFID_FORECAST_VERSION = "forecast-v1"

STAGE_FIT_PARSE_FILE = "fit_parse_file"
STAGE_FIT_PARSE_PARAMS = "fit_parse_params"
STAGE_FIT_MODEL = "fit_model"
STAGE_FIT_REPORT = "fit_report"
STAGE_FORECAST_PARSE_FILE = "forecast_parse_file"
STAGE_FORECAST_RUN = "forecast_run"

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_FIT = "fit"
JOB_FORECAST = "forecast"

@dataclass
class DemoFitResult:
    formula: str
    alpha: List[List[float]]
    beta: List[List[float]]
    headers_alpha: List[str]
    headers_beta: List[str]
    Performance: List[List[float]]
    R2_individual: List[float]
    R2_individual_labels: List[str]

@dataclass
class DemoFitJoin:
    result: DemoFitResult


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    fit_job_id: Optional[str] = None
    result_url: str = ""
    final_node_id: Optional[str] = None
    node_ids: List[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class FitArtifact:
    model: Any
    result_join: DemoFitJoin
    report_text: str
    params: Dict[str, Any]
    created_at: datetime
    lock: Any = field(default_factory=threading.Lock)
    model_artifact_key: str = ""


@dataclass
class ForecastArtifact:
    csv_bytes: bytes
    filename: str
    media_type: str
    created_at: datetime


@dataclass
class DagNodeRecord:
    node_id: str
    job_id: str
    stage: str
    status: str
    depends_on: List[str]
    payload: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None
    output_artifact_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    ready_enqueued: bool = False


@dataclass
class MaterializedArtifact:
    artifact_key: str
    kind: str
    value: Any
    created_at: datetime
    last_accessed_at: datetime
    source_job_id: str
    source_node_id: str


@dataclass
class JobMetrics:
    jobs_submitted: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    rejected_jobs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    durations_seconds: List[float] = field(default_factory=list)

class DATFIDModel:
    """
    Demo-safe replacement for the private DATFID model.
    Uses simple OLS with optional lagged features and trend.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        y: str,
        lag_y: Any = None,
        lagged_features: Any = None,
        current_features: Any = None,
        filter_by_significance: bool = False,
        meanvar_test: bool = False,
        signif: float = 0.05,
    ):
        self.df = df.copy()
        self.id_col = id_col
        self.time_col = time_col
        self.y = y
        self.lag_y = lag_y
        self.lagged_features = lagged_features if isinstance(lagged_features, dict) else {}
        if current_features == "all":
            self.current_features = "all"
        elif isinstance(current_features, list):
            self.current_features = current_features
        else:
            self.current_features = []
        self.filter_by_significance = filter_by_significance
        self.meanvar_test = meanvar_test
        self.signif = signif
        self._fitted_model = None
        self._train_columns: List[str] = []
        self._last_y_by_id: Dict[str, float] = {}
        self._last_y_global: float = 0.0

    def _get_lag_int(self, value: Any, default: int = 1) -> int:
        try:
            out = int(value)
            return out if out > 0 else default
        except Exception:
            return default

    def _sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.id_col in df.columns and self.time_col in df.columns:
            return df.sort_values([self.id_col, self.time_col]).copy()
        if self.time_col in df.columns:
            return df.sort_values([self.time_col]).copy()
        return df.copy()

    def _trend(self, df: pd.DataFrame) -> pd.Series:
        if self.time_col in df.columns:
            ts = pd.to_datetime(df[self.time_col], errors="coerce")
            if ts.notna().any():
                base = ts.min()
                return (ts - base).dt.days.fillna(0.0).astype(float)
        return pd.Series(np.arange(len(df), dtype=float), index=df.index)

    def _resolve_current_features(self, df: pd.DataFrame) -> List[str]:
        if self.current_features == "all":
            return [
                c for c in df.columns
                if c not in {self.id_col, self.time_col, self.y}
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        return [c for c in self.current_features if c in df.columns]

    def _build_matrix(self, df: pd.DataFrame, is_forecast: bool) -> pd.DataFrame:
        work = self._sorted(df)
        x = pd.DataFrame(index=work.index)
        x["trend_index"] = self._trend(work)

        for col in self._resolve_current_features(work):
            x[col] = pd.to_numeric(work[col], errors="coerce")

        if self.y in work.columns:
            lag_y_int = self._get_lag_int(self.lag_y, default=1) if self.lag_y else None
            if lag_y_int:
                lag_name = f"{self.y}_lag_{lag_y_int}"
                if self.id_col in work.columns:
                    x[lag_name] = work.groupby(self.id_col, sort=False)[self.y].shift(lag_y_int)
                else:
                    x[lag_name] = work[self.y].shift(lag_y_int)

        for feat, lag in (self.lagged_features or {}).items():
            if feat not in work.columns:
                continue
            lag_int = self._get_lag_int(lag, default=1)
            col_name = f"{feat}_lag_{lag_int}"
            if self.id_col in work.columns:
                x[col_name] = work.groupby(self.id_col, sort=False)[feat].shift(lag_int)
            else:
                x[col_name] = work[feat].shift(lag_int)

        x = x.apply(pd.to_numeric, errors="coerce")
        if is_forecast:
            x = x.fillna(0.0)
        return x

    def fit(self) -> DemoFitJoin:
        if self.y not in self.df.columns:
            raise ValueError(f"Target column '{self.y}' is missing.")

        train = self._sorted(self.df)
        x = self._build_matrix(train, is_forecast=False)
        y = pd.to_numeric(train[self.y], errors="coerce")
        valid = y.notna()
        if x.shape[1] > 0:
            valid = valid & x.notna().all(axis=1)
        x = x.loc[valid].copy()
        y = y.loc[valid].copy()

        if len(y) < 3:
            raise ValueError("Not enough valid rows to fit demo model (need >= 3).")

        x_const = sm.add_constant(x, has_constant="add")
        self._fitted_model = sm.OLS(y.astype(float), x_const.astype(float)).fit()
        self._train_columns = list(x_const.columns)

        if self.id_col in train.columns:
            last_vals = train.groupby(self.id_col, sort=False)[self.y].last().dropna()
            self._last_y_by_id = {str(k): float(v) for k, v in last_vals.items()}
        self._last_y_global = float(y.iloc[-1]) if len(y) else 0.0

        params = self._fitted_model.params
        bse = self._fitted_model.bse
        tvals = self._fitted_model.tvalues
        pvals = self._fitted_model.pvalues

        const_name = "const" if "const" in params.index else params.index[0]
        beta_names = [n for n in params.index if n != const_name]

        alpha = [[float(params.get(const_name, 0.0))], [float(bse.get(const_name, 0.0))], [float(tvals.get(const_name, 0.0))], [float(pvals.get(const_name, 1.0))]]
        beta = [
            [float(params.get(n, 0.0)) for n in beta_names],
            [float(bse.get(n, 0.0)) for n in beta_names],
            [float(tvals.get(n, 0.0)) for n in beta_names],
            [float(pvals.get(n, 1.0)) for n in beta_names],
        ]

        pred = self._fitted_model.predict(x_const)
        mse = float(np.mean((y - pred) ** 2))
        mae = float(np.mean(np.abs(y - pred)))
        r2 = float(getattr(self._fitted_model, "rsquared", 0.0))
        r2_adj = float(getattr(self._fitted_model, "rsquared_adj", r2))
        perf = [
            [r2, r2],
            [r2_adj, r2_adj],
            [r2, r2_adj],
            [mse, mse],
            [mae, mae],
        ]

        r2_individual: List[float] = []
        r2_labels: List[str] = []
        if self.id_col in train.columns:
            joined = pd.DataFrame({
                self.id_col: train.loc[valid, self.id_col].astype(str),
                "_y": y.values,
                "_p": pred.values,
            })
            for id_val, sub in joined.groupby(self.id_col, sort=False):
                den = float(((sub["_y"] - sub["_y"].mean()) ** 2).sum())
                if den <= 0:
                    r2_i = 0.0
                else:
                    num = float(((sub["_y"] - sub["_p"]) ** 2).sum())
                    r2_i = 1.0 - (num / den)
                r2_labels.append(str(id_val))
                r2_individual.append(r2_i)

        formula = f"{self.y} ~ " + " + ".join(self._train_columns)
        fit_result = DemoFitResult(
            formula=formula,
            alpha=alpha,
            beta=beta,
            headers_alpha=[const_name],
            headers_beta=beta_names,
            Performance=perf,
            R2_individual=r2_individual,
            R2_individual_labels=r2_labels,
        )
        return DemoFitJoin(result=fit_result)

    def forecast(self, extern_self: Any, df_forecast: pd.DataFrame) -> pd.DataFrame:
        if self._fitted_model is None:
            raise ValueError("Model not fitted.")

        out = self._sorted(df_forecast).copy()
        x = self._build_matrix(out, is_forecast=True)
        x_const = sm.add_constant(x, has_constant="add")

        for col in self._train_columns:
            if col not in x_const.columns:
                x_const[col] = 0.0
        x_const = x_const[self._train_columns].astype(float)

        pred = self._fitted_model.predict(x_const)
        out[f"{self.y}_forecast"] = np.asarray(pred, dtype=float)
        out["forecast"] = out[f"{self.y}_forecast"]
        return out

def _maybe_json_list(s: Optional[str]):
    if s is None or s == "":
        return []
    s = s.strip()
    if s.lower() == "all":
        return "all"
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return val
        return []
    except Exception:
        # allow comma-separated as a fallback
        return [x.strip() for x in s.split(",") if x.strip()]

def _maybe_json_dict(s: Optional[str]):
    if not s:
        return {}
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}
    
def _read_table_from_upload(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    data = upload.file.read()
    return _read_table_from_bytes(name, data)

def _read_table_from_bytes(filename: str, data: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(data)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    # default to Excel (supports .xls, .xlsx)
    return pd.read_excel(bio)

def _normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve the existing file endpoint behavior: mutate and return the same frame.
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

def _parse_fit_file_params(
    lag_y: str,
    lagged_features: str,
    current_features: str,
    filter_by_significance: str,
    meanvar_test: str,
    signif: str,
) -> Dict[str, Any]:
    lagged = _maybe_json_dict(lagged_features)
    curr = _maybe_json_list(current_features)
    filt_sig = str(filter_by_significance).strip().lower() == "true"
    mv_test = str(meanvar_test).strip().lower() == "true"
    ly = None if (lag_y is None or lag_y.strip() == "") else lag_y.strip()
    try:
        sig_val = float(signif)
    except (TypeError, ValueError):
        sig_val = 0.05
    return {
        "lag_y": ly,
        "lagged_features": lagged,
        "current_features": curr,
        "filter_by_significance": filt_sig,
        "meanvar_test": mv_test,
        "signif": sig_val,
    }

def _fit_model_from_dataframe(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    y: str,
    params: Dict[str, Any],
) -> tuple:
    model = DATFIDModel(
        df=df,
        id_col=id_col,
        time_col=time_col,
        y=y,
        lag_y=params["lag_y"],
        lagged_features=params["lagged_features"],
        current_features=params["current_features"],
        filter_by_significance=params["filter_by_significance"],
        meanvar_test=params["meanvar_test"],
        signif=params["signif"],
    )
    result_join = model.fit()
    return model, result_join

def _forecast_dataframe_to_csv_bytes(forecast_df: pd.DataFrame) -> bytes:
    # Keep output equivalent to /modelforecast-file/.
    buf = io.StringIO()
    forecast_df.to_csv(buf, index=False)
    csv_text = "sep=,\n" + buf.getvalue()
    return csv_text.encode("utf-8-sig")

def _write_text_attachment_tmp(text: str, suffix: str = ".txt") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(text.encode("utf-8"))
    tmp.flush(); tmp.close()
    return tmp.name

def _write_bytes_attachment_tmp(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush(); tmp.close()
    return tmp.name

def _result_to_text(result_obj: Any) -> str:
    """
    Build a compact, readable DATFID model summary.

    Expected fields in `result_obj` (object with attributes or a dict):
      - formula: str
      - alpha: 2D array-like (rows ~ [Estimate, SE, T, P], columns = time-invariant features)
      - beta:  2D array-like (rows ~ [Estimate, SE, T, P], columns = time-variant features)
      - headers_alpha: list[str] (names for alpha columns)
      - headers_beta:  list[str] (names for beta columns)
      - Performance: 2D array-like with the last 5 rows (in this order):
            R2 within, R2 between, R2 overall, MSE, MAE
        and 2 columns:
            2SFE, 2SFE_c
      - R2_individual: 1D array-like of per-individual R² (optional)
      - R2_individual_labels: list[str] of same length as R2_individual (optional)
        If provided, labels will be shown instead of ID numbers in the summary.

    The function tolerates:
      - dict or object input
      - missing headers (falls back to generic names)
      - extra rows in alpha/beta (truncated to 4)
      - extra rows in Performance (only last 5 kept)
    """

    # --- Safe getters for both dicts and objects
    def get(key, default=None):
        if isinstance(result_obj, dict):
            return result_obj.get(key, default)
        return getattr(result_obj, key, default)

    def as_2d(a):
        if a is None:
            return np.empty((0, 0), dtype=float)
        arr = np.asarray(a, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    # --- Pull fields
    formula = (get("formula", "") or "").strip()

    alpha_arr = as_2d(get("alpha"))
    beta_arr  = as_2d(get("beta"))
    perf_arr  = as_2d(get("Performance"))

    headers_alpha = list(get("headers_alpha", [])) or [f"Alpha_{i+1}" for i in range(alpha_arr.shape[1])]
    headers_beta  = list(get("headers_beta", []))  or [f"Beta_{i+1}"  for i in range(beta_arr.shape[1])]

    r2_individual = get("R2_individual", None)
    r2_labels     = get("R2_individual_labels", None)

    # --- Shape/label guards
    row_labels = ["Estimate", "Standard Error", "T statistic", "P value"]

    if alpha_arr.shape[0] > 4:
        alpha_arr = alpha_arr[:4, :]
    if beta_arr.shape[0] > 4:
        beta_arr = beta_arr[:4, :]

    # If perf has >5 rows, keep the last 5 (assumes metrics are at the end)
    if perf_arr.shape[0] >= 5:
        perf_arr = perf_arr[-5:, :]
    perf_rows = ["R2 within", "R2 between", "R2 overall", "MSE", "MAE"]
    perf_cols = ["2SFE", "2SFE_c"]
    # Guard columns
    n_perf_cols = min(perf_arr.shape[1], 2)
    perf_cols = perf_cols[:n_perf_cols]

    # --- Build tables
    alpha_df = pd.DataFrame(alpha_arr, index=row_labels[:alpha_arr.shape[0]],
                            columns=headers_alpha[:alpha_arr.shape[1]])
    beta_df  = pd.DataFrame(beta_arr,  index=row_labels[:beta_arr.shape[0]],
                            columns=headers_beta[:beta_arr.shape[1]])
    perf_df  = pd.DataFrame(perf_arr,  index=perf_rows[:perf_arr.shape[0]],
                            columns=perf_cols)

    # --- R² summary (min/median/max)
    r2_lines = []
    if r2_individual is not None:
        r2_vals = np.asarray(r2_individual, dtype=float).ravel()
        if r2_vals.size > 0 and np.isfinite(r2_vals).any():
            # Prepare labels (either provided or fallback to 1-based IDs)
            if r2_labels and len(r2_labels) == r2_vals.size:
                ids = np.array(list(r2_labels), dtype=object)
                def _lab(idx): return str(ids[idx])
            else:
                def _lab(idx): return f"ID {idx+1}"

            r2_min = float(np.nanmin(r2_vals))
            r2_med = float(np.nanmedian(r2_vals))
            r2_max = float(np.nanmax(r2_vals))

            # Nearest index to each statistic (handles non-exact medians)
            imin = int(np.nanargmin(r2_vals))
            imed = int(np.nanargmin(np.abs(r2_vals - r2_med)))
            imax = int(np.nanargmax(r2_vals))

            r2_lines = [
                f"min ({_lab(imin)}): {r2_min:.6g}",
                f"median ({_lab(imed)}): {r2_med:.6g}",
                f"max ({_lab(imax)}): {r2_max:.6g}",
            ]

    # --- Pretty printers
    ffmt = lambda x: f"{x:.6g}"
    def _df_text(df: pd.DataFrame) -> str:
        # Right-justified columns; scientific format where appropriate
        return df.to_string(justify="right", float_format=ffmt)

    # --- Compose report
    parts = []
    parts.append("DATFID Fit Result")
    parts.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    parts.append("=" * 72)
    parts.append("=== Model Summary ===\n")
    parts.append("Formula:")
    parts.append(f" {formula}\n")

    parts.append("Alpha (time invariant):")
    parts.append(_df_text(alpha_df) + "\n")

    parts.append("Beta (time variant):")
    parts.append(_df_text(beta_df) + "\n")

    parts.append("Performance metrics:")
    parts.append(_df_text(perf_df) + "\n")

    if r2_lines:
        parts.append("Individual R² summary:")
        parts.extend(r2_lines)

    return ("\n".join(parts)).rstrip() + "\n"


# ---------- In-memory DAG job orchestration ----------
_store_lock = threading.RLock()
_ready_node_queue: queue.Queue = queue.Queue()
_jobs: Dict[str, JobRecord] = {}
_dag_nodes: Dict[str, DagNodeRecord] = {}
_materialized_artifacts: Dict[str, MaterializedArtifact] = {}
_cache_index: Dict[str, str] = {}
_fit_artifacts: Dict[str, FitArtifact] = {}
_forecast_artifacts: Dict[str, ForecastArtifact] = {}
_metrics = JobMetrics()
_workers: List[threading.Thread] = []
_workers_started = False
_shutdown_event = threading.Event()


def _utcnow() -> datetime:
    return datetime.utcnow()


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat() + "Z"


def _job_duration_seconds(job: JobRecord, now: Optional[datetime] = None) -> Optional[float]:
    if job.started_at is None:
        return None
    end = job.finished_at or (now if job.status == JOB_RUNNING else None)
    if end is None:
        return None
    return round((end - job.started_at).total_seconds(), 6)


def _job_to_dict(job: JobRecord) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "created_at": _iso_or_none(job.created_at),
        "started_at": _iso_or_none(job.started_at),
        "finished_at": _iso_or_none(job.finished_at),
        "duration_seconds": _job_duration_seconds(job, now=_utcnow()),
        "error": job.error,
        "fit_job_id": job.fit_job_id,
        "status_url": f"/jobs/{job.job_id}",
        "result_url": job.result_url,
        "final_node_id": job.final_node_id,
        "dag_node_count": len(job.node_ids),
        "cache_hits": job.cache_hits,
        "cache_misses": job.cache_misses,
    }


def _node_id(job_id: str, stage: str) -> str:
    return f"{job_id}:{stage}"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _cache_key(stage: str, identity: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return f"{stage}:{digest}"


def _file_parse_cache_key(stage: str, filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        extension = ".csv"
    elif name.endswith(".xls"):
        extension = ".xls"
    elif name.endswith(".xlsx"):
        extension = ".xlsx"
    else:
        extension = "excel-default"
    return _cache_key(stage, {
        "version": DATFID_PARSE_VERSION,
        "extension": extension,
        "content_sha256": _hash_bytes(data),
    })


def _make_artifact_key(kind: str, cache_key: Optional[str] = None) -> str:
    if cache_key:
        return f"artifact:{cache_key}"
    return f"artifact:{kind}:{uuid.uuid4().hex}"


def _active_job_count_locked() -> int:
    return sum(1 for job in _jobs.values() if job.status in {JOB_QUEUED, JOB_RUNNING})


def _record_duration_locked(duration_seconds: Optional[float]) -> None:
    if duration_seconds is None:
        return
    _metrics.durations_seconds.append(duration_seconds)
    max_durations = max(DATFID_MAX_STORED_JOBS, 100)
    if len(_metrics.durations_seconds) > max_durations:
        del _metrics.durations_seconds[:len(_metrics.durations_seconds) - max_durations]


def _protected_materialized_artifact_keys_locked() -> set:
    protected = set()
    active_job_ids = {
        job.job_id
        for job in _jobs.values()
        if job.status in {JOB_QUEUED, JOB_RUNNING}
    }
    for node in _dag_nodes.values():
        if node.job_id in active_job_ids and node.output_artifact_key:
            protected.add(node.output_artifact_key)

    for job in _jobs.values():
        if job.kind == JOB_FORECAST and job.status in {JOB_QUEUED, JOB_RUNNING} and job.fit_job_id:
            fit_artifact = _fit_artifacts.get(job.fit_job_id)
            if fit_artifact and fit_artifact.model_artifact_key:
                protected.add(fit_artifact.model_artifact_key)
    return protected


def _evict_old_materialized_artifacts_locked() -> None:
    if len(_materialized_artifacts) <= DATFID_MAX_MATERIALIZED_ARTIFACTS:
        return

    protected = _protected_materialized_artifact_keys_locked()
    candidates = [
        artifact for key, artifact in _materialized_artifacts.items()
        if key not in protected
    ]
    candidates.sort(key=lambda artifact: artifact.last_accessed_at)

    for artifact in candidates:
        if len(_materialized_artifacts) <= DATFID_MAX_MATERIALIZED_ARTIFACTS:
            break
        _materialized_artifacts.pop(artifact.artifact_key, None)
        for cache_key, artifact_key in list(_cache_index.items()):
            if artifact_key == artifact.artifact_key:
                _cache_index.pop(cache_key, None)


def _evict_old_completed_jobs_locked() -> None:
    if len(_jobs) <= DATFID_MAX_STORED_JOBS:
        return

    protected_fit_ids = {
        job.fit_job_id
        for job in _jobs.values()
        if job.kind == JOB_FORECAST and job.status in {JOB_QUEUED, JOB_RUNNING} and job.fit_job_id
    }

    candidates = [
        job for job in _jobs.values()
        if job.status in {JOB_SUCCEEDED, JOB_FAILED} and job.job_id not in protected_fit_ids
    ]
    candidates.sort(key=lambda job: job.finished_at or job.created_at)

    for job in candidates:
        if len(_jobs) <= DATFID_MAX_STORED_JOBS:
            break
        _jobs.pop(job.job_id, None)
        _fit_artifacts.pop(job.job_id, None)
        _forecast_artifacts.pop(job.job_id, None)
        for node_id in job.node_ids:
            _dag_nodes.pop(node_id, None)


def _store_artifact_locked(
    kind: str,
    value: Any,
    source_job_id: str,
    source_node_id: str,
    cache_key: Optional[str] = None,
) -> str:
    now = _utcnow()
    artifact_key = _make_artifact_key(kind, cache_key if DATFID_DAG_CACHE_ENABLED else None)
    artifact = MaterializedArtifact(
        artifact_key=artifact_key,
        kind=kind,
        value=value,
        created_at=now,
        last_accessed_at=now,
        source_job_id=source_job_id,
        source_node_id=source_node_id,
    )
    _materialized_artifacts[artifact_key] = artifact
    if DATFID_DAG_CACHE_ENABLED and cache_key:
        _cache_index[cache_key] = artifact_key
    return artifact_key


def _get_cached_artifact_key_locked(cache_key: Optional[str]) -> Optional[str]:
    if not DATFID_DAG_CACHE_ENABLED or not cache_key:
        return None
    artifact_key = _cache_index.get(cache_key)
    artifact = _materialized_artifacts.get(artifact_key or "")
    if artifact is None:
        if artifact_key:
            _cache_index.pop(cache_key, None)
        return None
    artifact.last_accessed_at = _utcnow()
    return artifact_key


def _get_artifact_value(artifact_key: Optional[str]) -> Any:
    if not artifact_key:
        raise RuntimeError("Missing upstream artifact.")
    with _store_lock:
        artifact = _materialized_artifacts.get(artifact_key)
        if artifact is None:
            raise RuntimeError(f"Materialized artifact is no longer available: {artifact_key}")
        artifact.last_accessed_at = _utcnow()
        return artifact.value


def _dependency_artifact_key(node: DagNodeRecord, stage: str) -> str:
    with _store_lock:
        for dep_id in node.depends_on:
            dep = _dag_nodes.get(dep_id)
            if dep and dep.stage == stage and dep.output_artifact_key:
                return dep.output_artifact_key
    raise RuntimeError(f"Missing dependency artifact for stage {stage}.")


def _enqueue_ready_nodes_locked(job_id: str) -> None:
    job = _jobs.get(job_id)
    if job is None or job.status == JOB_FAILED:
        return
    for node_id in job.node_ids:
        node = _dag_nodes.get(node_id)
        if node is None or node.status != JOB_QUEUED or node.ready_enqueued:
            continue
        dependencies_ready = all(
            (_dag_nodes.get(dep_id) is not None and _dag_nodes[dep_id].status == JOB_SUCCEEDED)
            for dep_id in node.depends_on
        )
        if dependencies_ready:
            node.ready_enqueued = True
            _ready_node_queue.put(node.node_id)


def _build_fit_dag(job_id: str, payload: Dict[str, Any]) -> List[DagNodeRecord]:
    file_cache_key = _file_parse_cache_key(
        STAGE_FIT_PARSE_FILE,
        payload["filename"],
        payload["file_bytes"],
    )
    parsed_params_for_key = _parse_fit_file_params(
        payload["lag_y"],
        payload["lagged_features"],
        payload["current_features"],
        payload["filter_by_significance"],
        payload["meanvar_test"],
        payload["signif"],
    )
    params_identity = {
        "id_col": payload["id_col"],
        "time_col": payload["time_col"],
        "y": payload["y"],
        "params": parsed_params_for_key,
    }
    model_cache_key = _cache_key(STAGE_FIT_MODEL, {
        "version": DATFID_MODEL_VERSION,
        "parsed_file": file_cache_key,
        **params_identity,
    })

    parse_file = DagNodeRecord(
        node_id=_node_id(job_id, STAGE_FIT_PARSE_FILE),
        job_id=job_id,
        stage=STAGE_FIT_PARSE_FILE,
        status=JOB_QUEUED,
        depends_on=[],
        payload={"filename": payload["filename"], "file_bytes": payload["file_bytes"]},
        cache_key=file_cache_key,
    )
    parse_params = DagNodeRecord(
        node_id=_node_id(job_id, STAGE_FIT_PARSE_PARAMS),
        job_id=job_id,
        stage=STAGE_FIT_PARSE_PARAMS,
        status=JOB_QUEUED,
        depends_on=[],
        payload={
            "id_col": payload["id_col"],
            "time_col": payload["time_col"],
            "y": payload["y"],
            "lag_y": payload["lag_y"],
            "lagged_features": payload["lagged_features"],
            "current_features": payload["current_features"],
            "filter_by_significance": payload["filter_by_significance"],
            "meanvar_test": payload["meanvar_test"],
            "signif": payload["signif"],
        },
    )
    fit_model = DagNodeRecord(
        node_id=_node_id(job_id, STAGE_FIT_MODEL),
        job_id=job_id,
        stage=STAGE_FIT_MODEL,
        status=JOB_QUEUED,
        depends_on=[parse_file.node_id, parse_params.node_id],
        payload=params_identity,
        cache_key=model_cache_key,
    )
    fit_report = DagNodeRecord(
        node_id=_node_id(job_id, STAGE_FIT_REPORT),
        job_id=job_id,
        stage=STAGE_FIT_REPORT,
        status=JOB_QUEUED,
        depends_on=[fit_model.node_id],
    )
    return [parse_file, parse_params, fit_model, fit_report]


def _build_forecast_dag(job_id: str, payload: Dict[str, Any], fit_artifact: FitArtifact) -> List[DagNodeRecord]:
    file_cache_key = _file_parse_cache_key(
        STAGE_FORECAST_PARSE_FILE,
        payload["filename"],
        payload["file_bytes"],
    )
    forecast_cache_key = _cache_key(STAGE_FORECAST_RUN, {
        "version": DATFID_FORECAST_VERSION,
        "model_artifact": fit_artifact.model_artifact_key or payload["fit_job_id"],
        "forecast_file": file_cache_key,
    })
    parse_file = DagNodeRecord(
        node_id=_node_id(job_id, STAGE_FORECAST_PARSE_FILE),
        job_id=job_id,
        stage=STAGE_FORECAST_PARSE_FILE,
        status=JOB_QUEUED,
        depends_on=[],
        payload={"filename": payload["filename"], "file_bytes": payload["file_bytes"]},
        cache_key=file_cache_key,
    )
    forecast_run = DagNodeRecord(
        node_id=_node_id(job_id, STAGE_FORECAST_RUN),
        job_id=job_id,
        stage=STAGE_FORECAST_RUN,
        status=JOB_QUEUED,
        depends_on=[parse_file.node_id],
        payload={"fit_job_id": payload["fit_job_id"]},
        cache_key=forecast_cache_key,
    )
    return [parse_file, forecast_run]


def _create_job(kind: str, payload: Dict[str, Any], fit_job_id: Optional[str] = None) -> JobRecord:
    prefix = "fit" if kind == JOB_FIT else "forecast"
    job_id = f"{prefix}_{uuid.uuid4().hex}"
    result_suffix = "fit-result" if kind == JOB_FIT else "forecast-result"
    job = JobRecord(
        job_id=job_id,
        kind=kind,
        status=JOB_QUEUED,
        created_at=_utcnow(),
        fit_job_id=fit_job_id,
        result_url=f"/jobs/{job_id}/{result_suffix}",
    )

    if kind == JOB_FIT:
        nodes = _build_fit_dag(job_id, payload)
    elif kind == JOB_FORECAST:
        with _store_lock:
            fit_artifact = _fit_artifacts.get(fit_job_id or "")
        if fit_artifact is None:
            raise HTTPException(status_code=410, detail="Fitted model is no longer available.")
        nodes = _build_forecast_dag(job_id, payload, fit_artifact)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown job kind: {kind}")

    with _store_lock:
        if _active_job_count_locked() >= DATFID_MAX_QUEUE_SIZE:
            _metrics.rejected_jobs += 1
            raise HTTPException(status_code=429, detail="Job queue is full. Try again later.")
        _jobs[job_id] = job
        job.node_ids = [node.node_id for node in nodes]
        job.final_node_id = nodes[-1].node_id
        for node in nodes:
            _dag_nodes[node.node_id] = node
        _metrics.jobs_submitted += 1
        _enqueue_ready_nodes_locked(job_id)

    return job


def _mark_node_running(node_id: str) -> Optional[DagNodeRecord]:
    with _store_lock:
        node = _dag_nodes.get(node_id)
        if node is None or node.status != JOB_QUEUED:
            return None
        job = _jobs.get(node.job_id)
        if job is None or job.status == JOB_FAILED:
            return None
        now = _utcnow()
        node.status = JOB_RUNNING
        node.started_at = now
        if job.status == JOB_QUEUED:
            job.status = JOB_RUNNING
            job.started_at = now
        return node


def _mark_node_succeeded(node_id: str, artifact_key: str) -> None:
    with _store_lock:
        node = _dag_nodes.get(node_id)
        if node is None:
            return
        node.status = JOB_SUCCEEDED
        node.finished_at = _utcnow()
        node.error = None
        node.output_artifact_key = artifact_key
        node.payload = {}

        job = _jobs.get(node.job_id)
        if job is None:
            return
        if job.final_node_id == node_id:
            job.status = JOB_SUCCEEDED
            job.finished_at = node.finished_at
            job.error = None
            _metrics.jobs_succeeded += 1
            _record_duration_locked(_job_duration_seconds(job))
            _evict_old_completed_jobs_locked()
            _evict_old_materialized_artifacts_locked()
        else:
            _enqueue_ready_nodes_locked(job.job_id)
            _evict_old_materialized_artifacts_locked()


def _mark_node_failed(node_id: str, error: str) -> None:
    with _store_lock:
        node = _dag_nodes.get(node_id)
        if node is not None:
            node.status = JOB_FAILED
            node.finished_at = _utcnow()
            node.error = error
            node.payload = {}
        job = _jobs.get(node.job_id) if node is not None else None
        if job is not None and job.status != JOB_FAILED:
            job.status = JOB_FAILED
            job.finished_at = _utcnow()
            job.error = error
            for other_id in job.node_ids:
                other = _dag_nodes.get(other_id)
                if other is not None and other.status == JOB_QUEUED:
                    other.status = JOB_FAILED
                    other.finished_at = job.finished_at
                    other.error = error
                    other.payload = {}
            _metrics.jobs_failed += 1
            _record_duration_locked(_job_duration_seconds(job))
            _evict_old_completed_jobs_locked()
            _evict_old_materialized_artifacts_locked()


def _record_cache_hit_locked(job_id: str) -> None:
    _metrics.cache_hits += 1
    job = _jobs.get(job_id)
    if job is not None:
        job.cache_hits += 1


def _record_cache_miss_locked(job_id: str) -> None:
    _metrics.cache_misses += 1
    job = _jobs.get(job_id)
    if job is not None:
        job.cache_misses += 1


def _execute_dag_node(node: DagNodeRecord) -> str:
    with _store_lock:
        cached_key = _get_cached_artifact_key_locked(node.cache_key)
        if cached_key is not None:
            if node.stage == STAGE_FORECAST_RUN:
                cached_value = _materialized_artifacts[cached_key].value
                if isinstance(cached_value, ForecastArtifact):
                    _forecast_artifacts[node.job_id] = cached_value
            _record_cache_hit_locked(node.job_id)
            return cached_key
        if DATFID_DAG_CACHE_ENABLED and node.cache_key:
            _record_cache_miss_locked(node.job_id)

    if node.stage == STAGE_FIT_PARSE_FILE:
        df = _read_table_from_bytes(node.payload["filename"], node.payload["file_bytes"])
        _normalize_datetime_columns(df)
        artifact_kind = "dataframe"
        artifact_value = df

    elif node.stage == STAGE_FIT_PARSE_PARAMS:
        params = _parse_fit_file_params(
            node.payload["lag_y"],
            node.payload["lagged_features"],
            node.payload["current_features"],
            node.payload["filter_by_significance"],
            node.payload["meanvar_test"],
            node.payload["signif"],
        )
        artifact_kind = "fit_params"
        artifact_value = {
            "id_col": node.payload["id_col"],
            "time_col": node.payload["time_col"],
            "y": node.payload["y"],
            **params,
        }

    elif node.stage == STAGE_FIT_MODEL:
        df_key = _dependency_artifact_key(node, STAGE_FIT_PARSE_FILE)
        params_key = _dependency_artifact_key(node, STAGE_FIT_PARSE_PARAMS)
        df = _get_artifact_value(df_key)
        params_payload = _get_artifact_value(params_key)
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        params = {
            "lag_y": params_payload["lag_y"],
            "lagged_features": params_payload["lagged_features"],
            "current_features": params_payload["current_features"],
            "filter_by_significance": params_payload["filter_by_significance"],
            "meanvar_test": params_payload["meanvar_test"],
            "signif": params_payload["signif"],
        }
        model, result_join = _fit_model_from_dataframe(
            df=df,
            id_col=params_payload["id_col"],
            time_col=params_payload["time_col"],
            y=params_payload["y"],
            params=params,
        )
        artifact_kind = "fit_model"
        artifact_value = FitArtifact(
            model=model,
            result_join=result_join,
            report_text="",
            params=params_payload,
            created_at=_utcnow(),
            model_artifact_key=_make_artifact_key("placeholder"),
        )

    elif node.stage == STAGE_FIT_REPORT:
        model_key = _dependency_artifact_key(node, STAGE_FIT_MODEL)
        model_artifact = _get_artifact_value(model_key)
        report_text = _result_to_text(model_artifact.result_join.result)
        job_fit_artifact = FitArtifact(
            model=model_artifact.model,
            result_join=model_artifact.result_join,
            report_text=report_text,
            params=model_artifact.params,
            created_at=_utcnow(),
            lock=model_artifact.lock,
            model_artifact_key=model_key,
        )
        with _store_lock:
            _fit_artifacts[node.job_id] = job_fit_artifact
        artifact_kind = "fit_report"
        artifact_value = report_text

    elif node.stage == STAGE_FORECAST_PARSE_FILE:
        df = _read_table_from_bytes(node.payload["filename"], node.payload["file_bytes"])
        _normalize_datetime_columns(df)
        artifact_kind = "dataframe"
        artifact_value = df

    elif node.stage == STAGE_FORECAST_RUN:
        forecast_df_key = _dependency_artifact_key(node, STAGE_FORECAST_PARSE_FILE)
        df_fc = _get_artifact_value(forecast_df_key)
        if isinstance(df_fc, pd.DataFrame):
            df_fc = df_fc.copy()
        fit_job_id = node.payload["fit_job_id"]
        with _store_lock:
            fit_artifact = _fit_artifacts.get(fit_job_id)
        if fit_artifact is None:
            raise RuntimeError("Fitted model artifact is no longer available.")
        with fit_artifact.lock:
            forecast_df = fit_artifact.model.forecast(
                extern_self=fit_artifact.result_join,
                df_forecast=df_fc,
            )
        artifact_value = ForecastArtifact(
            csv_bytes=_forecast_dataframe_to_csv_bytes(forecast_df),
            filename="forecast.csv",
            media_type="text/csv; charset=utf-8",
            created_at=_utcnow(),
        )
        with _store_lock:
            _forecast_artifacts[node.job_id] = artifact_value
        artifact_kind = "forecast_csv"

    else:
        raise RuntimeError(f"Unknown DAG stage: {node.stage}")

    with _store_lock:
        artifact_key = _store_artifact_locked(
            kind=artifact_kind,
            value=artifact_value,
            source_job_id=node.job_id,
            source_node_id=node.node_id,
            cache_key=node.cache_key,
        )
        stored_value = _materialized_artifacts[artifact_key].value
        if isinstance(stored_value, FitArtifact):
            stored_value.model_artifact_key = artifact_key
        if node.stage == STAGE_FORECAST_RUN:
            _forecast_artifacts[node.job_id] = stored_value
        return artifact_key


def _worker_loop(worker_index: int) -> None:
    while not _shutdown_event.is_set():
        try:
            node_id = _ready_node_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        node = _mark_node_running(node_id)
        try:
            if node is not None:
                artifact_key = _execute_dag_node(node)
                _mark_node_succeeded(node.node_id, artifact_key)
        except Exception as exc:
            traceback.print_exc()
            if node is not None:
                _mark_node_failed(node.node_id, str(exc))
        finally:
            _ready_node_queue.task_done()


def _start_job_workers() -> None:
    global _workers_started
    with _store_lock:
        if _workers_started:
            return
        _shutdown_event.clear()
        for index in range(DATFID_MAX_WORKERS):
            worker = threading.Thread(
                target=_worker_loop,
                args=(index,),
                name=f"datfid-job-worker-{index + 1}",
                daemon=True,
            )
            worker.start()
            _workers.append(worker)
        _workers_started = True


def _stop_job_workers() -> None:
    global _workers_started
    _shutdown_event.set()
    for worker in list(_workers):
        worker.join(timeout=1.0)
    with _store_lock:
        _workers[:] = [worker for worker in _workers if worker.is_alive()]
        _workers_started = bool(_workers)


def _response_for_created_job(job: JobRecord) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "status_url": f"/jobs/{job.job_id}",
            "result_url": job.result_url,
        },
    )


def _pending_result_response(job: JobRecord) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "status_url": f"/jobs/{job.job_id}",
        },
    )


def _failed_result_response(job: JobRecord) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "error": job.error,
        },
    )


def _metrics_payload() -> Dict[str, Any]:
    with _store_lock:
        durations = list(_metrics.durations_seconds)
        jobs_running = sum(1 for job in _jobs.values() if job.status == JOB_RUNNING)
        jobs_queued = sum(1 for job in _jobs.values() if job.status == JOB_QUEUED)
        dag_nodes_queued = sum(1 for node in _dag_nodes.values() if node.status == JOB_QUEUED)
        dag_nodes_running = sum(1 for node in _dag_nodes.values() if node.status == JOB_RUNNING)
        dag_nodes_succeeded = sum(1 for node in _dag_nodes.values() if node.status == JOB_SUCCEEDED)
        dag_nodes_failed = sum(1 for node in _dag_nodes.values() if node.status == JOB_FAILED)
        if durations:
            ordered = sorted(durations)
            p95_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
            avg_duration = round(float(sum(ordered) / len(ordered)), 6)
            p95_duration = round(float(ordered[p95_index]), 6)
        else:
            avg_duration = None
            p95_duration = None
        return {
            "queue_size": jobs_queued,
            "max_queue_size": DATFID_MAX_QUEUE_SIZE,
            "max_workers": DATFID_MAX_WORKERS,
            "max_stored_jobs": DATFID_MAX_STORED_JOBS,
            "max_materialized_artifacts": DATFID_MAX_MATERIALIZED_ARTIFACTS,
            "stored_job_count": len(_jobs),
            "stored_fit_artifacts": len(_fit_artifacts),
            "stored_forecast_artifacts": len(_forecast_artifacts),
            "jobs_submitted": _metrics.jobs_submitted,
            "jobs_running": jobs_running,
            "jobs_succeeded": _metrics.jobs_succeeded,
            "jobs_failed": _metrics.jobs_failed,
            "average_duration_seconds": avg_duration,
            "p95_duration_seconds": p95_duration,
            "rejected_jobs": _metrics.rejected_jobs,
            "cache_enabled": DATFID_DAG_CACHE_ENABLED,
            "cache_hits": _metrics.cache_hits,
            "cache_misses": _metrics.cache_misses,
            "materialized_artifacts": len(_materialized_artifacts),
            "dag_ready_queue_size": _ready_node_queue.qsize(),
            "dag_nodes_queued": dag_nodes_queued,
            "dag_nodes_running": dag_nodes_running,
            "dag_nodes_succeeded": dag_nodes_succeeded,
            "dag_nodes_failed": dag_nodes_failed,
        }

# ---------- FastAPI app ----------
app = FastAPI(
    title="DATFID API",
    description="Public demo API",
    docs_url="/docs",
    redoc_url=None,
    default_response_class=ORJSONResponse,
)

@app.on_event("startup")
def startup_job_workers():
    _start_job_workers()

@app.on_event("shutdown")
def shutdown_job_workers():
    _stop_job_workers()

@app.get("/")
def root():
    return {"message": "DATFID API is alive."}

# ✅ Validate directly here
@app.get("/secure-ping/")
def secure_ping():
    return {"ok": True}


# ---------- Job/session endpoints ----------
@app.post("/jobs/fit-file")
async def create_fit_file_job(
    file: UploadFile = File(...),
    id_col: str = Form(...),
    time_col: str = Form(...),
    y: str = Form(...),
    lag_y: str = Form(""),
    lagged_features: str = Form(""),
    current_features: str = Form(""),
    filter_by_significance: str = Form("false"),
    meanvar_test: str = Form("false"),
    signif: str = Form("0.05"),
):
    file_bytes = await file.read()
    payload = {
        "file_bytes": file_bytes,
        "filename": file.filename or "",
        "id_col": id_col,
        "time_col": time_col,
        "y": y,
        "lag_y": lag_y,
        "lagged_features": lagged_features,
        "current_features": current_features,
        "filter_by_significance": filter_by_significance,
        "meanvar_test": meanvar_test,
        "signif": signif,
    }
    job = _create_job(JOB_FIT, payload)
    return _response_for_created_job(job)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _store_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_to_dict(job)


@app.get("/jobs/{job_id}/fit-result")
def get_fit_result(job_id: str):
    with _store_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.kind != JOB_FIT:
            raise HTTPException(status_code=400, detail="Job is not a fit job.")
        if job.status in {JOB_QUEUED, JOB_RUNNING}:
            return _pending_result_response(job)
        if job.status == JOB_FAILED:
            return _failed_result_response(job)
        artifact = _fit_artifacts.get(job_id)

    if artifact is None:
        raise HTTPException(status_code=410, detail="Fit result is no longer available.")
    return Response(
        content=artifact.report_text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="result.txt"'},
    )


@app.post("/jobs/{fit_job_id}/forecast-file")
async def create_forecast_file_job(
    fit_job_id: str,
    df_forecast: UploadFile = File(...),
):
    with _store_lock:
        fit_job = _jobs.get(fit_job_id)
        if fit_job is None:
            raise HTTPException(status_code=404, detail="Fit job not found.")
        if fit_job.kind != JOB_FIT:
            raise HTTPException(status_code=400, detail="Referenced job is not a fit job.")
        if fit_job.status in {JOB_QUEUED, JOB_RUNNING}:
            raise HTTPException(status_code=409, detail="Fit job is not complete yet.")
        if fit_job.status == JOB_FAILED:
            raise HTTPException(status_code=409, detail=f"Fit job failed: {fit_job.error}")
        if fit_job_id not in _fit_artifacts:
            raise HTTPException(status_code=410, detail="Fitted model is no longer available.")

    file_bytes = await df_forecast.read()
    payload = {
        "fit_job_id": fit_job_id,
        "file_bytes": file_bytes,
        "filename": df_forecast.filename or "",
    }
    job = _create_job(JOB_FORECAST, payload, fit_job_id=fit_job_id)
    return _response_for_created_job(job)


@app.get("/jobs/{forecast_job_id}/forecast-result")
def get_forecast_result(forecast_job_id: str):
    with _store_lock:
        job = _jobs.get(forecast_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.kind != JOB_FORECAST:
            raise HTTPException(status_code=400, detail="Job is not a forecast job.")
        if job.status in {JOB_QUEUED, JOB_RUNNING}:
            return _pending_result_response(job)
        if job.status == JOB_FAILED:
            return _failed_result_response(job)
        artifact = _forecast_artifacts.get(forecast_job_id)

    if artifact is None:
        raise HTTPException(status_code=410, detail="Forecast result is no longer available.")
    return Response(
        content=artifact.csv_bytes,
        media_type=artifact.media_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact.filename}"'},
    )


@app.get("/metrics")
def metrics():
    return _metrics_payload()

# ---------- Model endpoints (guarded) ----------
@app.post("/modelfit/")
async def modelfit(
    df: List[Dict] = Body(...),
    id_col: str = Body(...),
    time_col: str = Body(...),
    y: str = Body(...),
    lag_y: Any = Body(None),
    lagged_features: Any = Body({}),
    current_features: Any = Body([]),
    filter_by_significance: bool = Body(False),
    meanvar_test: bool = Body(False),
    signif: Any = Body(0.05),
):
    global stored_model, stored_result_join
    try:
        sig_val = float(signif)
    except (TypeError, ValueError):
        sig_val = 0.05
    df1 = pd.DataFrame(df)
    try:
        model = DATFIDModel(
            df=df1,
            id_col=id_col,
            time_col=time_col,
            y=y,
            lag_y=lag_y,
            lagged_features=lagged_features,
            current_features=current_features,
            filter_by_significance=filter_by_significance,
            meanvar_test=meanvar_test,
            signif=sig_val,
        )
        result_join = model.fit()
        result = result_join.result

        # save for forecast
        stored_model = model
        stored_result_join = result_join

        # jsonify result object
        result_dict = {}
        for k, v in result.__dict__.items():
            if isinstance(v, pd.DataFrame):
                result_dict[k] = v.to_dict(orient="records")
            elif isinstance(v, pd.Series):
                result_dict[k] = v.to_dict()
            elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                result_dict[k] = v
            else:
                result_dict[k] = str(v)
        return result_dict
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during model fit: {str(e)}")

@app.post("/modelforecast/")
async def modelforecast(
    payload: Any = Body(...),
):
    global stored_model, stored_result_join
    if stored_model is None or stored_result_join is None:
        raise HTTPException(status_code=400, detail="Model not fitted. Call /modelfit/ first.")
    try:
        # Accept both payload styles:
        # 1) raw list: [...]
        # 2) wrapped dict: {"df_forecast": [...]}
        if isinstance(payload, dict) and "df_forecast" in payload:
            df_forecast = payload.get("df_forecast")
        else:
            df_forecast = payload
        if not isinstance(df_forecast, list):
            raise HTTPException(
                status_code=422,
                detail="Payload must be a list of rows or {'df_forecast': [rows]}",
            )
        df_forecast1 = pd.DataFrame(df_forecast)
        forecast_df = stored_model.forecast(extern_self=stored_result_join, df_forecast=df_forecast1)
        return forecast_df.to_dict(orient="records")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during model forecast: {str(e)}")

@app.post("/modelfit-file/")
async def modelfit_file(
    file: UploadFile = File(...),
    id_col: str = Form(...),
    time_col: str = Form(...),
    y: str = Form(...),
    lag_y: str = Form(""),
    lagged_features: str = Form(""),
    current_features: str = Form(""),
    filter_by_significance: str = Form("false"),
    meanvar_test: str = Form("false"),
    signif: str = Form("0.05"),
):
    global stored_model, stored_result_join
    try:
        df = _read_table_from_upload(file)
        _normalize_datetime_columns(df)
        params = _parse_fit_file_params(
            lag_y,
            lagged_features,
            current_features,
            filter_by_significance,
            meanvar_test,
            signif,
        )
        model, result_join = _fit_model_from_dataframe(
            df=df,
            id_col=id_col,
            time_col=time_col,
            y=y,
            params=params,
        )
        stored_model = model
        stored_result_join = result_join

        # Build textual report
        report_text = _result_to_text(result_join.result)
        # Write to a temp file and return as attachment
        tmp_path = _write_text_attachment_tmp(report_text, suffix=".txt")

        fname = "result.txt"
        return FileResponse(
            tmp_path,
            media_type="text/plain; charset=utf-8",
            filename=fname,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during model fit (file): {str(e)}")
    
@app.post("/modelforecast-file/")
async def modelforecast_file(
    df_forecast: UploadFile = File(...),
):
    global stored_model, stored_result_join
    if stored_model is None or stored_result_join is None:
        raise HTTPException(status_code=400, detail="Model not fitted. Call /modelfit-file/ (or /modelfit/) first.")
    try:
        df_fc = _read_table_from_upload(df_forecast)
        _normalize_datetime_columns(df_fc)

        forecast_df = stored_model.forecast(extern_self=stored_result_join, df_forecast=df_fc)

        # Save to CSV and return (Excel-friendly)
        csv_bytes = _forecast_dataframe_to_csv_bytes(forecast_df)
        tmp_path = _write_bytes_attachment_tmp(csv_bytes, suffix=".csv")

        return FileResponse(
            tmp_path,
            media_type="text/csv; charset=utf-8",
            filename="forecast.csv",
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during model forecast (file): {str(e)}")
