import os, asyncio, httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# --- Public config ---
# Hardcoded demo upstream URL (no secret/env needed).
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://huseyinkocer-datfid-api.hf.space").rstrip("/")
DEMO_FORWARD_URL   = os.getenv("DEMO_FORWARD_URL", "").rstrip("/")  # optional demo-only backend (unused in hackathon flow)
# This Space's public URL (used to ping self while waiting so HF does not put this Space to sleep). Override with SELF_URL env if different.
SELF_URL = os.getenv("SELF_URL", "https://huseyinkocer-datfid-master.hf.space").rstrip("/")

if not UPSTREAM_URL.startswith("https://"):
    raise RuntimeError("Invalid upstream URL in code.")
if ".hf.space" not in UPSTREAM_URL:
    raise RuntimeError(
        "Invalid upstream URL. Use a live runtime URL like https://<name>.hf.space."
    )

app = FastAPI(title="DATFID Public Proxy", docs_url="/docs", redoc_url=None)

# Prefer X-Forwarded-For (HF sits behind a proxy)
def client_ip(request: Request):
    xff = request.headers.get("x-forwarded-for")
    return xff.split(",")[0].strip() if xff else (request.client.host or "0.0.0.0")

limiter = Limiter(key_func=client_ip)  # or get_remote_address
GLOBAL_LIMIT = limiter.limit("100/10minute", key_func=lambda: "global:any") # global limit

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Too many requests, slow down."})

SDK_MAX_BODY_BYTES = int(os.getenv("SDK_MAX_BODY_BYTES", "25000000"))  # 25MB default for demo
SDK_MAX_BODY_BYTES_extended = int(os.getenv("SDK_MAX_BODY_BYTES_extended", "125000000")) # 125MB default for prod 

# How long to wait for upstream (API) response; long runs may need 30+ min (1800+)
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "900"))  # 15 minutes
# While waiting for a response: every PING_INTERVAL seconds we ping the backend (secure_ping) and, if SELF_URL is set, we also ping ourselves (keep-alive) so this Space stays awake
PING_INTERVAL = float(os.getenv("PING_INTERVAL", "270"))

# Early global body-size guard (runs before routes)
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")

    # demo routes use the smaller cap, others use extended
    path = request.url.path or ""
    cap = SDK_MAX_BODY_BYTES if "-demo" in path else SDK_MAX_BODY_BYTES_extended

    if cl and int(cl) > cap:
        return JSONResponse(
            {"detail": f"Payload too large (> {cap} bytes)"},
            status_code=413,
        )
    return await call_next(request)

# CORS for browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://datfid.com",
        "https://www.datfid.com"
    ],
    # Optional: allow Vercel preview domains
    # allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=86400,
)

# to ensure we don’t leak hop-by-hop headers
def _filter_resp_headers(headers: dict) -> dict:
    # pass through useful headers but strip hop-by-hop
    allowed = {"content-disposition"}
    return {k: v for k, v in headers.items() if k.lower() in allowed}

async def _ping_upstream_loop(ping_url: str, headers: dict, interval: float, self_url: str = ""):
    """Every `interval` seconds: if self_url set, GET self_url/keep-alive (keep this Space awake), then GET ping_url (backend). Stops when cancelled."""
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if self_url:
                    try:
                        await client.get(f"{self_url}/keep-alive")
                    except Exception:
                        pass
                await client.get(ping_url, headers=headers)
        except asyncio.CancelledError:
            break
        except Exception:
            pass

def _start_ping_task(ping_url: str, headers: dict, self_url: str = ""):
    """Start background ping task: every PING_INTERVAL seconds ping backend and, if self_url set, also ping self."""
    if PING_INTERVAL <= 0:
        return None
    return asyncio.create_task(_ping_upstream_loop(ping_url, headers, PING_INTERVAL, self_url))

async def _cancel_ping_task(task: asyncio.Task | None):
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

async def _forward(path: str, method: str = "GET", json_body=None):
    """
    Forward request to upstream API (public demo mode).
    While waiting, every PING_INTERVAL seconds: secure_ping to backend and,
    if SELF_URL set, ping self (keep-alive).
    """
    url = f"{UPSTREAM_URL}{path}"
    headers = {"Accept": "application/json"}

    # While waiting: ping backend (secure_ping) and, if SELF_URL set, ping self so this Space stays awake
    ping_url = f"{UPSTREAM_URL}/secure-ping/"
    ping_headers = {}

    timeout = httpx.Timeout(UPSTREAM_TIMEOUT)
    r = None

    async def do_request():
        nonlocal r
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.request(method, url, headers=headers, json=json_body)

    request_task = asyncio.create_task(do_request())
    wait_sec = PING_INTERVAL if PING_INTERVAL > 0 else 0.0
    try:
        while not request_task.done():
            if wait_sec <= 0:
                await request_task
                break
            ping_sleep = asyncio.create_task(asyncio.sleep(wait_sec))
            done, pending = await asyncio.wait(
                {request_task, ping_sleep},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=UPSTREAM_TIMEOUT + 10,
            )
            for t in pending:
                if t is not request_task:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            if request_task in done:
                break
            async with httpx.AsyncClient(timeout=10.0) as c:
                if SELF_URL:
                    try:
                        await c.get(f"{SELF_URL}/keep-alive")
                    except Exception:
                        pass
                try:
                    await c.get(ping_url, headers=ping_headers)
                except Exception:
                    pass
        if not request_task.done():
            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass
        await request_task
    except asyncio.CancelledError:
        request_task.cancel()
        try:
            await request_task
        except asyncio.CancelledError:
            pass
        raise
    exc = request_task.exception()
    if exc is not None:
        raise exc
    if r is None:
        raise RuntimeError("Upstream request did not complete")

    ct = r.headers.get("content-type", "")

    if "application/json" in ct:
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            return JSONResponse(status_code=r.status_code, content={"error": r.text[:500]})
    # Fallback: return short text envelope if non-JSON
    return JSONResponse(status_code=r.status_code, content={"text": r.text[:1000]})

async def _forward_stream(path: str, files=None, data=None, method: str = "POST"):
    url = f"{UPSTREAM_URL}{path}"
    headers = {"Accept": "*/*"}

    ping_headers = {}
    ping_task = _start_ping_task(f"{UPSTREAM_URL}/secure-ping/", ping_headers, SELF_URL)
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    # Don't open the context yet; the iterator must own the context lifetime.
    stream_ctx = client.stream(method, url, headers=headers, files=files, data=data)

    # Mutable holders we can fill once the stream opens
    status_holder = {"code": 200}
    media_type_holder = {"ct": "application/octet-stream"}
    headers_holder = {}

    async def body_iter():
        chunk_queue: asyncio.Queue = asyncio.Queue()
        stream_done = {"done": False, "exc": None}

        async def stream_reader():
            try:
                async with stream_ctx as resp:
                    status_holder["code"] = resp.status_code
                    media_type_holder["ct"] = resp.headers.get("content-type", "application/octet-stream")
                    headers_holder.update(_filter_resp_headers(resp.headers))
                    if resp.status_code >= 400:
                        chunk = await resp.aread()
                        await chunk_queue.put(chunk)
                        await chunk_queue.put(None)
                        return
                    async for chunk in resp.aiter_raw():
                        await chunk_queue.put(chunk)
                    await chunk_queue.put(None)
            except Exception as e:
                stream_done["exc"] = e
                await chunk_queue.put(None)
            finally:
                stream_done["done"] = True

        reader_task = asyncio.create_task(stream_reader())
        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    if stream_done["exc"]:
                        raise stream_done["exc"]
                    break
                yield chunk
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            await _cancel_ping_task(ping_task)
            await client.aclose()

    response = StreamingResponse(
        body_iter(),
        status_code=status_holder["code"],
        media_type=media_type_holder["ct"],
        headers=headers_holder,
    )
    return response


async def _forward_multipart_json(path: str, files=None, data=None, method: str = "POST"):
    """POST multipart to upstream and return JSON. While waiting: secure_ping to backend; if SELF_URL set, ping self."""
    url = f"{UPSTREAM_URL}{path}"
    headers = {"Accept": "application/json"}
    ping_url = f"{UPSTREAM_URL}/secure-ping/"
    ping_headers = {}
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT)
    r = None

    async def do_request():
        nonlocal r
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.request(method, url, headers=headers, files=files, data=data)

    request_task = asyncio.create_task(do_request())
    wait_sec = PING_INTERVAL if PING_INTERVAL > 0 else 0.0
    try:
        while not request_task.done():
            if wait_sec <= 0:
                await request_task
                break
            ping_sleep = asyncio.create_task(asyncio.sleep(wait_sec))
            done, pending = await asyncio.wait(
                {request_task, ping_sleep},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=UPSTREAM_TIMEOUT + 10,
            )
            for t in pending:
                if t is not request_task:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            if request_task in done:
                break
            async with httpx.AsyncClient(timeout=10.0) as c:
                if SELF_URL:
                    try:
                        await c.get(f"{SELF_URL}/keep-alive")
                    except Exception:
                        pass
                try:
                    await c.get(ping_url, headers=ping_headers)
                except Exception:
                    pass
        if not request_task.done():
            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass
        await request_task
    except asyncio.CancelledError:
        request_task.cancel()
        try:
            await request_task
        except asyncio.CancelledError:
            pass
        raise
    exc = request_task.exception()
    if exc is not None:
        raise exc
    if r is None:
        raise RuntimeError("Upstream request did not complete")
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            return JSONResponse(status_code=r.status_code, content={"error": r.text[:500]})
    return JSONResponse(status_code=r.status_code, content={"text": r.text[:1000]})


# for demo
async def _forward_demo_stream(path: str, *, files: dict | None, data: dict | None, method: str = "POST"):
    if not DEMO_FORWARD_URL:
        raise HTTPException(status_code=500, detail="Demo not configured.")

    url = DEMO_FORWARD_URL + path
    headers = {"Accept": "*/*"}
    ping_headers = {}
    ping_task = _start_ping_task(f"{DEMO_FORWARD_URL.rstrip('/')}/", ping_headers)

    timeout = httpx.Timeout(120.0)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    stream_ctx = client.stream(method, url, headers=headers, files=files, data=data)

    status_holder = {"code": 200}
    media_type_holder = {"ct": "application/octet-stream"}
    headers_holder = {}

    async def body_iter():
        chunk_queue: asyncio.Queue = asyncio.Queue()
        stream_done = {"done": False, "exc": None}

        async def stream_reader():
            try:
                async with stream_ctx as resp:
                    status_holder["code"] = resp.status_code
                    media_type_holder["ct"] = resp.headers.get("content-type", "application/octet-stream")
                    headers_holder.update(_filter_resp_headers(resp.headers))
                    if resp.status_code >= 400:
                        chunk = await resp.aread()
                        await chunk_queue.put(chunk)
                        await chunk_queue.put(None)
                        return
                    async for chunk in resp.aiter_raw():
                        await chunk_queue.put(chunk)
                    await chunk_queue.put(None)
            except Exception as e:
                stream_done["exc"] = e
                await chunk_queue.put(None)
            finally:
                stream_done["done"] = True

        reader_task = asyncio.create_task(stream_reader())
        try:
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    if stream_done["exc"]:
                        raise stream_done["exc"]
                    break
                yield chunk
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            await _cancel_ping_task(ping_task)
            await client.aclose()

    response = StreamingResponse(
        body_iter(),
        status_code=status_holder["code"],
        media_type=media_type_holder["ct"],
        headers=headers_holder,
    )
    return response

@app.get("/")
async def root(req: Request):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            ping = await client.get(f"{UPSTREAM_URL}/")
        return {
            "message": "DATFID master demo is alive.",
            "upstream_url": UPSTREAM_URL,
            "upstream_status": ping.status_code,
            "upstream_ok": ping.status_code < 400,
        }
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={
                "message": "DATFID master demo is alive, but upstream is unreachable.",
                "upstream_url": UPSTREAM_URL,
                "error": str(e)[:200],
            },
        )


@app.get("/keep-alive")
async def keep_alive():
    """
    Hit this to keep this Space (datfid_master) and the backend awake.
    - External cron (e.g. every 4 min): UptimeRobot, cron-job.org — keeps both spaces awake when idle.
    - During a long request (>5 min): the client can call this in a background thread every ~4 min
      so HF does not put this Space to sleep while it is waiting for the backend (no response
      sent to the client yet). This endpoint also pings the upstream (backend). No auth required.
    """
    try:
        url = f"{UPSTREAM_URL}/"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
        return JSONResponse(content={"ok": True, "upstream_status": r.status_code})
    except Exception as e:
        return JSONResponse(status_code=502, content={"ok": False, "error": str(e)[:200]})

@app.get("/secure-ping/")
async def secure_ping(req: Request):
    return await _forward("/secure-ping/", "GET")


@app.post("/modelfit/")
async def modelfit(req: Request):
    body = await req.json()
    return await _forward("/modelfit/", "POST", json_body=body)

@app.post("/modelforecast/")
async def modelforecast(req: Request):
    body = await req.json()
    return await _forward("/modelforecast/", "POST", json_body=body)

@app.post("/modelfit-file/")
async def modelfit_file(
    req: Request,
    file: UploadFile = File(...),
    id_col: str = Form(...),
    time_col: str = Form(...),
    y: str = Form(...),
    # optional knobs
    lag_y: str = Form(""),
    lagged_features: str = Form(""),          # JSON string or empty
    current_features: str = Form(""),         # "all" | JSON string | ""
    filter_by_significance: str = Form("false"),
    meanvar_test: str = Form("false"),
    signif: str = Form("0.05"),
):
    raw = await file.read()
    if len(raw) > SDK_MAX_BODY_BYTES_extended:
        raise HTTPException(status_code=413, detail="Payload too large.")

    files = {
        "file": (file.filename, raw, file.content_type or "application/octet-stream"),
    }
    data = {
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
    return await _forward_stream("/modelfit-file/", files=files, data=data, method="POST")

@app.post("/modelforecast-file/")
async def modelforecast_file(
    req: Request,
    df_forecast: UploadFile = File(...),
):
    raw = await df_forecast.read()
    if len(raw) > SDK_MAX_BODY_BYTES_extended:
        raise HTTPException(status_code=413, detail="Payload too large.")

    files = {
        "df_forecast": (df_forecast.filename, raw, df_forecast.content_type or "application/octet-stream"),
    }
    return await _forward_stream("/modelforecast-file/", files=files, data=None, method="POST")

@app.get("/health-demo-proxy")
async def health_demo_proxy():
    # convenience endpoint to test demo proxy hop
    try:
        return await _forward_demo_stream("/health-demo", files=None, data=None, method="GET")
    except HTTPException as e:
        # bubble up errors so you can diagnose missing config, wrong URL, etc.
        raise e
