# audio-trial backend — Slice 0

FastAPI service that powers the audio-model trial web app.

## Quick start (local)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set DEEPGRAM_API_KEY if you want to use Deepgram

uvicorn main:app --reload --port 8000
```

Verify:

```bash
curl http://localhost:8000/api/health
# → {"status":"ok","version":"0.1.0"}

curl http://localhost:8000/api/adapters
# → {"adapters":[{"id":"deepgram",...},{"id":"faster_whisper",...}]}
```

## Docker

```bash
docker build -t audio-trial-backend .
docker run -p 8000:8000 --env-file .env audio-trial-backend
```

## Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/adapters` | List registered adapters with metadata |
| `POST` | `/api/clips` | Upload audio file (multipart); returns Clip JSON |
| `GET` | `/api/clips` | List all clips |
| `GET` | `/api/clips/{id}/audio` | Stream raw audio file |
| `POST` | `/api/runs` | Start synchronous single-adapter run; returns Run JSON |
| `GET` | `/api/runs/{id}` | Fetch a completed run |
| `WS` | `/ws/run/{run_id}` | Stream run events (connect before POST /api/runs) |

### WebSocket events

```jsonc
// StageStarted — emitted immediately when the run begins
{"event": "StageStarted", "run_id": "...", "adapter": "deepgram", "timestamp": "..."}

// StagePartial — transcript preview once the adapter returns
{"event": "StagePartial", "run_id": "...", "adapter": "deepgram", "text": "hello world", "timestamp": "..."}

// StageCompleted — full result with latency + cost
{"event": "StageCompleted", "run_id": "...", "adapter": "deepgram", "latency_ms": 820.3, "cost_usd": 0.000012, "result": {...}, "timestamp": "..."}

// StageFailed — adapter threw an exception
{"event": "StageFailed", "run_id": "...", "adapter": "deepgram", "error": "RuntimeError: ...", "timestamp": "..."}
```

## Architecture notes (for future maintainers)

### Why a single-stage runner in Slice 0?

The plan spec (§P0) deliberately forbids a full DAG runner to keep Slice 0
shippable in one sprint.  The runner in `main.py` (`create_run`) does:
  1. Resolve clip → adapter → call `adapter.transcribe(path, config)`
  2. Emit WebSocket events before/after the call
  3. Persist the Run as a JSONL line

Multi-stage topological-sort DAG execution lands in Slice 1 inside
`backend/pipelines/runner.py`.

### Adapter protocol

Every adapter is a plain Python class — no base class, structural typing via
`adapters/base.py::Adapter` (Protocol).  Adding an adapter means:
  1. Create `backend/adapters/<name>_adapter.py` with the required fields + `async transcribe()`
  2. `registry.register(<Name>Adapter())` in `main.py`

The Registry serialises adapters to JSON for `/api/adapters` without needing
any extra plumbing.

### Storage layout

```
data/
  clips/<clip_id>/
    manifest.json      # Clip metadata
    audio.<ext>        # Raw audio bytes
  runs/
    <adapter>.jsonl    # Append-only run log (one JSON object per line)
```

`data/` is git-ignored.  On Modal this maps to a persistent Volume.

### CORS

`FRONTEND_ORIGIN` env var (default `http://localhost:5173`) is the only
allowed origin.  Add the deployed frontend URL to this var in production.

### Extending to multi-stage (Slice 1 TODO)

Replace `create_run` in `main.py` with a call to the DAG runner in
`pipelines/runner.py`.  The runner will fan out events per-stage over the
same WebSocket connection; the `StageStarted/StageCompleted` schema already
carries an `adapter` field to distinguish stages.
