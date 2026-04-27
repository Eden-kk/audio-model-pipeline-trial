# Audio model trial app — web v1 (audio-only MVP)

**Remote repo:** [`github.com/Eden-kk/audio-model-pipeline-trial`](https://github.com/Eden-kk/audio-model-pipeline-trial) — all P0 + P1 work lands here, separate from `ambient-deploy`. The trial-app reuses adapters and metrics from `ambient-deploy/benchmarks/` via Python package import or git submodule (decision deferred to Slice 0 of implementation).

## Context

AR-glasses voice-assistant research (informed by Meta's AR-glass audio path — Ray-Ban Meta, Aria, Orion — and adjacent reference designs) is producing a steady stream of new audio model candidates: cloud APIs, edge models, streaming vs batch, open-weight vs closed. Today the only way to compare them is to write a one-off script per candidate. We need a **reusable trial harness** — a web playground + pipeline composer + corpus builder — that scales as new models keep arriving and lets us evaluate them against AR-glass-relevant scenarios (outdoor traffic, indoor cafe, multi-speaker, code-switching, whisper voice, accented speech).

This is *evaluation infrastructure*, not a customer product. Speed of iteration and clarity of comparison matter more than polish.

## Scope (audio only for MVP)

- **In MVP (P0):** ASR, TTS, speaker verification, diarization. Audio input only — mic + audio file upload.
- **In P1:** pipeline composer, side-by-side comparison, more adapters, public-benchmark importers.
- **Deferred to P2 (separate plan):** video input, realtime omni models (Gemini Live, OpenAI gpt-realtime, Moshi, Qwen-Omni), iOS companion, LLM auto-classification of clips.

## Locked decisions

| Decision | Pick | Why |
|----------|------|-----|
| Platform | **Web first** (React + Vite + TS); iOS in P2 separate plan | Pipeline-graph viz is the killer feature and far cleaner on web. Iteration speed > on-device fidelity for a trial app. iOS becomes Phase 3 when on-device models matter. |
| MVP scope | **Audio only — no video, no omni** | Cuts ~30 % of v1 surface; lets MVP ship in ~3 weeks instead of 4. Video + omni land in P1 once the audio path is solid. |
| Auto-classify | **Heuristic + manual chip-tagging** | On ingest extract objective metadata (duration, SNR, speaker count, language). Scenarios are one-click chips in UI. Predictable, no per-ingest LLM cost. LLM auto-classify lands in P2. |
| Hosting (self-host model side) | **Modal Labs (NVIDIA L4 + A100)** | See §Backend hosting below. AMD route is yellow on software (Parakeet ROCm is community fork; pyannote no official ROCm) and Modal has zero-ops DX advantage at our 5–200 calls/day scale. Switch criteria captured. |

## Backend hosting — Modal vs AMD GPU clouds

**Pick: Modal.** Decision factors (all evidence in the research notes):

- **Software ecosystem TODAY (2025–2026):**
  - vLLM on ROCm: **green** (first-class platform Jan 2026, official wheels + Docker).
  - PyTorch on ROCm: **green** (mature).
  - pyannote on ROCm: **yellow** (works in AMD's ROCm Docker but no official upstream support, torchcodec lacks ROCm wheels).
  - NeMo / Parakeet on ROCm: **yellow-red** (community fork only; needs `HSA_OVERRIDE_GFX_VERSION` magic).
- **DX delta:** Modal has typed `@modal.cls` deploys, volumes, secrets, websocket ASGI, hot reloads, dashboards, zero ops. RunPod is closest on AMD side but still adds custom containers and cold-start variance. TensorWave/Crusoe/Hot Aisle add Terraform / K8s / bare-metal SSH.
- **Cost at our scale:** at 5–200 calls/day across the Parakeet+pyannote+Qwen trio: Modal ≈ $0.0030/call, RunPod MI300X ≈ $0.0045/call (low traffic) or $0.0038/call (peak). Modal *cheaper* for trial-phase workloads.
- **Modal does NOT offer AMD GPUs** as of Apr 2026.

**Switch trigger** (re-evaluate AMD when any of these are true):
1. Sustained traffic exceeds **~100 sustained RPS** (~100k calls/day) — the cost crossover flips.
2. **NeMo+Parakeet** stops being a critical path (e.g., Deepgram covers all our ASR needs, no self-host).
3. **In-house ROCm expertise** appears (a hire or contractor) — unlocks TensorWave $1.50/hr bare-metal at 50 % of Modal cost long-term.

## Reference design (north stars, with sources)

Survey of 13 audio/AI playgrounds. Closest analogue to what we're building: **Daggr (Gradio's DAG builder) + AssemblyAI Playground hybrid**.

**Top 3 patterns to steal (in priority order):**

1. **Port-typed DAG canvas with stage drill-in inspection** (from Daggr).
   - Each adapter is a node; ports are typed (`audio_pcm`, `text`, `embedding`, `audio_stream`, `audio_file`).
   - Connectors are only allowed between compatible types — invalid wiring is rejected at design time.
   - Click any node to inspect its current input + output + latency + raw response.
   - Re-run a single stage with modified config.
2. **Configuration sidebar + full-width results pane** (from AssemblyAI Playground).
   - Left: feature toggles for the active stage (model variant, language, voice, threshold).
   - Right: results — streaming transcript, waveform, metrics card.
   - Inline code generation for "use this config in your code."
3. **Synced waveform scrubber + interim/final transcript highlighting** (from ElevenLabs Studio + Deepgram Console).
   - Canvas-rendered waveform with draggable playhead.
   - Live recording renders waveform in real time.
   - Streaming transcript shows interim tokens grayed, final tokens solid.

**What's *missing* in every playground we surveyed (our differentiation):**
- Per-stage latency timeline across a multi-stage pipeline.
- A/B diff visualization (waveform diff, text diff, confidence diff).
- Run history + replay ("save this run, clone it, tweak one param, see diff").
- Per-stage cost attribution.
- Multi-clip batch comparison with scenario-bucketed scoring.

These five gaps are what give our app product-market fit beyond "yet another playground."

## Information architecture (sidebar)

```
┌────────────────────────────────────────────┐
│  ▷ 🎙  Playground         (single-model    │
│         quick-test, mic/upload, like        │
│         AssemblyAI's playground)            │
│                                             │
│  ▷ 🧩  Pipelines          (recipe gallery   │
│         + drag-drop composer (P1))          │
│                                             │
│  ▷ ▶️  Run                (live execution   │
│         view with port-typed DAG +          │
│         stage drill-in)                     │
│                                             │
│  ▷ 📚  Corpus             (clip library:    │
│         filter by scenario chip, language,  │
│         duration, speaker count, SNR)       │
│                                             │
│  ▷ 📊  Compare            (A/B two pipelines│
│         on same clip — the "missing"        │
│         feature from survey)                │
│                                             │
│  ▷ ⚙️  Settings           (API keys, cost   │
│         caps, scenario taxonomy editor)     │
└────────────────────────────────────────────┘
```

Each sidebar entry is an **independent function** — none depends on another being live. A user can stay in Playground for a whole session without ever opening Pipelines.

## Model categorization

The pipeline composer is **constrained by model categories**. Each adapter declares a category, an input-port set, and an output-port set. The composer's wiring rules check port-type compatibility.

| Category | Input ports | Output ports | Adapters in MVP |
|----------|-------------|--------------|-----------------|
| **VAD** | `audio_pcm` | `audio_pcm` + `speech_segments[]` | (P1) Silero |
| **ASR (transcription)** | `audio_file` \| `audio_stream` | `text` + `word_timing[]` | Deepgram, OpenAI Transcribe |
| **Speaker verification** | `audio_enroll` + `audio_test` | `score` + `match_decision` | pyannote_verify, Resemblyzer |
| **Diarization** | `audio_file` | `segments[speaker, start, end]` | (P1) pyannote_diar |
| **TTS** | `text` + `voice_config` | `audio_stream` | Cartesia, ElevenLabs |
| **Intent / Tool LLM** | `text` | `text` + `tool_calls[]` | (P1) Qwen-7B |
| **Realtime omni** | `audio_stream` | `audio_stream` + `text` | (P2 — deferred) |

Composer constraint examples:
- `Diarization → ASR` is allowed: diarization output's `segments[]` carry timestamps the ASR can window into.
- `TTS → ASR → TTS` is allowed (round-trip quality test).
- `VAD → Speaker verification` is allowed: VAD's `speech_segments[]` feed the verifier's enroll-and-score loop.
- `ASR → Speaker verification` is **rejected** at design time: incompatible port types (text → audio_enroll).

The Pipelines page renders the category palette as the left rail of the composer; users drag categories onto the canvas, then pick the specific adapter from a dropdown on the node.

## Capability tiers

### P0 — MVP audio web app (~3 weeks; narrowed scope)

| # | Capability | Reuse |
|---|-----------|-------|
| 0.1 | **Backend FastAPI on Modal** with `/api/pipelines`, `/api/clips`, `/api/runs`; WebSocket `/ws/run/{id}` for streaming events | `00_ui_server.py` pattern |
| 0.2 | **6 adapters wired** through a category-typed `Stage` protocol: Deepgram (ASR), OpenAI Transcribe (ASR), Cartesia TTS, ElevenLabs TTS, pyannote_verify (speaker), Resemblyzer (speaker) | `benchmarks/adapters/*` (rewrap with category metadata) |
| 0.3 | **4 pre-built pipeline recipes** in `Pipelines` page: ASR-only, TTS-only, Speaker-verify-only, ASR-then-TTS-roundtrip | New code |
| 0.4 | **Audio input modes**: live mic via WebRTC + AudioWorklet @ 16k Int16 PCM; audio file upload (WAV/MP3/Opus). **No video, no omni in MVP.** | New code |
| 0.5 | **Pipeline visualization** in `Run` page: vertical port-typed DAG with per-stage status, latency, model name; nodes light up as run progresses; failed nodes red | react-flow + custom node styles |
| 0.6 | **Stage drill-in panel**: click any node → slide-out panel with input preview (waveform/text), output preview, latency timeline, raw model response (collapsed) | New component |
| 0.7 | **`Playground` page** for single-model quick test (mic/upload → pick category → pick adapter → run) — AssemblyAI-shaped UI | New code |
| 0.8 | **`Corpus` page** with clip storage on Modal Volume at `/vol/trial-app/clips/<clip_id>/`; auto-extract: duration, sample rate, channels, format, SNR (energy-based), speaker count (pyannote pass), language ID (Whisper-turbo via Groq); manual scenario chip tagging | New code; reuse pyannote modal deploy |
| 0.9 | **Sidebar nav**: Playground / Pipelines / Run / Corpus / Settings. (Compare lands in P1.) | New code |
| 0.10 | **Result persistence as JSONL** at `/vol/trial-app/runs/<scenario>/<pipeline>.jsonl` — append-only; queryable by scenario × pipeline × clip | New code |

### P1 — composer + comparison + omni (~4 weeks)

| # | Capability |
|---|-----------|
| 1.1 | **Drag-and-drop pipeline composer** — react-flow editor in `Pipelines` page; category palette as left rail; port-type validation at wiring time |
| 1.2 | **Adapter expansion** — wire Parakeet (Modal), pyannote_diar (Modal), Gemini Live (omni), Stitched pipeline, AssemblyAI, Speechmatics, Gladia, GroqWhisper, OpenAI TTS, Picovoice Eagle, Silero VAD |
| 1.3 | **`Compare` page** — pick a clip + two pipelines; run both; render waterfall chart of per-stage timings side-by-side; text-diff for ASR; audio side-by-side for TTS; metric comparison table |
| 1.4 | **Realtime omni + video input** wiring (Gemini Live, eventually gpt-realtime, Moshi when WS bug fixes) |
| 1.5 | **MOS rating UI** — inline 1–5 stars on TTS output; aggregate in compare view |
| 1.6 | **Public-benchmark presets** — LibriSpeech, MUSAN, FLEURS, Common Voice clips importable as scenarios |

### Out of scope for v1 (filed for P2 separate plan)

- iOS companion app for on-device models (Whisper.cpp, Core ML, ANE)
- LLM-driven auto-classification (Qwen-VL on video, Qwen-7B on transcripts)
- Real AR-glass field-recording corpus
- Public benchmark export to HuggingFace Datasets format
- Multi-user auth / team workspaces
- AMD GPU migration (deferred unless switch trigger fires)

## Data model

```typescript
type ModelCategory = "vad" | "asr" | "speaker_verify" | "diarization" | "tts" | "intent_llm" | "realtime_omni"

type Adapter = {
  id: string                              // "deepgram" | "cartesia_tts" | ...
  category: ModelCategory
  display_name: string
  hosting: "cloud" | "modal" | "edge"
  vendor: string
  inputs: { name: string; type: PortType }[]
  outputs: { name: string; type: PortType }[]
  config_schema: JSONSchema
  cost_per_call_estimate_usd: number | null
}

type PortType = "audio_file" | "audio_stream" | "audio_pcm" | "text"
              | "word_timing" | "speech_segments" | "speaker_segments"
              | "embedding" | "score" | "tool_calls"

type Stage = { id: string; adapter: string; config: object; position?: { x:number; y:number } }
type Pipeline = { id: string; name: string; is_recipe: boolean; stages: Stage[]; edges: { from: string; to: string; port: string }[] }

type Clip = {
  id: string; source: "record" | "upload"; modality: "audio"
  duration_s: number; sample_rate: number; channels: number; format: string
  language_detected: string | null; snr_db: number | null
  speaker_count_estimate: number | null
  user_tags: string[]; scenarios: string[]
  uploaded_by: string; created_at: string
}

type Run = {
  id: string; pipeline_id: string; clip_id: string
  started_at: string; finished_at: string | null
  per_stage_timings: Record<string, { latency_ms: number; cost_usd: number }>
  per_stage_io: Record<string, { input_preview: string; output_preview: string; raw_response: unknown }>
  metrics: { wer?: number; eer?: number; mos?: number; total_latency_ms: number; total_cost_usd: number }
  error: string | null
}
```

## Repository layout

```
trial-app/                          (= github.com/Eden-kk/audio-model-pipeline-trial)
├── backend/
│   ├── main.py                      # FastAPI entrypoint
│   ├── pipelines/
│   │   ├── runner.py                # generic DAG executor
│   │   ├── recipes.py               # P0 pre-built recipes
│   │   └── streaming.py             # WebSocket protocol for live runs
│   ├── ingest/
│   │   ├── recorder.py              # WS-based audio capture
│   │   ├── upload.py                # multipart upload + ffmpeg normalize
│   │   ├── tagger.py                # auto-extract metadata
│   │   └── store.py                 # Modal Volume layout + manifest CRUD
│   ├── adapters/
│   │   ├── registry.py              # category-typed Adapter registry
│   │   └── (imports ambient-deploy/benchmarks/adapters/* — no copy)
│   ├── metrics/                     # imports from benchmarks/metrics.py
│   └── storage/{clips.py,runs.py}
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Playground.tsx       # single-model quick test
│   │   │   ├── Pipelines.tsx        # picker (P0) + composer (P1)
│   │   │   ├── Run.tsx              # live DAG execution
│   │   │   ├── Corpus.tsx           # clip library
│   │   │   ├── Compare.tsx          # P1 only
│   │   │   └── Settings.tsx
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── PipelineGraph.tsx    # react-flow wrapper
│   │   │   ├── StagePanel.tsx       # right slide-out drill-in
│   │   │   ├── WaveformScrubber.tsx # synced playhead
│   │   │   ├── MetricsCard.tsx
│   │   │   ├── ScenarioChips.tsx
│   │   │   └── AdapterCard.tsx      # category palette card
│   │   └── hooks/useRunStream.ts
│   └── package.json
├── modal_app.py                     # Modal deploy entry
└── README.md
```

## Reuse map

| Need | Existing artifact | Path (in `ambient-deploy`) |
|------|-------------------|------|
| ASR adapters (11 cloud + 1 self-host) | wrap with category metadata | `benchmarks/adapters/{deepgram,openai_transcribe,parakeet_modal,...}.py` |
| TTS adapters (4 cloud) | wrap | `benchmarks/adapters/{cartesia_tts,elevenlabs_tts,openai_tts,rev_ai}.py` |
| Speaker-verify (3) | wrap | `benchmarks/adapters/{pyannote_verify,resemblyzer,picovoice_eagle}.py` |
| Realtime E2E (P1 only) | wrap | `benchmarks/adapters/{gemini_live_e2e,moshi_e2e,stitched_pipeline}.py` |
| WER/EER/MOS/DER metrics | import directly | `benchmarks/metrics.py` |
| Modal deploy template | adapt | `deployments/parakeet-asr/serve.py` |
| WebSocket UI server pattern | adapt | `deployments/whisper-v3-pure/scripts/12-decoupled-loop/00_ui_server.py` |
| Test corpora | optional importer in P1 | `benchmarks/test_clips/{en,zh,verification_noise,...}` |

## Verification plan

1. `cd trial-app/backend && python main.py` runs locally; `curl localhost:8000/api/adapters` returns the 6 MVP adapters with category + ports declared.
2. `cd trial-app/frontend && pnpm dev` opens; sidebar shows Playground / Pipelines / Run / Corpus / Settings.
3. **Playground:** record a 5 s mic clip → pick category=ASR → pick Deepgram → run → transcription renders in < 2 s with latency badge.
4. **Pipelines → Run:** open `ASR-then-TTS-roundtrip` recipe → click Run on the same clip → both stages light up sequentially; stage drill-in shows ASR text and TTS audio output.
5. **Corpus:** the recorded clip appears with auto-extracted duration/SNR/language; tag it with `outdoor-traffic`; filter chip shows only it.
6. **Wiring constraint demo (manual test):** in P1, attempting to wire ASR's `text` output to Speaker-verify's `audio_enroll` input is rejected with a port-type error.
7. **JSONL persistence:** `cat /vol/trial-app/runs/scenario=outdoor-traffic/pipeline=asr_then_tts.jsonl` shows one line per run with full metrics.
8. **Cost guardrail:** with `MAX_DAILY_COST_USD=1.00` set, the 1001st cent of API spend in a day is rejected with a clear UI error.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| WebRTC mic in Safari iOS quirky | Fall back to `MediaRecorder` + manual upload on iOS Safari; document the constraint |
| Adapter failures during multi-stage run | Stage runner emits `StageFailed`; downstream skipped; UI shows red node + error preview |
| Cost runaway | Frontend shows estimated cost before run; backend caps daily spend per adapter via env var |
| Modal cold-start adds 90 s on Parakeet (P1) | Frontend shows "warming model..." with progress; pre-warm cron for top 3 adapters |
| Storage growth | Manifests are compact JSON; raw clips age out after 90 days unless tagged `permanent` |
| Scope creep on the composer | P0 forbids the custom composer entirely — recipes only; composer ships in P1 |
| Adapter API drift | Each adapter has a smoke test in CI hitting the real API on a 3 s reference clip; nightly schedule |

## Phasing summary

| Phase | Duration | Outcome | Plan file |
|-------|----------|---------|-----------|
| **P0 — Audio MVP** | ~3 weeks | Web app with sidebar IA, 6 adapters, 4 recipes, port-typed visualization, manual tagging, JSONL results | this plan |
| **P1 — Composer + omni + compare** | ~4 weeks | Drag-drop composer, video + omni adapters, side-by-side compare, MOS rating, +9 adapters | this plan |
| **P2 — iOS + auto-classify + field corpus** | TBD | On-device benchmarks, LLM auto-classification, AR-glass field corpus, dataset export | **separate plan, written when P1 ships** |

Total path to v1.5: ~7 weeks. iOS + advanced features deliberately deferred.
