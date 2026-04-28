# `backend/data/` — local persistence layout

This directory holds everything the trial-app writes to disk. None of it
is committed (the parent `.gitignore` excludes it); back it up by hand
before destructive ops.

## Layout

```
backend/data/
├── clips/<clip_id>/
│   ├── audio.wav              ← canonical PCM the adapters read
│   ├── source.<orig>          ← only for video uploads (kept for re-extract)
│   └── manifest.json          ← Clip dataclass (storage/clips.py)
├── enrollments/
│   └── <profile_id>.json      ← speaker-verify reference embeddings
├── runs/
│   └── runs.jsonl             ← append-only Run records (one per /api/runs)
└── haoclaw_outbox.jsonl       ← slow-loop intent envelopes (Plan C-extended)
```

## AR-glass benchmark capture flow (Plan D)

When you record live audio through `/playground` → "Stream from mic"
with the **Save to corpus** checkbox ticked, the backend:

1. Streams 16 kHz PCM to the chosen vendor (Deepgram or AssemblyAI) for
   real-time transcription, *and* accumulates the bytes into a buffer.
2. On stop (or 5-minute cap), writes a 16-bit mono WAV under
   `clips/<new_id>/audio.wav` and a manifest with:
   - `source: "live-mic"`
   - `scenarios: ["ar-glass-capture", "live-mic", "vendor-{deepgram|assemblyai}", ...]`
   - `captured_transcript`: the vendor's final stitched text (a *seed*,
     not ground truth — see below)
   - `captured_transcript_segments`: list of `{start, end, text, is_final}`
     per finalized vendor turn
   - audio-derived auto-tags: `lang-{xx}`, `snr-clean|mid|noisy`,
     `mostly-speech|partial-speech|mostly-silence`,
     `single-speaker|multi-speaker`, `short|medium|long-form`
3. Emits a `ClipSaved` WS event with the new `clip_id` so the UI
   can deep-link.

## Refining `captured_transcript` into a ground-truth reference

The vendor's streaming output is good but not perfect — Deepgram tends
to over-correct, AssemblyAI's universal-streaming model adds extra
punctuation. Before using a captured clip as a benchmark reference,
hand-correct it:

```bash
# 1. List capture clips
curl -s http://localhost:8000/api/clips | jq '
  .clips[] | select(.scenarios | index("ar-glass-capture")) |
  {id, duration_s, captured_transcript}'

# 2. Open the audio in any player
open backend/data/clips/<clip_id>/audio.wav

# 3. Edit the transcript string and PATCH it back
curl -X PATCH http://localhost:8000/api/clips/<clip_id> \
  -H 'Content-Type: application/json' \
  -d '{"captured_transcript": "the corrected reference text here"}'
```

The `captured_transcript_segments` list is preserved as-is unless you
also patch it (timing rarely needs editing).

## Filtering captures from the Corpus page

In the UI:

- click the `ar-glass-capture` scenario chip in the filter bar to show
  only your captures
- the `vendor-deepgram` / `vendor-assemblyai` chips let you A/B which
  streaming engine produced cleaner reference text

## Building a benchmark export

A future Plan D follow-up will add `scripts/export_benchmark.py` which
walks `clips/`, picks `ar-glass-capture` rows with non-empty
`captured_transcript`, and emits `refs.jsonl` in the same shape the
ambient-deploy benchmark harness expects.

## Cleanup

To drop everything and start fresh:

```bash
rm -rf backend/data/clips/*       # corpus
rm -rf backend/data/runs/*        # run history
rm -f  backend/data/haoclaw_outbox.jsonl   # slow-loop outbox
# enrollments/ and api keys are kept by default
```

The `/api/clips` and `/api/runs/recipe` endpoints will recreate any
manifests they need.
