import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import MicRecorder from '../components/MicRecorder'
import MicStream from '../components/MicStream'
import AudioFileDrop from '../components/AudioFileDrop'
import {
  listAdapters,
  listClips,
  clipAudioUrl,
  uploadClip,
  startRun,
  getRun,
  connectRunWS,
  type Adapter,
  type Clip,
  type RunEvent,
} from '../lib/api'
import { cx } from '../lib/cx'
import { formatLanguage } from '../lib/lang'
import LangSupportBadge, { langSupport } from '../components/LangSupportBadge'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StageResult {
  stage_id?: string
  transcript?: string
  latency_ms?: number
  raw_response?: unknown
  error?: string
  // Streaming-only — renders a typing cursor + monotonically growing text
  is_streaming?: boolean
  partial_count?: number
  is_final?: boolean
}

type RunState = 'idle' | 'uploading' | 'running' | 'done' | 'error'

// ---------------------------------------------------------------------------
// Latency badge — light, pill-shaped, color tracks performance
// ---------------------------------------------------------------------------

function LatencyBadge({ ms }: { ms: number }) {
  const color =
    ms < 500 ? 'bg-green-100 text-green-800 border-green-200'
    : ms < 2000 ? 'bg-amber-100 text-amber-800 border-amber-200'
    : 'bg-red-100 text-red-800 border-red-200'
  return (
    <span className={cx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono border', color)}>
      {ms.toFixed(0)} ms
    </span>
  )
}

// ---------------------------------------------------------------------------
// Playground page
// ---------------------------------------------------------------------------

export default function Playground() {
  const [searchParams, setSearchParams] = useSearchParams()
  const preselectedClipId = searchParams.get('clip')

  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [adaptersError, setAdaptersError] = useState<string | null>(null)
  const [selectedAdapter, setSelectedAdapter] = useState<string>('')
  const [selectedLanguage, setSelectedLanguage] = useState<string>('')
  const [pendingBlob, setPendingBlob] = useState<{ blob: Blob; mime: string } | null>(null)
  // When the user arrives via Corpus → "Run in Playground", we deep-link
  // through `?clip=<id>` and run directly against the existing corpus row
  // (no re-upload). The clip metadata is fetched once on mount.
  const [preselectedClip, setPreselectedClip] = useState<Clip | null>(null)
  const [preselectError, setPreselectError] = useState<string | null>(null)
  const [runState, setRunState] = useState<RunState>('idle')
  const [statusMsg, setStatusMsg] = useState<string>('')
  const [events, setEvents] = useState<RunEvent[]>([])
  const [results, setResults] = useState<StageResult[]>([])
  const [runError, setRunError] = useState<string | null>(null)
  const [inputMode, setInputMode] = useState<'record' | 'upload' | 'mic-stream'>('upload')
  const [language, setLanguage] = useState<string>('auto')
  // True once a mic-stream WS has opened (StageStarted received); reset when
  // the stream ends. Used to lock the language picker for mic-stream mode.
  const [micStreamActive, setMicStreamActive] = useState(false)

  // Load adapters on mount
  useEffect(() => {
    listAdapters('asr')
      .then((list) => {
        setAdapters(list)
        if (list.length > 0) setSelectedAdapter(list[0].id)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err)
        setAdaptersError(msg)
      })
  }, [])

  // Resolve `?clip=<id>` from the URL — fetched after mount so the clip
  // panel renders with full metadata (filename, duration, scenarios)
  // rather than a bare ID.
  useEffect(() => {
    if (!preselectedClipId) {
      setPreselectedClip(null)
      setPreselectError(null)
      return
    }
    let cancelled = false
    setPreselectError(null)
    listClips()
      .then((clips) => {
        if (cancelled) return
        const found = clips.find((c) => c.id === preselectedClipId)
        if (!found) {
          setPreselectError(
            `Clip ${preselectedClipId.slice(0, 8)}… not found — was it deleted?`,
          )
          setPreselectedClip(null)
          return
        }
        setPreselectedClip(found)
        // Drop any pending blob so the picked clip is unambiguously "the input".
        setPendingBlob(null)
        // Force input-mode to upload so the audio preview area is visible
        // (mic-stream mode hides everything below it).
        setInputMode('upload')
        setRunState('idle')
        setResults([])
        setRunError(null)
        setEvents([])
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setPreselectError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [preselectedClipId])

  // Recompute language picker default when adapter or clip changes
  useEffect(() => {
    const adapter = adapters.find(a => a.id === selectedAdapter)
    if (!adapter?.multilang) { setSelectedLanguage(''); return }
    const langs = adapter.supported_languages ?? []
    const fromClip = preselectedClip?.language_detected
    setSelectedLanguage(
      (fromClip && langs.includes(fromClip)) ? fromClip : (langs[0] ?? '')
    )
  }, [selectedAdapter, adapters, preselectedClip?.language_detected, preselectedClip?.id])

  // Mic-stream picker default — pick "auto" when supported (realtime
  // adapters), otherwise the first valid code so the <select> value
  // always matches an actual <option> the adapter accepts.
  useEffect(() => {
    const adapter = adapters.find(a => a.id === selectedAdapter)
    if (!adapter) return
    const langs = adapter.supported_languages ?? []
    if (langs.length === 0) return
    setLanguage(langs.includes('auto') ? 'auto' : langs[0])
  }, [selectedAdapter, adapters])

  const handleBlob = useCallback((blob: Blob, mime: string) => {
    setPendingBlob({ blob, mime })
    // A fresh blob takes precedence over the deep-linked clip — clear it
    // so handleRun() doesn't misroute the run.
    setPreselectedClip(null)
    if (preselectedClipId) {
      // Remove `?clip=` from the URL so a refresh doesn't restore the clip.
      setSearchParams({}, { replace: true })
    }
    setRunState('idle')
    setResults([])
    setRunError(null)
    setEvents([])
  }, [preselectedClipId, setSearchParams])

  function clearPreselection() {
    setPreselectedClip(null)
    setSearchParams({}, { replace: true })
  }

  async function handleRun() {
    if (!selectedAdapter) return
    if (!pendingBlob && !preselectedClip) return

    setResults([])
    setRunError(null)
    setEvents([])

    // Two run paths:
    //   1. Deep-linked corpus clip — skip upload, run directly by clip_id
    //   2. Fresh blob from upload/record — upload first, then run
    let clip: { id: string }
    if (preselectedClip) {
      clip = { id: preselectedClip.id }
      setRunState('running')
      setStatusMsg('Starting run…')
    } else {
      setRunState('uploading')
      setStatusMsg('Uploading clip…')
      try {
        clip = await uploadClip(pendingBlob!.blob, pendingBlob!.mime)
        setStatusMsg('Starting run…')
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err)
        setRunError(`Upload failed: ${msg}`)
        setRunState('error')
        return
      }
    }

    setRunState('running')
    setStatusMsg('Running…')

    const isStreaming = !!selectedMeta?.is_streaming
    const collected: StageResult[] = [{
      stage_id: selectedAdapter,
      transcript: '',
      is_streaming: isStreaming,
      partial_count: 0,
      is_final: false,
    }]
    setResults([...collected])
    let wsCleanup: (() => void) | null = null

    // Kick off the POST. For streaming adapters, the backend returns
    // immediately with an in-flight run_id; for batch, it waits for the
    // synchronous transcribe to finish.
    const cfg: Record<string, unknown> = {}
    if (selectedLanguage) cfg.language = selectedLanguage
    const postPromise = startRun(clip.id, selectedAdapter, cfg).catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      setRunError(`Run failed: ${msg}`)
      setRunState('error')
      return null
    })

    const run = await postPromise
    if (!run) return

    if (run.error) {
      setRunError(`Adapter error: ${run.error}`)
      setRunState('error')
      return
    }

    // ── Streaming branch: response carries empty text; WS will stream
    // partials in real time. We open WS, render token-by-token, mark done
    // on StageCompleted.
    // ── Batch branch:    response carries the full text; we render that
    // immediately, then attach WS as a passive overlay (it'll deliver one
    // StagePartial + StageCompleted that match the response).
    if (!isStreaming) {
      const text = run.result?.text ?? run.output_preview ?? ''
      collected[0] = {
        ...collected[0],
        transcript: text,
        latency_ms: run.latency_ms ?? undefined,
        raw_response: run.result?.raw_response ?? run.result,
        is_final: true,
      }
      setResults([...collected])
      setRunState('done')
      setStatusMsg('Done')
    } else {
      setStatusMsg('Streaming…')
    }

    wsCleanup = connectRunWS(
      run.id,
      (ev) => {
        setEvents((prev) => [...prev, ev])
        const etype = ev.event ?? ev.type

        if (etype === 'StageProgress') {
          const partial = ev.partial_text ?? ''
          collected[0] = {
            ...collected[0],
            transcript: partial,
            partial_count: (collected[0].partial_count ?? 0) + 1,
            is_final: false,
          }
          setResults([...collected])
        }

        if (etype === 'StageCompleted') {
          const finalText = (ev.result?.text as string | undefined)
                         ?? collected[0]?.transcript
                         ?? ''
          collected[0] = {
            ...collected[0],
            transcript: finalText,
            latency_ms: ev.latency_ms ?? collected[0].latency_ms,
            raw_response: (ev.result?.raw_response as unknown)
                       ?? ev.result
                       ?? collected[0].raw_response,
            is_final: true,
          }
          setResults([...collected])
          setRunState('done')
          setStatusMsg('Done')
          wsCleanup?.()

          // Defensive reconcile: fetch the canonical run record from REST
          // and force-overwrite the transcript. The WS path occasionally
          // lands the user a stale partial (we've seen Deepgram's accumulator
          // fall short on long clips); the saved run always has the full
          // text. Functional setResults to avoid stale-closure issues, and
          // a 250 ms delay so append_run() definitely lands first.
          const reconcileRunId = run.id
          window.setTimeout(() => {
            getRun(reconcileRunId).then((authoritative) => {
              const canonical = (authoritative.result?.text ?? '').trim()
              if (!canonical) return
              setResults((prev) => {
                const head = prev[0] ?? {
                  stage_id: selectedAdapter,
                  is_streaming: true,
                }
                if ((head.transcript ?? '').trim() === canonical) {
                  return prev   // already shows the canonical text
                }
                return [{
                  ...head,
                  transcript: canonical,
                  latency_ms: authoritative.latency_ms ?? head.latency_ms,
                  raw_response: authoritative.result?.raw_response
                             ?? authoritative.result
                             ?? head.raw_response,
                  is_final: true,
                }]
              })
            }).catch(() => { /* best-effort */ })
          }, 250)
        }

        if (etype === 'StageFailed') {
          setRunError(ev.error ?? 'Unknown error')
          setRunState('error')
          wsCleanup?.()
        }
      },
    )
  }

  const busy = runState === 'uploading' || runState === 'running'
  const hasAudio = pendingBlob !== null || preselectedClip !== null
  const canRun = hasAudio && !!selectedAdapter && !busy

  const primaryResult = results.find((r) => r.transcript)
  const selectedMeta = adapters.find((a) => a.id === selectedAdapter)

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">Playground</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Quick-test a single ASR model against a mic recording or uploaded file (audio or video).
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6 flex flex-col gap-6">
        {/* ── Top section: config + input ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Adapter picker */}
          <div className="card flex flex-col gap-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Adapter</h2>

            {adaptersError ? (
              <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                No backend connected — adapter list unavailable.
                <span className="block text-xs text-amber-700 mt-1">{adaptersError}</span>
              </div>
            ) : adapters.length === 0 ? (
              <div className="text-sm text-gray-500 italic">Loading adapters…</div>
            ) : (
              <select
                value={selectedAdapter}
                onChange={(e) => setSelectedAdapter(e.target.value)}
                disabled={busy}
                className="field w-full"
              >
                {adapters.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.is_streaming ? '● ' : '○ '}
                    {a.display_name} — {a.vendor}
                  </option>
                ))}
              </select>
            )}

            {selectedMeta && (
              <div className="flex flex-wrap gap-1.5 text-xs">
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200">
                  {selectedMeta.category}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200">
                  {selectedMeta.hosting}
                </span>
                {selectedMeta.is_streaming ? (
                  <span className="px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 font-medium">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-red-500 mr-1 align-middle animate-pulse" />
                    stream
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 border border-gray-200">
                    batch
                  </span>
                )}
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200 font-mono">
                  {selectedMeta.id}
                </span>
                <LangSupportBadge adapter={selectedMeta} />
              </div>
            )}

            {selectedMeta && langSupport(selectedMeta) === 'single' && (
              <p className="text-xs text-gray-400">
                {(selectedMeta.supported_languages?.[0] ?? 'en').toUpperCase()} only
                {' '}— no language to pick.
              </p>
            )}
            {selectedMeta && langSupport(selectedMeta) !== 'single' && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500" htmlFor="lang-select">
                  Language
                  {inputMode === 'mic-stream' && micStreamActive && (
                    <span
                      className="ml-1.5 text-amber-600"
                      title="Language is locked once the stream starts"
                    >
                      (locked — stream active)
                    </span>
                  )}
                </label>
                <select
                  id="lang-select"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  disabled={busy || (inputMode === 'mic-stream' && micStreamActive)}
                  className="field w-full"
                  title={
                    inputMode === 'mic-stream' && micStreamActive
                      ? 'Language is locked once the stream starts'
                      : undefined
                  }
                >
                  {(selectedMeta.supported_languages ?? []).map((code) => (
                    <option key={code} value={code}>
                      {formatLanguage(code)}
                    </option>
                  ))}
                </select>
                {inputMode === 'mic-stream' && !micStreamActive && (
                  <p className="text-xs text-gray-400">
                    {langSupport(selectedMeta) === 'realtime'
                      ? 'Detects language switches mid-stream. Pick a specific code only if you want to lock to one.'
                      : 'Locks at session start; cannot change once streaming begins.'}
                  </p>
                )}
              </div>
            )}

            {adaptersError && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Or enter adapter ID manually</label>
                <input
                  type="text"
                  value={selectedAdapter}
                  onChange={(e) => setSelectedAdapter(e.target.value)}
                  disabled={busy}
                  placeholder="e.g. deepgram"
                  className="field w-full"
                />
              </div>
            )}
          </div>

          {/* Audio input */}
          <div className="card flex flex-col gap-4">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Audio Input</h2>

            {/* Deep-linked clip from Corpus → "Run in Playground". When set,
                the run uses this clip directly (no re-upload). The tabs
                below stay enabled so the user can swap to a fresh recording
                if they want — that clears the preselection. */}
            {preselectedClip && (
              <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 flex flex-col gap-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-semibold text-blue-900 uppercase tracking-wider">
                      Selected from Corpus
                    </div>
                    <div className="mt-1 text-sm text-gray-900 truncate">
                      {preselectedClip.original_filename || preselectedClip.filename}
                    </div>
                    <div className="text-xs text-gray-600 mt-0.5">
                      {preselectedClip.duration_s.toFixed(1)}s ·{' '}
                      {preselectedClip.sample_rate} Hz ·{' '}
                      {preselectedClip.channels} ch
                      {preselectedClip.language_detected && (
                        <> · {preselectedClip.language_detected}</>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={clearPreselection}
                    disabled={busy}
                    className="text-xs text-blue-700 hover:text-blue-900 underline disabled:opacity-50"
                    title="Use a fresh upload/recording instead"
                  >
                    Clear
                  </button>
                </div>
                <audio
                  controls
                  src={clipAudioUrl(preselectedClip.id)}
                  className="w-full h-9"
                />
                {preselectedClip.scenarios.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {preselectedClip.scenarios.slice(0, 6).map((s) => (
                      <span
                        key={s}
                        className="px-1.5 py-0.5 rounded-full bg-white text-blue-800 text-[10px] border border-blue-200"
                      >
                        {s}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            {preselectError && (
              <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                {preselectError}
              </div>
            )}

            <div className="flex gap-2 flex-wrap">
              {([
                { id: 'upload', label: 'File upload', requiresStreaming: false },
                { id: 'record', label: 'Mic record', requiresStreaming: false },
                { id: 'mic-stream', label: 'Mic stream (live)', requiresStreaming: true },
              ] as const).map(({ id, label, requiresStreaming }) => {
                const blocked = requiresStreaming && !selectedMeta?.is_streaming
                return (
                  <button
                    key={id}
                    type="button"
                    disabled={busy || blocked}
                    onClick={() => { setInputMode(id); setMicStreamActive(false) }}
                    title={blocked ? 'Pick a streaming-capable adapter (●) first' : undefined}
                    className={cx(
                      inputMode === id ? 'btn-pill-dark' : 'btn-pill-outline',
                      'text-xs',
                      (busy || blocked) && 'opacity-50 cursor-not-allowed',
                    )}
                  >
                    {label}
                  </button>
                )
              })}
            </div>

            {inputMode === 'upload' && (
              <AudioFileDrop onBlob={handleBlob} disabled={busy} />
            )}
            {inputMode === 'record' && (
              <MicRecorder onBlob={handleBlob} disabled={busy} />
            )}
            {inputMode === 'mic-stream' && selectedMeta?.is_streaming && (
              <MicStream
                adapter={selectedAdapter}
                language={language}
                disabled={busy}
                // The user clicking Stop in MicStream resets its internal
                // `streaming` flag immediately, but our `runState` would stay
                // 'running' until the backend's StageCompleted arrives 200-
                // 500 ms later — leaving the start button disabled in the
                // meantime. Mirror the local-stop here so the user can kick
                // off another stream right away.
                onLocalStop={() => {
                  setMicStreamActive(false)
                  if (runState === 'running') {
                    setRunState('done')
                    setStatusMsg('Done')
                    setResults((prev) => {
                      if (prev.length === 0) return prev
                      const head = prev[0]
                      return [{ ...head, is_final: true }]
                    })
                  }
                }}
                onEvent={(ev) => {
                  setEvents((prev) => [...prev, ev])
                  if (ev.event === 'StageStarted') {
                    setMicStreamActive(true)
                    setRunState('running')
                    setStatusMsg('Streaming…')
                    setRunError(null)
                    setResults([{
                      stage_id: selectedAdapter,
                      transcript: '',
                      is_streaming: true,
                      partial_count: 0,
                      is_final: false,
                    }])
                  }
                  if (ev.event === 'StageProgress') {
                    setResults((prev) => {
                      const r = prev[0] ?? {
                        stage_id: selectedAdapter,
                        is_streaming: true,
                      }
                      return [{
                        ...r,
                        transcript: ev.partial_text ?? '',
                        partial_count: (r.partial_count ?? 0) + 1,
                        is_final: false,
                      }]
                    })
                  }
                  if (ev.event === 'StageCompleted') {
                    setMicStreamActive(false)
                    setResults((prev) => {
                      const r = prev[0] ?? { stage_id: selectedAdapter, is_streaming: true }
                      return [{
                        ...r,
                        transcript: ev.result?.text ?? r.transcript ?? '',
                        latency_ms: ev.latency_ms,
                        raw_response: ev.result,
                        is_final: true,
                      }]
                    })
                    setRunState('done')
                    setStatusMsg('Done')
                  }
                  if (ev.event === 'StageFailed') {
                    setMicStreamActive(false)
                    setRunError(ev.error ?? 'Streaming failed')
                    setRunState('error')
                  }
                }}
              />
            )}

            {/* "Audio ready" status — guard on pendingBlob, not hasAudio, since
                a deep-linked preselected clip also flips hasAudio true but has
                no blob (its own panel above already shows the clip metadata). */}
            {pendingBlob && inputMode !== 'mic-stream' && (
              <p className="text-xs text-green-700">
                Audio ready — {(pendingBlob.blob.size / 1024).toFixed(1)} KB ({pendingBlob.mime})
              </p>
            )}
          </div>
        </div>

        {/* Language picker — only shown for multi-lang adapters */}
        {selectedMeta?.multilang && (
          <div className="card flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Language</h2>
              <LangSupportBadge adapter={selectedMeta} />
            </div>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              disabled={busy}
              className="field w-full max-w-xs"
            >
              {(selectedMeta.supported_languages ?? []).map((code) => (
                <option key={code} value={code}>
                  {formatLanguage(code)}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-400">
              {langSupport(selectedMeta) === 'realtime'
                ? 'Pick "auto" to let the model detect.'
                : 'Locked at run start.'}
            </p>
          </div>
        )}

        {/* Run button — hidden in mic-stream mode (the stream button itself starts the run) */}
        {inputMode !== 'mic-stream' && (
        <div className="flex items-center gap-4">
          <button
            type="button"
            disabled={!canRun}
            onClick={() => void handleRun()}
            className="btn-pill-dark px-6 py-2.5"
          >
            {busy ? statusMsg : 'Run'}
          </button>
          {busy && (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
              {statusMsg}
            </div>
          )}
        </div>
        )}

        {/* ── Results section ── */}
        {runError && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            {runError}
          </div>
        )}

        {(results.length > 0 || runState === 'done') && (
          <div className="card overflow-hidden p-0 shrink-0">
            <div className="px-5 py-3 border-b border-gray-200 flex items-center gap-3 bg-gray-50">
              <span className="text-sm font-semibold text-gray-700">Transcript</span>
              {primaryResult?.latency_ms != null && (
                <LatencyBadge ms={primaryResult.latency_ms} />
              )}
            </div>

            <div className="px-5 py-4">
              {primaryResult?.transcript ? (
                <p className="text-gray-900 leading-relaxed whitespace-pre-wrap break-words">
                  {primaryResult.transcript}
                  {primaryResult.is_streaming && !primaryResult.is_final && (
                    <span
                      className="inline-block w-2 h-5 bg-gray-900 ml-0.5 align-middle animate-pulse"
                      aria-hidden="true"
                    />
                  )}
                </p>
              ) : (
                <p className="text-gray-500 italic text-sm">
                  {runState === 'done' ? 'No transcript returned.'
                    : primaryResult?.is_streaming ? 'Connecting to streaming model…'
                    : 'Waiting for transcript…'}
                </p>
              )}
              {primaryResult?.is_streaming && (primaryResult.partial_count ?? 0) > 0 && (
                <p className="text-xs text-gray-500 mt-2">
                  {primaryResult.is_final
                    ? `final after ${primaryResult.partial_count} partial${primaryResult.partial_count === 1 ? '' : 's'}`
                    : `partial #${primaryResult.partial_count} · streaming live`}
                </p>
              )}
            </div>

            {results.length > 1 && (
              <div className="px-5 pb-4 flex flex-col gap-2 border-t border-gray-100 pt-3">
                {results.map((r, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-gray-500">
                    <span className="font-mono">{r.stage_id ?? `stage_${i}`}</span>
                    {r.latency_ms != null && <LatencyBadge ms={r.latency_ms} />}
                  </div>
                ))}
              </div>
            )}

            {primaryResult?.raw_response != null && (
              <div className="px-5 pb-4 border-t border-gray-100 pt-3">
                <details>
                  <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-800">Raw response</summary>
                  <pre className="mt-2 text-xs text-gray-700 bg-gray-50 rounded-lg p-3 overflow-auto max-h-64 border border-gray-200">
                    {JSON.stringify(primaryResult.raw_response, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        )}

        {events.length > 0 && (
          <details className="text-xs">
            <summary className="text-gray-500 cursor-pointer hover:text-gray-800 select-none">
              WebSocket events ({events.length})
            </summary>
            <div className="mt-2 bg-white border border-gray-200 rounded-lg p-3 overflow-auto max-h-48 space-y-1">
              {events.map((ev, i) => (
                <div key={i} className="font-mono text-gray-700">
                  <span className="text-gray-900 font-semibold">{ev.event ?? ev.type}</span>
                  {ev.adapter && <span className="text-gray-500 ml-1">({ev.adapter})</span>}
                  {ev.partial_text != null && (
                    <span className="text-gray-500 ml-1">
                      → {ev.partial_text.slice(0, 50)}{ev.partial_text.length > 50 ? '…' : ''}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}
