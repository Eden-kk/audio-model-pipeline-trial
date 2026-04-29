import { useEffect, useMemo, useRef, useState } from 'react'
import {
  listAdapters,
  type Adapter,
} from '../lib/api'
import { startOmniSession, type OmniEvent, type OmniSessionHandle } from '../lib/wsAudio'
import { startCameraCapture, type CameraCaptureHandle } from '../lib/cameraStream'
import { startVAD, type TurnDetectorHandle } from '../lib/turnDetector'
import { cx } from '../lib/cx'

// ─── Local types ─────────────────────────────────────────────────────────

/** Slice O6.3: two modes for A/B comparison.
 *
 *   no-interrupt     VAD fires `flush()` on silence; if user starts talking
 *                    again while model is still replying, audio keeps playing
 *                    and the new utterance queues for after the reply finishes.
 *
 *   interruptable    Same VAD-driven flush, BUT user-speech-start mid-reply
 *                    immediately cancels the model's audio playback +
 *                    sends `{event:'interrupt'}` so the server-side adapter
 *                    drops further response chunks.
 */
type ConversationMode = 'no-interrupt' | 'interruptable'

type SessionState = 'idle' | 'opening' | 'listening' | 'speaking' | 'awaiting' | 'replying' | 'closing' | 'error'

interface CaptionLine {
  /** Stable key for React. */
  id: number
  /** 'partial' = still streaming; 'final' = locked-in. */
  kind: 'partial' | 'final' | 'response' | 'tool' | 'meta'
  text: string
  ts_ms: number
}

interface LatencyDial {
  ttfa_ms: number | null    // time-to-first-event after flush
  total_ms: number | null   // flush → done
  cost_usd: number          // accumulated session cost
}

interface WearerSnapshot {
  user_segments: number
  stranger_segments: number
  n_segments: number
  window_s: number
  /** Wall-clock when received — drives staleness in the UI. */
  ts_ms: number
}

// ─── Page ────────────────────────────────────────────────────────────────

export default function Realtime() {
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [adapterId, setAdapterId] = useState<string>('')
  const [state, setState] = useState<SessionState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [captions, setCaptions] = useState<CaptionLine[]>([])
  const [dial, setDial] = useState<LatencyDial>({ ttfa_ms: null, total_ms: null, cost_usd: 0 })

  const sessionRef = useRef<OmniSessionHandle | null>(null)
  // Wall-clock timer for the in-flight utterance — set on `flush()`, read
  // when the first response event lands so we can compute TTFA on the
  // client side independently of the server's latency_ms.
  const flushAtMsRef = useRef<number>(0)
  const captionIdRef = useRef<number>(0)

  // ── Slice O6.3: conversation mode + VAD state
  const [mode, setMode] = useState<ConversationMode>('no-interrupt')
  const [vadLevel, setVadLevel] = useState(0)         // 0..1 — drives VU meter
  const [vadActive, setVadActive] = useState(false)   // currently in a speech burst
  const [interruptCount, setInterruptCount] = useState(0)
  const vadRef = useRef<TurnDetectorHandle | null>(null)
  // Stable refs for state (used inside VAD callbacks which capture once)
  const stateRef = useRef<SessionState>('idle')
  const modeRef = useRef<ConversationMode>('no-interrupt')
  useEffect(() => { stateRef.current = state }, [state])
  useEffect(() => { modeRef.current = mode }, [mode])

  // ── Wearer-tag heartbeat (Slice O5)
  const [wearerEnrolled, setWearerEnrolled] = useState<boolean | null>(null)
  const [wearer, setWearer] = useState<WearerSnapshot | null>(null)

  // ── Camera state (Slice O3)
  const [cameraOn, setCameraOn] = useState(false)
  const [fps, setFps] = useState(2)
  const [framesSent, setFramesSent] = useState(0)
  const [bytesSent, setBytesSent] = useState(0)
  const [lastFrameUrl, setLastFrameUrl] = useState<string | null>(null)
  const videoElRef = useRef<HTMLVideoElement | null>(null)
  const cameraRef = useRef<CameraCaptureHandle | null>(null)

  // ── Load realtime_omni adapters on mount
  useEffect(() => {
    listAdapters('realtime_omni')
      .then((list) => {
        setAdapters(list)
        if (list.length > 0) setAdapterId(list[0].id)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  // ── Push a caption line — partials replace any previous partial line.
  const pushCaption = (kind: CaptionLine['kind'], text: string) => {
    setCaptions((prev) => {
      const next = kind === 'partial'
        ? prev.filter((l) => l.kind !== 'partial')
        : prev
      return [...next, { id: ++captionIdRef.current, kind, text, ts_ms: Date.now() }]
    })
  }

  // ── Start / stop handlers (Slice O6.3 — auto-turn always on)
  async function handleStart() {
    if (!adapterId) return
    setError(null)
    setCaptions([])
    setDial({ ttfa_ms: null, total_ms: null, cost_usd: 0 })
    setInterruptCount(0)
    setState('opening')
    try {
      const session = await startOmniSession({
        adapter: adapterId,
        onEvent: handleEvent,
        onError: (e) => {
          setError(e.message)
          setState('error')
        },
      })
      sessionRef.current = session

      // Spin up Silero VAD now. Same MicVAD instance lives until handleStop
      // tears it down. The VAD has its own getUserMedia call but the
      // browser hands back the same physical mic to both (mic permission
      // already granted by startOmniSession).
      try {
        const vad = await startVAD({
          onSpeechStart: () => {
            setVadActive(true)
            // Slice O6.3 interrupt: if the model is mid-reply when the
            // user starts speaking again AND we're in interruptable mode,
            // cancel playback + signal the adapter to drop response chunks.
            if (
              modeRef.current === 'interruptable'
              && (stateRef.current === 'replying' || stateRef.current === 'awaiting')
            ) {
              const cancelled = sessionRef.current?.cancelPlayback() ?? 0
              sessionRef.current?.sendControl({ event: 'interrupt' })
              setInterruptCount((c) => c + 1)
              pushCaption('meta', `⚡ interrupt — cancelled ${cancelled} queued audio chunks`)
            }
            setState('speaking')
          },
          onSpeechEnd: () => {
            setVadActive(false)
            // Auto-flush: user finished talking. Adapter will accumulate
            // PCM frames the worklet has already pushed and emit a request.
            if (sessionRef.current) {
              flushAtMsRef.current = performance.now()
              sessionRef.current.flush()
              setState('awaiting')
              setDial((d) => ({ ...d, ttfa_ms: null, total_ms: null }))
              pushCaption('meta', '↳ silence detected — flushing utterance')
            }
          },
          onFrameProcessed: (probs) => {
            setVadLevel(probs.isSpeech)
          },
          onError: (e) => {
            setError(`VAD: ${e.message}`)
          },
        })
        vadRef.current = vad
        await vad.start()
      } catch (e) {
        setError(`VAD failed to load: ${e instanceof Error ? e.message : String(e)}`)
      }
      setState('listening')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setState('error')
    }
  }

  async function handleStop() {
    setState('closing')
    try { await vadRef.current?.stop() } catch { /* */ }
    vadRef.current = null
    try {
      await sessionRef.current?.stop()
    } finally {
      sessionRef.current = null
      setState('idle')
    }
  }

  // ── Event router from the WS
  function handleEvent(ev: OmniEvent) {
    // First post-flush event → TTFA dial
    if (ev.type && ev.type !== 'done' && flushAtMsRef.current > 0 && dial.ttfa_ms == null) {
      setDial((d) => ({ ...d, ttfa_ms: performance.now() - flushAtMsRef.current }))
    }
    if (ev.event === 'OmniError') {
      setError(ev.error || 'unknown adapter error')
      setState('error')
      return
    }
    if (ev.event === 'OmniReady') {
      pushCaption('meta', `✓ adapter ${ev.adapter} ready · wearer enrolled: ${ev.wearer_enrolled ? 'yes' : 'no (heartbeat off)'}`)
      setWearerEnrolled(!!ev.wearer_enrolled)
      return
    }
    if (ev.event === 'WearerTag') {
      setWearer({
        user_segments: ev.user_segments ?? 0,
        stranger_segments: ev.stranger_segments ?? 0,
        n_segments: ev.n_segments ?? 0,
        window_s: ev.window_s ?? 6,
        ts_ms: Date.now(),
      })
      return
    }
    if (ev.type === 'transcript') {
      pushCaption(ev.is_final ? 'final' : 'partial', ev.text || '')
      return
    }
    if (ev.type === 'text_delta') {
      // For now treat each text_delta as a complete response line —
      // refactor to streaming append once a real streaming adapter (Gemini Live) lands.
      pushCaption('response', ev.text || '')
      setState('replying')
      return
    }
    if (ev.type === 'audio_b64') {
      pushCaption('meta', `🔊 received audio chunk (${(ev.data || '').length} b64 bytes)`)
      return
    }
    if (ev.type === 'tool_call') {
      pushCaption('tool', `→ ${ev.name}(${JSON.stringify(ev.args ?? {})})`)
      return
    }
    if (ev.type === 'done') {
      const total = flushAtMsRef.current > 0 ? performance.now() - flushAtMsRef.current : null
      setDial((d) => ({
        ttfa_ms: d.ttfa_ms,
        total_ms: total,
        cost_usd: d.cost_usd + (ev.cost_usd ?? 0),
      }))
      flushAtMsRef.current = 0
      if (ev.error) {
        pushCaption('meta', `✗ ${ev.error}`)
        setState('error')
      } else {
        pushCaption('meta', `✓ done · server-side ${ev.latency_ms?.toFixed(0)} ms`)
        setState('listening')
      }
    }
  }

  // ── Camera start/stop. Toggling cameraOn while a session is live opens
  //    or closes the capture. If `cameraOn` is set BEFORE a session starts,
  //    the second effect (below) picks it up once `sessionRef.current` is
  //    populated. This split keeps the dependencies honest.
  useEffect(() => {
    const sess = sessionRef.current
    if (!cameraOn || !sess || !videoElRef.current) {
      // Tear down if camera was on and we're now off (or session ended)
      if (cameraRef.current) {
        cameraRef.current.stop()
        cameraRef.current = null
        setLastFrameUrl(null)
      }
      return
    }
    let cancelled = false
    let lastBlobUrl: string | null = null
    void (async () => {
      try {
        const cap = await startCameraCapture({
          session: sess,
          videoEl: videoElRef.current!,
          fps,
          onFrameSent: (jpeg, bytes, idx) => {
            setFramesSent(idx)
            setBytesSent((prev) => prev + bytes)
            // Build a thumbnail URL from the last frame; revoke the previous
            // one so we don't leak object URLs across the session.
            const blob = new Blob([new Uint8Array(jpeg)], { type: 'image/jpeg' })
            const url = URL.createObjectURL(blob)
            if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl)
            lastBlobUrl = url
            setLastFrameUrl(url)
          },
          onError: (e) => {
            setError(e.message)
            setCameraOn(false)
          },
        })
        if (cancelled) cap.stop()
        else cameraRef.current = cap
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e))
          setCameraOn(false)
        }
      }
    })()
    return () => {
      cancelled = true
      if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl)
    }
  }, [cameraOn, fps, state])

  // ── Cleanup on unmount
  useEffect(() => {
    return () => {
      try { vadRef.current?.stop() } catch { /* */ }
      try { cameraRef.current?.stop() } catch { /* */ }
      try { sessionRef.current?.abort() } catch { /* */ }
    }
  }, [])

  const adapterChoices = useMemo(() => adapters, [adapters])

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">Realtime</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Bidirectional <span className="font-mono text-xs bg-gray-100 px-1 rounded">realtime_omni</span> session.
          Hold the mic, speak, then release to flush; the adapter streams text + audio back.
          Fast loop · session not persisted to outbox.
        </p>
      </div>

      {/* Controls strip */}
      <div className="border-b border-gray-200 bg-white px-6 py-3 flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Adapter</label>
          <select
            value={adapterId}
            onChange={(e) => setAdapterId(e.target.value)}
            disabled={state !== 'idle'}
            className="field text-sm min-w-[260px]"
          >
            {adapterChoices.length === 0 && <option value="">(no realtime_omni adapters)</option>}
            {adapterChoices.map((a) => (
              <option key={a.id} value={a.id}>{a.display_name}</option>
            ))}
          </select>
        </div>

        {/* Mode radio — auto-turn always on once session starts; only the
            interrupt behavior differs between the two options. Disabled
            mid-session so flipping doesn't half-cancel an in-flight reply. */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Conversation mode</label>
          <div className="flex gap-2 text-xs">
            {(['no-interrupt', 'interruptable'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                disabled={state !== 'idle' && state !== 'error'}
                className={cx(
                  'px-3 py-1.5 rounded-full border transition-colors',
                  mode === m
                    ? 'bg-gray-900 text-white border-gray-900'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100',
                  state !== 'idle' && state !== 'error' && 'opacity-60 cursor-not-allowed',
                )}
                title={
                  m === 'no-interrupt'
                    ? 'Model finishes its reply even if you start talking again'
                    : 'Speaking mid-reply cancels playback + drops further chunks'
                }
              >
                {m === 'no-interrupt' ? 'auto-turn (no-interrupt)' : 'auto-turn (interruptable)'}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3 ml-auto">
          {state === 'idle' || state === 'error' ? (
            <button
              type="button"
              onClick={() => void handleStart()}
              disabled={!adapterId}
              className="btn-pill-dark text-sm px-5 py-2"
            >
              ▶ Start session
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void handleStop()}
              className="btn-pill-outline text-sm px-4 py-2"
            >
              ⏹ Stop
            </button>
          )}
        </div>
      </div>

      {/* State + latency dial */}
      <div className="border-b border-gray-200 bg-white px-6 py-2 flex items-center flex-wrap gap-x-6 gap-y-1 text-xs font-mono text-gray-600">
        <span>
          state: <span className={cx(
            'font-semibold',
            state === 'listening' && 'text-green-700',
            state === 'awaiting' && 'text-amber-700',
            state === 'replying' && 'text-amber-700',
            state === 'error' && 'text-red-700',
          )}>{state}</span>
        </span>
        <span>TTFA: {dial.ttfa_ms != null ? `${dial.ttfa_ms.toFixed(0)} ms` : '—'}</span>
        <span>total: {dial.total_ms != null ? `${dial.total_ms.toFixed(0)} ms` : '—'}</span>
        <span>cost: ${dial.cost_usd.toFixed(4)}</span>
        {cameraOn && (
          <span>frames: {framesSent} · {(bytesSent / 1024).toFixed(0)} KB</span>
        )}
        {/* Wearer-tag overlay (heartbeat result; Slice O5) */}
        {wearerEnrolled === false && (
          <span className="inline-flex items-center gap-1 text-amber-700">
            ⚠ no wearer enrolled
          </span>
        )}
        {wearer && (
          <span className="inline-flex items-center gap-1">
            wearer-tag (last {wearer.window_s.toFixed(0)}s):
            <span className={cx(
              'inline-flex items-center px-1.5 py-0.5 rounded text-[10px]',
              wearer.user_segments > wearer.stranger_segments
                ? 'bg-green-100 text-green-800'
                : wearer.stranger_segments > 0
                  ? 'bg-gray-200 text-gray-700'
                  : 'bg-gray-100 text-gray-500',
            )}>
              {wearer.user_segments > wearer.stranger_segments
                ? `wearer (${wearer.user_segments}/${wearer.n_segments})`
                : wearer.stranger_segments > 0
                  ? `stranger (${wearer.stranger_segments}/${wearer.n_segments})`
                  : 'silent'}
            </span>
          </span>
        )}
        {/* TTFA-degraded badge — current MiniCPM-o adapter is HTTP-fallback, not WS */}
        {adapterId === 'minicpm_o' && (
          <span className="ml-auto inline-flex items-center px-2 py-0.5 rounded-full text-[10px] bg-amber-50 text-amber-800 border border-amber-200">
            ⚠ TTFA degraded · chunked HTTP fallback (v1)
          </span>
        )}
      </div>

      {error && (
        <div className="mx-6 mt-4 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Body — 3-column on lg, stacked on smaller screens */}
      <div className="flex-1 overflow-auto p-6 grid grid-cols-1 lg:grid-cols-[260px_1fr_180px] gap-4">
        {/* ── Left: camera column ─────────────────────────────────────── */}
        <div className="card flex flex-col gap-3">
          <p className="text-xs text-gray-500 uppercase tracking-wider">Camera</p>
          <div className="aspect-[4/3] bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
            {/* Self-view always rendered so the cameraStream module has a
                stable target; visibility is toggled via camera state. */}
            <video
              ref={videoElRef}
              className={cx('w-full h-full object-cover', !cameraOn && 'hidden')}
              autoPlay
              muted
              playsInline
            />
            {!cameraOn && (
              <span className="text-xs text-gray-400 italic">camera off</span>
            )}
          </div>
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={cameraOn}
              onChange={(e) => {
                if (state === 'idle' && e.target.checked) {
                  setError('Start a session first, then enable the camera.')
                  return
                }
                setCameraOn(e.target.checked)
                setFramesSent(0)
                setBytesSent(0)
              }}
              disabled={state === 'idle' || state === 'opening'}
            />
            Enable camera
          </label>
          <div className="flex items-center gap-2 text-xs">
            <label className="text-gray-500">fps</label>
            <select
              value={fps}
              onChange={(e) => setFps(Number(e.target.value))}
              className="field text-xs flex-1"
              disabled={!cameraOn}
            >
              <option value={1}>1</option>
              <option value={2}>2 (default)</option>
              <option value={3}>3</option>
              <option value={5}>5 (max)</option>
            </select>
          </div>
          <p className="text-[10px] text-gray-400 leading-relaxed">
            Webcam frames are JPEG-encoded at q=0.7 and sent alongside mic audio.
            The adapter holds the most-recent frame and sends it with each utterance.
          </p>
        </div>

        {/* ── Center: caption strip ───────────────────────────────────── */}
        <div className="card">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">Caption stream</p>
          {captions.length === 0 ? (
            <p className="text-sm text-gray-400 italic">
              No events yet. Click <span className="font-mono">Start session</span>,
              speak, then click <span className="font-mono">Flush utterance</span>.
            </p>
          ) : (
            <ul className="space-y-2">
              {captions.map((c) => (
                <li key={c.id} className="flex items-start gap-2 text-sm">
                  <span className="text-[10px] font-mono text-gray-400 mt-1 w-12 shrink-0">
                    {new Date(c.ts_ms).toLocaleTimeString().split(' ')[0]}
                  </span>
                  <span className={cx(
                    'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono mt-0.5 shrink-0 w-16 justify-center',
                    c.kind === 'partial' && 'bg-gray-100 text-gray-500',
                    c.kind === 'final' && 'bg-blue-100 text-blue-800',
                    c.kind === 'response' && 'bg-pink-100 text-pink-800',
                    c.kind === 'tool' && 'bg-emerald-100 text-emerald-800',
                    c.kind === 'meta' && 'bg-gray-100 text-gray-500',
                  )}>
                    {c.kind}
                  </span>
                  <span className={cx(
                    'flex-1 break-words',
                    c.kind === 'partial' && 'text-gray-500 italic',
                    c.kind === 'response' && 'text-gray-900 font-medium',
                  )}>
                    {c.text}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* ── Right: last-frame thumbnail strip ───────────────────────── */}
        <div className="card flex flex-col gap-2">
          <p className="text-xs text-gray-500 uppercase tracking-wider">Last frame sent</p>
          {lastFrameUrl ? (
            <>
              <img
                src={lastFrameUrl}
                alt="last frame sent to adapter"
                className="w-full rounded border border-gray-200"
              />
              <p className="text-[10px] text-gray-400 font-mono break-all">
                {framesSent} frame{framesSent === 1 ? '' : 's'} · last ~{lastFrameUrl ? '?' : '0'}
              </p>
            </>
          ) : (
            <p className="text-xs text-gray-400 italic">
              No frames sent yet. Enable camera + start a session.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
