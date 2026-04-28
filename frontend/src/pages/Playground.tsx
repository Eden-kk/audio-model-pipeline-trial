import { useState, useEffect, useCallback } from 'react'
import MicRecorder from '../components/MicRecorder'
import AudioFileDrop from '../components/AudioFileDrop'
import {
  listAdapters,
  uploadClip,
  startRun,
  connectRunWS,
  type Adapter,
  type RunEvent,
} from '../lib/api'
import { cx } from '../lib/cx'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StageResult {
  stage_id?: string
  transcript?: string
  latency_ms?: number
  raw_response?: unknown
  error?: string
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
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [adaptersError, setAdaptersError] = useState<string | null>(null)
  const [selectedAdapter, setSelectedAdapter] = useState<string>('')
  const [pendingBlob, setPendingBlob] = useState<{ blob: Blob; mime: string } | null>(null)
  const [runState, setRunState] = useState<RunState>('idle')
  const [statusMsg, setStatusMsg] = useState<string>('')
  const [events, setEvents] = useState<RunEvent[]>([])
  const [results, setResults] = useState<StageResult[]>([])
  const [runError, setRunError] = useState<string | null>(null)
  const [inputMode, setInputMode] = useState<'record' | 'upload'>('upload')

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

  const handleBlob = useCallback((blob: Blob, mime: string) => {
    setPendingBlob({ blob, mime })
    setRunState('idle')
    setResults([])
    setRunError(null)
    setEvents([])
  }, [])

  async function handleRun() {
    if (!pendingBlob) return
    if (!selectedAdapter) return

    setRunState('uploading')
    setStatusMsg('Uploading clip…')
    setResults([])
    setRunError(null)
    setEvents([])

    let clip
    try {
      clip = await uploadClip(pendingBlob.blob, pendingBlob.mime)
      setStatusMsg('Starting run…')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setRunError(`Upload failed: ${msg}`)
      setRunState('error')
      return
    }

    let run
    try {
      run = await startRun(clip.id, selectedAdapter)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setRunError(`Run failed: ${msg}`)
      setRunState('error')
      return
    }

    setRunState('running')
    setStatusMsg('Running…')

    const collected: StageResult[] = []
    const cleanup = connectRunWS(
      run.id,
      (ev) => {
        setEvents((prev) => [...prev, ev])

        if (ev.type === 'stage.finished') {
          const r: StageResult = {
            stage_id: ev.stage_id,
            transcript: (ev.data.transcript as string | undefined) ?? (ev.data.output_preview as string | undefined),
            latency_ms: ev.data.latency_ms as number | undefined,
            raw_response: ev.data.raw_response,
          }
          collected.push(r)
          setResults([...collected])
        }

        if (ev.type === 'run.finished') {
          setRunState('done')
          setStatusMsg('Done')
          cleanup()
        }

        if (ev.type === 'run.error') {
          setRunError((ev.data.error as string | undefined) ?? 'Unknown error')
          setRunState('error')
          cleanup()
        }
      },
      () => {
        if (collected.length > 0) {
          setRunState('done')
          setStatusMsg('Done')
        }
      },
      () => {
        setRunError('WebSocket connection failed.')
        setRunState('error')
      },
    )
  }

  const busy = runState === 'uploading' || runState === 'running'
  const hasAudio = pendingBlob !== null
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
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200 font-mono">
                  {selectedMeta.id}
                </span>
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

            <div className="flex gap-2">
              {(['upload', 'record'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  disabled={busy}
                  onClick={() => setInputMode(mode)}
                  className={cx(
                    inputMode === mode ? 'btn-pill-dark' : 'btn-pill-outline',
                    'text-xs',
                    busy && 'opacity-50 cursor-not-allowed',
                  )}
                >
                  {mode === 'upload' ? 'File upload' : 'Mic record'}
                </button>
              ))}
            </div>

            {inputMode === 'upload' ? (
              <AudioFileDrop onBlob={handleBlob} disabled={busy} />
            ) : (
              <MicRecorder onBlob={handleBlob} disabled={busy} />
            )}

            {hasAudio && (
              <p className="text-xs text-green-700">
                Audio ready — {(pendingBlob!.blob.size / 1024).toFixed(1)} KB ({pendingBlob!.mime})
              </p>
            )}
          </div>
        </div>

        {/* Run button */}
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

        {/* ── Results section ── */}
        {runError && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            {runError}
          </div>
        )}

        {(results.length > 0 || runState === 'done') && (
          <div className="card overflow-hidden p-0">
            <div className="px-5 py-3 border-b border-gray-200 flex items-center gap-3 bg-gray-50">
              <span className="text-sm font-semibold text-gray-700">Transcript</span>
              {primaryResult?.latency_ms != null && (
                <LatencyBadge ms={primaryResult.latency_ms} />
              )}
            </div>

            <div className="px-5 py-4">
              {primaryResult?.transcript ? (
                <p className="text-gray-900 leading-relaxed whitespace-pre-wrap">{primaryResult.transcript}</p>
              ) : (
                <p className="text-gray-500 italic text-sm">
                  {runState === 'done' ? 'No transcript returned.' : 'Waiting for transcript…'}
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
                  <span className="text-gray-900 font-semibold">{ev.type}</span>
                  {ev.stage_id && <span className="text-gray-500 ml-1">({ev.stage_id})</span>}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}
