import { useRef, useState } from 'react'
import { startMicStream, type MicStreamEvent, type MicStreamHandle } from '../lib/micStream'
import { cx } from '../lib/cx'

interface Props {
  adapter: string
  onEvent: (ev: MicStreamEvent) => void
  /** Fires the moment the user clicks Stop (locally). Lets the parent
   *  drop out of its "running" state immediately so the start button
   *  is re-enabled — without waiting for the backend's StageCompleted
   *  to arrive (which can lag 200-500 ms). */
  onLocalStop?: () => void
  disabled?: boolean
  /** BCP-47 language code or 'auto'. Locked at stream start. */
  language?: string
}

/** Live-mic streaming button. Records via AudioWorklet and streams 16 kHz
 *  PCM through /ws/mic to the chosen vendor's streaming WS. Tokens come
 *  back as StageProgress events. Click again to stop.
 *
 *  Plan D A6 — adds a "Save to corpus" checkbox. When ticked, the backend
 *  persists the captured audio + streaming transcript as a live-mic corpus
 *  clip with the ar-glass-capture / live-mic / vendor-X scenarios stamped
 *  in (so the user can build the AR-glass benchmark by speaking into the
 *  mic). The clip_id arrives in a `ClipSaved` event and gets surfaced
 *  inline so the user can jump to the new corpus row. */
export default function MicStream({ adapter, onEvent, onLocalStop, disabled = false, language }: Props) {
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [seconds, setSeconds] = useState(0)
  const [saveToCorpus, setSaveToCorpus] = useState(false)
  const [savedClipId, setSavedClipId] = useState<string | null>(null)
  const handleRef = useRef<MicStreamHandle | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  async function start() {
    setError(null)
    setSeconds(0)
    setSavedClipId(null)
    try {
      const handle = await startMicStream({
        adapter,
        save: saveToCorpus,
        language,
        onEvent: (ev) => {
          onEvent(ev)
          if (ev.event === 'ClipSaved' && ev.clip_id) {
            setSavedClipId(ev.clip_id)
          }
          if (ev.event === 'StageCompleted' || ev.event === 'StageFailed') {
            setStreaming(false)
            if (timerRef.current) clearInterval(timerRef.current)
            handleRef.current = null
          }
        },
        onError: (err) => setError(err.message),
      })
      handleRef.current = handle
      setStreaming(true)
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function stop() {
    const h = handleRef.current
    if (h) {
      await h.stop()
    }
    if (timerRef.current) clearInterval(timerRef.current)
    // Flip UI back to "Stream from mic" right away — don't wait for the
    // backend's StageCompleted to arrive. The vendor still has 100-500 ms
    // of in-flight audio to flush, but from the user's perspective they
    // hit Stop and the button should immediately reflect that.
    handleRef.current = null
    setStreaming(false)
    setSeconds(0)
    onLocalStop?.()
  }

  const label = streaming ? `Stop (${seconds}s)` : 'Stream from mic'

  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        // Once streaming has started, the button doubles as Stop and MUST
        // remain clickable even when the parent flips a "busy" flag (which
        // happens the moment StageStarted fires and the parent sets
        // runState='running'). Only respect `disabled` when the user is
        // about to *start* a fresh stream.
        disabled={disabled && !streaming}
        onClick={streaming ? stop : start}
        className={cx(
          'btn-pill',
          streaming
            ? 'bg-red-600 hover:bg-red-700 text-white'
            : 'btn-pill-dark',
        )}
      >
        {streaming ? (
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" x2="12" y1="19" y2="22" />
          </svg>
        )}
        {label}
        {streaming && (
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-white animate-pulse ml-2" />
        )}
      </button>

      <label className="flex items-center gap-1.5 text-xs text-gray-700 select-none">
        <input
          type="checkbox"
          className="h-3.5 w-3.5 rounded border-gray-300"
          checked={saveToCorpus}
          disabled={streaming}
          onChange={(e) => setSaveToCorpus(e.target.checked)}
        />
        Save to corpus
        <span className="text-gray-400">
          (tags: <code className="text-[10px]">ar-glass-capture</code>)
        </span>
      </label>

      <p className="text-xs text-gray-600">
        Live-streams 16 kHz PCM directly to the model. Transcript appears as you speak.
      </p>
      {savedClipId && !streaming && (
        <p className="text-xs text-green-700">
          Saved to corpus —{' '}
          <a
            href={`/corpus?focus=${savedClipId}`}
            className="underline hover:text-green-900"
          >
            view clip {savedClipId.slice(0, 8)}…
          </a>
        </p>
      )}
      {error && <p className="text-red-600 text-xs">{error}</p>}
    </div>
  )
}
