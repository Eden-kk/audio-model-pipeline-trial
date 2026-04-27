import { useState, useRef, useEffect } from 'react'
import { cx } from '../lib/cx'

interface Props {
  onBlob: (blob: Blob, mime: string) => void
  disabled?: boolean
}

/** Preferred MIME types in order. Falls back to first supported. */
function pickMime(): string {
  const candidates = [
    'audio/wav',
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
  ]
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c
  }
  return ''
}

export default function MicRecorder({ onBlob, disabled = false }: Props) {
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [seconds, setSeconds] = useState(0)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Clean up on unmount
  useEffect(() => {
    return () => {
      recorderRef.current?.stop()
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  async function startRecording() {
    setError(null)
    chunksRef.current = []
    setSeconds(0)

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Microphone permission denied.')
      return
    }

    const mime = pickMime()
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    recorderRef.current = recorder

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      const usedMime = recorder.mimeType || mime || 'audio/webm'
      const blob = new Blob(chunksRef.current, { type: usedMime })
      onBlob(blob, usedMime)
      if (timerRef.current) clearInterval(timerRef.current)
      setRecording(false)
    }

    recorder.start(100)
    setRecording(true)
    timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000)
  }

  function stopRecording() {
    recorderRef.current?.stop()
  }

  const label = recording ? `Stop (${seconds}s)` : 'Record'

  return (
    <div className="flex flex-col items-start gap-1">
      <button
        type="button"
        disabled={disabled}
        onClick={recording ? stopRecording : startRecording}
        className={cx(
          'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
          recording
            ? 'bg-red-600 hover:bg-red-700 text-white'
            : 'bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50 disabled:cursor-not-allowed',
        )}
      >
        {/* mic/stop icon */}
        {recording ? (
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
      </button>
      {error && <p className="text-red-400 text-xs">{error}</p>}
    </div>
  )
}
