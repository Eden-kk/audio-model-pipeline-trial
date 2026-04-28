import { useState, useRef, useCallback } from 'react'
import { cx } from '../lib/cx'

interface Props {
  onBlob: (blob: Blob, mime: string) => void
  disabled?: boolean
}

// Accept audio AND video container files. Video uploads have their audio
// track extracted server-side via ffmpeg in /api/clips, so the canonical
// audio that adapters see is always 16 kHz mono PCM regardless of source.
const ACCEPTED = [
  'audio/wav', 'audio/wave', 'audio/mpeg', 'audio/mp3', 'audio/ogg',
  'audio/webm', 'audio/flac', 'audio/x-flac', 'audio/mp4',
  'video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska',
  'video/x-msvideo', 'video/mpeg',
]
const VIDEO_EXT_RE = /\.(mp4|mov|webm|mkv|avi|m4v|ts|mts)$/i
const AUDIO_EXT_RE = /\.(wav|mp3|ogg|webm|flac|m4a|opus|aac)$/i

function fileToBlob(file: File): Promise<{ blob: Blob; mime: string }> {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const buf = e.target!.result as ArrayBuffer
      const mime = file.type || 'audio/wav'
      resolve({ blob: new Blob([buf], { type: mime }), mime })
    }
    reader.readAsArrayBuffer(file)
  })
}

export default function AudioFileDrop({ onBlob, disabled = false }: Props) {
  const [dragging, setDragging] = useState(false)
  const [fileName, setFileName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    async (file: File) => {
      setError(null)
      const matches =
        ACCEPTED.some((t) => file.type.startsWith(t.split(';')[0])) ||
        AUDIO_EXT_RE.test(file.name) ||
        VIDEO_EXT_RE.test(file.name)
      if (!matches) {
        setError(`Unsupported file type: ${file.type || file.name}`)
        return
      }
      setFileName(file.name)
      const { blob, mime } = await fileToBlob(file)
      onBlob(blob, mime)
    },
    [onBlob],
  )

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    const file = e.dataTransfer.files[0]
    if (file) void handleFile(file)
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) void handleFile(file)
    e.target.value = ''
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={cx(
        'flex flex-col items-center justify-center gap-2 w-full border-2 border-dashed rounded-xl p-6 cursor-pointer transition-colors select-none',
        dragging ? 'border-indigo-400 bg-indigo-950/30' : 'border-gray-700 hover:border-gray-500',
        disabled && 'opacity-50 cursor-not-allowed',
      )}
    >
      <svg className="w-8 h-8 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" x2="12" y1="3" y2="15" />
      </svg>
      {fileName ? (
        <span className="text-sm text-indigo-300">{fileName}</span>
      ) : (
        <span className="text-sm text-gray-400">Drag & drop an audio or video file, or click to browse</span>
      )}
      <span className="text-xs text-gray-600">Audio: WAV · MP3 · OGG · FLAC · M4A — Video: MP4 · MOV · WebM · MKV (audio track auto-extracted)</span>
      {error && <p className="text-red-400 text-xs">{error}</p>}
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.m4a,.opus,.aac,.flac"
        className="hidden"
        onChange={onInputChange}
        disabled={disabled}
      />
    </div>
  )
}
