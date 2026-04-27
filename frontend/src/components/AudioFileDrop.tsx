import { useState, useRef, useCallback } from 'react'
import { cx } from '../lib/cx'

interface Props {
  onBlob: (blob: Blob, mime: string) => void
  disabled?: boolean
}

const ACCEPTED = ['audio/wav', 'audio/wave', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm', 'audio/flac', 'audio/x-flac', 'audio/mp4']

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
      if (!ACCEPTED.some((t) => file.type.startsWith(t.split(';')[0])) && !file.name.match(/\.(wav|mp3|ogg|webm|flac|m4a|opus)$/i)) {
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
        <span className="text-sm text-gray-400">Drag & drop an audio file, or click to browse</span>
      )}
      <span className="text-xs text-gray-600">WAV · MP3 · OGG · WebM · FLAC</span>
      {error && <p className="text-red-400 text-xs">{error}</p>}
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={onInputChange}
        disabled={disabled}
      />
    </div>
  )
}
