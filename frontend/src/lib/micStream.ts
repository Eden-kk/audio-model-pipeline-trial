// Live-mic streaming controller.
//
// Manages: getUserMedia → AudioContext @ 16 kHz → AudioWorkletNode → WS.
// Worklet (frontend/public/mic-pcm-worklet.js) emits Int16Array buffers;
// we forward them as binary WS frames to /ws/mic?adapter=…
//
// Server emits StageStarted / StageProgress / StageCompleted / StageFailed
// (same shape as POST /api/runs streaming runs).

const TARGET_SR = 16000   // matches what /ws/mic expects

const BASE = (import.meta.env.VITE_API_URL as string | undefined)
  ?? (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin)

export interface MicStreamEvent {
  event:
    | 'StageStarted'
    | 'StageProgress'
    | 'StageCompleted'
    | 'StageFailed'
    /** Plan D A6 — emitted right after StageCompleted when the client
     *  opted in via {save: true}; carries the new corpus clip_id so the
     *  UI can deep-link to the saved row. */
    | 'ClipSaved'
  partial_text?: string
  partial_index?: number
  result?: { text?: string; words?: unknown[]; language?: string; wall_time_s?: number }
  latency_ms?: number
  error?: string
  adapter?: string
  /** Set on `ClipSaved` events. */
  clip_id?: string
}

export interface MicStreamHandle {
  /** Tell the server we're done speaking; the vendor will flush and emit
   *  the final transcript via StageCompleted. After that the WS closes. */
  stop: () => Promise<void>
  /** Force-close everything immediately (mic + WS). */
  abort: () => void
}

interface StartOptions {
  adapter: string
  onEvent: (ev: MicStreamEvent) => void
  onError?: (err: Error) => void
  /** Plan D A6 — when true, the backend persists the captured PCM
   *  + streaming transcript as a corpus clip on stop. Default false
   *  preserves the legacy ephemeral live-stream UX. */
  save?: boolean
}

export async function startMicStream({
  adapter,
  onEvent,
  onError,
  save = false,
}: StartOptions): Promise<MicStreamHandle> {
  // 1) Get the mic
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
        sampleRate: TARGET_SR,
      } as MediaTrackConstraints,
    })
  } catch (err) {
    const e = err instanceof Error ? err : new Error(String(err))
    onError?.(new Error(`Microphone permission denied: ${e.message}`))
    throw e
  }

  // 2) Create an AudioContext at our target rate. Chrome/Safari respect
  //    the sampleRate hint; Firefox may fall back to 48k — the worklet
  //    still works because we read whatever the browser gives us.
  const audioCtx = new (window.AudioContext || (window as unknown as {
    webkitAudioContext: typeof AudioContext
  }).webkitAudioContext)({ sampleRate: TARGET_SR })
  await audioCtx.audioWorklet.addModule('/mic-pcm-worklet.js')

  const source = audioCtx.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(audioCtx, 'pcm-worklet', {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    channelCount: 1,
  })

  // 3) Open the WS to backend, wait for it to open
  const wsBase = BASE.replace(/^http/, 'ws')
  const wsUrl = `${wsBase}/ws/mic?adapter=${encodeURIComponent(adapter)}&sample_rate=${audioCtx.sampleRate}${save ? '&save=1' : ''}`
  const ws = new WebSocket(wsUrl)
  ws.binaryType = 'arraybuffer'

  await new Promise<void>((resolve, reject) => {
    const onOpen = () => { ws.removeEventListener('error', onErr); resolve() }
    const onErr = (e: Event) => { ws.removeEventListener('open', onOpen); reject(new Error(`WS open failed: ${String(e)}`)) }
    ws.addEventListener('open', onOpen, { once: true })
    ws.addEventListener('error', onErr, { once: true })
  })

  // 4) Pipe Int16 buffers from the worklet → WS as binary frames
  node.port.onmessage = (e) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(e.data as ArrayBuffer)
    }
  }
  source.connect(node)

  // 5) Pipe events from WS → onEvent
  ws.addEventListener('message', (msg) => {
    if (typeof msg.data !== 'string') return
    try {
      const ev = JSON.parse(msg.data) as MicStreamEvent
      onEvent(ev)
    } catch {
      // ignore malformed
    }
  })
  ws.addEventListener('error', (e) => {
    onError?.(new Error(`WS error: ${String(e)}`))
  })

  // 6) Cleanup helpers
  let cleaned = false
  const cleanup = () => {
    if (cleaned) return
    cleaned = true
    try { node.port.onmessage = null } catch { /* */ }
    try { node.disconnect() } catch { /* */ }
    try { source.disconnect() } catch { /* */ }
    stream.getTracks().forEach((t) => t.stop())
    void audioCtx.close().catch(() => undefined)
  }

  ws.addEventListener('close', cleanup)

  return {
    stop: async () => {
      // Tell the server to flush vendor → emit final result
      try {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'stop' }))
        }
      } catch { /* */ }
      cleanup()
    },
    abort: () => {
      try { ws.close(1000, 'aborted') } catch { /* */ }
      cleanup()
    },
  }
}
