// Live-mic streaming controller.
//
// Manages: getUserMedia → AudioContext @ 16 kHz → AudioWorkletNode → WS.
// Worklet (frontend/public/mic-pcm-worklet.js) emits Int16Array buffers;
// we forward them as binary WS frames to /ws/mic?adapter=…
//
// Server emits StageStarted / StageProgress / StageCompleted / StageFailed
// (same shape as POST /api/runs streaming runs).

const TARGET_SR = 16000   // matches what /ws/mic expects

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export interface MicStreamEvent {
  event: 'StageStarted' | 'StageProgress' | 'StageCompleted' | 'StageFailed'
  partial_text?: string
  partial_index?: number
  result?: { text?: string; words?: unknown[]; language?: string; wall_time_s?: number }
  latency_ms?: number
  error?: string
  adapter?: string
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
}

export async function startMicStream({
  adapter,
  onEvent,
  onError,
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
  const wsUrl = `${wsBase}/ws/mic?adapter=${encodeURIComponent(adapter)}&sample_rate=${audioCtx.sampleRate}`
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
