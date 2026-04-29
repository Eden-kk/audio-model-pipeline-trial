// Bidirectional realtime-audio WS controller for the /realtime page.
//
// Differs from micStream.ts in three ways:
//   1. The outbound binary protocol prepends a 1-byte type tag to every
//      frame (0x01 audio PCM · 0x02 JPEG video · 0x03 flush) so the same
//      WS multiplexes mic + camera (camera lands in Slice O3).
//   2. We also PLAY back audio chunks the server sends in `audio_b64`
//      omni events — via a WebAudio AudioBufferSourceNode queue.
//   3. We expose push-to-talk semantics (`flush()` to mark end-of-utterance)
//      rather than just a `stop()` that ends the whole session.
//
// micStream.ts stays untouched for back-compat with /playground's live-mic
// transcription pages — different protocol, different server endpoint.

const TARGET_SR = 16000
const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

// Outbound binary frame discriminators — must match
// backend/realtime/omni_proxy.py FRAME_AUDIO/FRAME_VIDEO/FRAME_FLUSH.
const FRAME_AUDIO = 0x01
const FRAME_VIDEO = 0x02
const FRAME_FLUSH = 0x03

/** Server → client realtime omni event (matches backend adapter contract). */
export interface OmniEvent {
  /** Reserved control / lifecycle events. */
  event?: 'OmniReady' | 'OmniError' | 'pong' | 'WearerTag'
  /** Adapter omni events: */
  type?: 'transcript' | 'text_delta' | 'audio_b64' | 'tool_call' | 'done'
  text?: string
  is_final?: boolean
  data?: string             // base64 audio payload on `audio_b64`
  sample_rate?: number
  name?: string             // tool_call.name
  args?: Record<string, unknown>
  latency_ms?: number
  cost_usd?: number
  error?: string
  server_ts_ms?: number
  adapter?: string
  ts_ms?: number
  /** WearerTag fields (Slice O5 heartbeat) */
  user_segments?: number
  stranger_segments?: number
  n_segments?: number
  window_s?: number
  /** OmniReady extras */
  wearer_enrolled?: boolean
}

export interface OmniSessionHandle {
  /** Mark the end of the current utterance. The adapter will then dispatch
   *  to its model and stream a response back. Returns immediately — wait
   *  for the `done` event in `onEvent` to know when the response landed. */
  flush: () => void
  /** End the entire session: stops mic + closes WS + tears down playback. */
  stop: () => Promise<void>
  /** Force-close everything immediately. */
  abort: () => void
  /** Indicates whether the server has accepted the session
   *  (OmniReady received). */
  ready: () => boolean
  /** Send one tagged binary frame on the existing WS. Used by
   *  `cameraStream.ts` to push 0x02 JPEG frames alongside the mic audio.
   *  Returns false if the WS isn't open (caller can drop or buffer). */
  sendBinary: (tag: number, payload: Uint8Array) => boolean
  /** Slice O6.3: cancel all queued + actively-playing audio chunks
   *  immediately. Used by interrupt mode when the user starts speaking
   *  while the model is mid-reply. Returns the count of cancelled
   *  sources for telemetry. */
  cancelPlayback: () => number
  /** Send a JSON control event (e.g. {event:'interrupt'} | {event:'flush'}).
   *  The WS proxy interprets these on the server side. */
  sendControl: (payload: Record<string, unknown>) => boolean
}

interface StartOptions {
  adapter: string
  profileId?: string
  onEvent: (ev: OmniEvent) => void
  onError?: (err: Error) => void
}

/** Open mic, open WS to /ws/omni, wire audio playback queue.
 *  Camera capture lands in Slice O3 — a separate `useWebcamFrames` hook
 *  can call `sendVideoFrame()` on the returned handle once that exists.
 */
export async function startOmniSession({
  adapter,
  profileId = 'wearer',
  onEvent,
  onError,
}: StartOptions): Promise<OmniSessionHandle> {
  // ── 1) Mic
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

  // ── 2) AudioContext + worklet (same module as micStream — emits Int16Array)
  const audioCtx = new (window.AudioContext || (window as unknown as {
    webkitAudioContext: typeof AudioContext
  }).webkitAudioContext)({ sampleRate: TARGET_SR })
  await audioCtx.audioWorklet.addModule('/mic-pcm-worklet.js')

  const source = audioCtx.createMediaStreamSource(stream)
  const node = new AudioWorkletNode(audioCtx, 'pcm-worklet', {
    numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
  })

  // ── 3) WS open
  const wsBase = BASE.replace(/^http/, 'ws')
  const wsUrl = `${wsBase}/ws/omni?adapter=${encodeURIComponent(adapter)}&profile_id=${encodeURIComponent(profileId)}`
  const ws = new WebSocket(wsUrl)
  ws.binaryType = 'arraybuffer'
  await new Promise<void>((resolve, reject) => {
    const onOpen = () => { ws.removeEventListener('error', onErr); resolve() }
    const onErr = (e: Event) => { ws.removeEventListener('open', onOpen); reject(new Error(`WS open failed: ${String(e)}`)) }
    ws.addEventListener('open', onOpen, { once: true })
    ws.addEventListener('error', onErr, { once: true })
  })

  // ── 4) Playback context + queue. Use a dedicated AudioContext (not
  //    audioCtx) so playback isn't constrained to the mic's 16 kHz. The
  //    server sends sample_rate per chunk; we resample via a temporary
  //    OfflineAudioContext if it differs from the playback ctx.
  const playCtx = new (window.AudioContext || (window as unknown as {
    webkitAudioContext: typeof AudioContext
  }).webkitAudioContext)()
  let nextStartTime = 0   // schedule chunks back-to-back so playback is gapless

  // Slice O6.3: track every active AudioBufferSourceNode so cancelPlayback()
  // can stop every chunk that's currently playing OR scheduled to play.
  // Cleared on each natural `ended` event; cleared en masse on cancel.
  const activeSources = new Set<AudioBufferSourceNode>()

  const playPcmBytes = async (pcmB64: string, sampleRate: number): Promise<void> => {
    try {
      const binStr = atob(pcmB64)
      const buf = new Uint8Array(binStr.length)
      for (let i = 0; i < binStr.length; i++) buf[i] = binStr.charCodeAt(i)
      let audioBuf: AudioBuffer
      try {
        // decodeAudioData mutates its input on some platforms; copy first.
        audioBuf = await playCtx.decodeAudioData(buf.buffer.slice(0))
      } catch {
        // Fallback: treat as raw Int16 PCM at the declared sample rate.
        const samples = new Int16Array(buf.buffer)
        const f32 = new Float32Array(samples.length)
        for (let i = 0; i < samples.length; i++) f32[i] = samples[i] / 32768
        audioBuf = playCtx.createBuffer(1, f32.length, sampleRate || TARGET_SR)
        audioBuf.copyToChannel(f32, 0)
      }
      const src = playCtx.createBufferSource()
      src.buffer = audioBuf
      src.connect(playCtx.destination)
      activeSources.add(src)
      src.addEventListener('ended', () => {
        activeSources.delete(src)
      })
      const startAt = Math.max(playCtx.currentTime, nextStartTime)
      src.start(startAt)
      nextStartTime = startAt + audioBuf.duration
    } catch (e) {
      onError?.(new Error(`audio playback failed: ${e instanceof Error ? e.message : String(e)}`))
    }
  }

  /** Stop all queued + actively-playing audio chunks. Returns count
   *  cancelled. Resets nextStartTime so subsequent chunks play immediately
   *  rather than chaining onto whatever was scheduled before. */
  const cancelPlayback = (): number => {
    let n = 0
    for (const src of activeSources) {
      try { src.stop(0) } catch { /* node already stopped */ }
      n++
    }
    activeSources.clear()
    nextStartTime = playCtx.currentTime
    return n
  }

  // ── 5) Mic worklet → tagged-binary WS frames
  // The worklet emits raw ArrayBuffers of Int16 PCM. We prepend a 0x01
  // type tag and forward. Allocate once per send to keep GC pressure low.
  node.port.onmessage = (e) => {
    if (ws.readyState !== WebSocket.OPEN) return
    const pcm = new Uint8Array(e.data as ArrayBuffer)
    const out = new Uint8Array(1 + pcm.byteLength)
    out[0] = FRAME_AUDIO
    out.set(pcm, 1)
    ws.send(out.buffer)
  }
  source.connect(node)

  // ── 6) Inbound omni events
  let ready = false
  ws.addEventListener('message', (msg) => {
    if (typeof msg.data !== 'string') return
    let parsed: OmniEvent
    try {
      parsed = JSON.parse(msg.data) as OmniEvent
    } catch {
      return
    }
    if (parsed.event === 'OmniReady') ready = true
    if (parsed.type === 'audio_b64' && parsed.data) {
      // Fire-and-forget — playback is async but events stay ordered
      // because we await `nextStartTime` inside playPcmBytes.
      void playPcmBytes(parsed.data, parsed.sample_rate ?? TARGET_SR)
    }
    onEvent(parsed)
  })
  ws.addEventListener('error', (e) => {
    onError?.(new Error(`WS error: ${String(e)}`))
  })

  // ── 7) Cleanup
  let cleaned = false
  const cleanup = () => {
    if (cleaned) return
    cleaned = true
    try { node.port.onmessage = null } catch { /* */ }
    try { node.disconnect() } catch { /* */ }
    try { source.disconnect() } catch { /* */ }
    stream.getTracks().forEach((t) => t.stop())
    void audioCtx.close().catch(() => undefined)
    void playCtx.close().catch(() => undefined)
  }
  ws.addEventListener('close', cleanup)

  const sendBinary = (tag: number, payload: Uint8Array): boolean => {
    if (ws.readyState !== WebSocket.OPEN) return false
    const out = new Uint8Array(1 + payload.byteLength)
    out[0] = tag
    out.set(payload, 1)
    ws.send(out.buffer)
    return true
  }

  const sendControl = (payload: Record<string, unknown>): boolean => {
    if (ws.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify(payload))
    return true
  }

  return {
    flush: () => {
      sendBinary(FRAME_FLUSH, new Uint8Array(0))
    },
    stop: async () => {
      try {
        if (ws.readyState === WebSocket.OPEN) {
          // Send a JSON `stop` so the server-side adapter loop drains
          // gracefully (vs us yanking the WS).
          ws.send(JSON.stringify({ event: 'stop' }))
        }
      } catch { /* */ }
      cancelPlayback()
      cleanup()
    },
    abort: () => {
      try { ws.close(1000, 'aborted') } catch { /* */ }
      cancelPlayback()
      cleanup()
    },
    ready: () => ready,
    sendBinary,
    cancelPlayback,
    sendControl,
  }
}

/** Tag for a video JPEG frame in the binary protocol — exported so
 *  `cameraStream.ts` doesn't have to redefine the constant. */
export const VIDEO_FRAME_TAG = FRAME_VIDEO
