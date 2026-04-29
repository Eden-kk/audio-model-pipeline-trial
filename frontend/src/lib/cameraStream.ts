// Browser webcam capture for realtime omni video input.
//
// Pulls frames off `<video>.srcObject` via a hidden `<canvas>`, encodes
// each as JPEG (image/jpeg q=0.7 ≈ 80 KB at 640×480), and forwards them
// onto the existing omni WS using the session's tagged-binary `sendBinary`
// hook (see lib/wsAudio.ts VIDEO_FRAME_TAG = 0x02).
//
// Defaults match the Plan C-extended-2 numbers: 2 fps, 640×480, q=0.7.
// Both fps and resolution are caller-controlled — the Realtime page will
// expose an fps dial and the resolution is implicit in the source video.

import { VIDEO_FRAME_TAG, type OmniSessionHandle } from './wsAudio'

export interface CameraCaptureHandle {
  /** Stop the interval timer + release the MediaStream. */
  stop: () => void
  /** Live count of frames sent — useful for the UI's "frames sent" stat. */
  framesSent: () => number
  /** Most-recent encoded JPEG bytes (for thumbnail strip). */
  lastFrameBytes: () => Uint8Array | null
}

interface StartOptions {
  /** Existing omni session to ride alongside the audio stream. */
  session: OmniSessionHandle
  /** The page's <video> element — driven by srcObject = MediaStream. */
  videoEl: HTMLVideoElement
  /** Capture rate. Default 2; cap is 5 (Plan C-extended-2 risk: bandwidth). */
  fps?: number
  /** JPEG quality 0..1, default 0.7 (~80 KB at 640×480). */
  quality?: number
  /** Optional: called every time a new frame is encoded + sent. */
  onFrameSent?: (jpeg: Uint8Array, bytes: number, idx: number) => void
  onError?: (err: Error) => void
}

/** Open the webcam, start streaming JPEG frames at `fps`. Throws if camera
 *  permission is denied. */
export async function startCameraCapture({
  session,
  videoEl,
  fps = 2,
  quality = 0.7,
  onFrameSent,
  onError,
}: StartOptions): Promise<CameraCaptureHandle> {
  const cappedFps = Math.max(0.5, Math.min(5, fps))
  const intervalMs = Math.round(1000 / cappedFps)

  // ── 1) Get camera
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    })
  } catch (err) {
    const e = err instanceof Error ? err : new Error(String(err))
    onError?.(new Error(`Camera permission denied: ${e.message}`))
    throw e
  }

  // ── 2) Pump into the page's <video> (for self-view)
  videoEl.srcObject = stream
  videoEl.muted = true   // prevent feedback loops if the page later unmutes
  videoEl.playsInline = true
  await videoEl.play().catch(() => undefined)

  // ── 3) Off-screen <canvas> for snapshotting. Sized lazily — we wait
  //    for the first metadata event so the natural width/height are known.
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d', { alpha: false })
  if (!ctx) {
    stream.getTracks().forEach((t) => t.stop())
    throw new Error('Browser refused 2d canvas context — cannot capture frames.')
  }

  const sizeFromVideo = () => {
    canvas.width = videoEl.videoWidth || 640
    canvas.height = videoEl.videoHeight || 480
  }
  sizeFromVideo()
  videoEl.addEventListener('loadedmetadata', sizeFromVideo)

  // ── 4) Capture loop
  let framesSent = 0
  let lastBytes: Uint8Array | null = null
  let stopped = false

  const tick = async () => {
    if (stopped) return
    if (videoEl.readyState < 2 /* HAVE_CURRENT_DATA */) return
    try {
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob((b) => resolve(b), 'image/jpeg', quality),
      )
      if (!blob || stopped) return
      const buf = new Uint8Array(await blob.arrayBuffer())
      const ok = session.sendBinary(VIDEO_FRAME_TAG, buf)
      if (!ok) return   // WS closed; will catch up on next tick
      framesSent += 1
      lastBytes = buf
      onFrameSent?.(buf, buf.byteLength, framesSent)
    } catch (e) {
      onError?.(new Error(`frame encode failed: ${e instanceof Error ? e.message : String(e)}`))
    }
  }

  const handle = window.setInterval(() => { void tick() }, intervalMs)

  return {
    stop: () => {
      stopped = true
      window.clearInterval(handle)
      try { videoEl.removeEventListener('loadedmetadata', sizeFromVideo) } catch { /* */ }
      stream.getTracks().forEach((t) => t.stop())
      try { videoEl.srcObject = null } catch { /* */ }
    },
    framesSent: () => framesSent,
    lastFrameBytes: () => lastBytes,
  }
}
