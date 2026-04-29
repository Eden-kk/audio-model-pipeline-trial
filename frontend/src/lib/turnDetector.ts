// Silero VAD wrapper for in-browser turn detection.
//
// Wraps `@ricky0123/vad-web` and exposes a stable `startVAD(...)` API the
// Realtime page calls. Same Silero v5 ONNX model that ports cleanly to
// VR-glass on-device deployments (Android NNAPI / iOS CoreML via
// onnxruntime-mobile) — keep the threshold tuning here in lockstep with
// what we'd ship to the headset.
//
// Asset paths are pinned to /vad/ which is served by Vite from
// frontend/public/vad/. See the install-time copy in the slice notes.
//
// Lifecycle:
//   const handle = await startVAD({ ... callbacks ... })
//   handle.start()              — begin listening (mic permission required)
//   handle.pause() / .resume()  — gate VAD without tearing down the model
//   handle.stop()               — release everything
//
// The library uses its own MediaStream (separate from wsAudio.ts's mic
// capture), but both ride on the same `getUserMedia` permission grant —
// the browser gives back the same physical mic to both consumers.

const VAD_BASE_PATH = '/vad/'   // public/vad/ served at this URL

export interface TurnDetectorHandle {
  start: () => Promise<void>
  pause: () => void
  resume: () => void
  stop: () => Promise<void>
  /** True between start() and stop(); independent of pause/resume state. */
  isRunning: () => boolean
  /** Most-recent voice-energy estimate (0..1). Pulled by the Realtime page
   *  for the live VU meter. */
  level: () => number
}

interface StartOptions {
  /** Called once when the user starts speaking after silence. */
  onSpeechStart?: () => void
  /** Called once when the user stops speaking (after the silence window). */
  onSpeechEnd?: (audioFloat32: Float32Array) => void
  /** Called continuously with frame-level data — drives a VU meter. */
  onFrameProcessed?: (probabilities: { isSpeech: number; notSpeech: number }) => void
  /** Called on permission denial / model-load failure. */
  onError?: (err: Error) => void
  /** Speech end fires after this many ms of below-threshold frames. Default 800. */
  silenceThresholdMs?: number
  /** Minimum utterance duration before speech_end can fire. Default 250ms. */
  minSpeechMs?: number
  /** Silero score threshold for "is speech" (0..1). Default 0.5; lower → more sensitive. */
  speechThreshold?: number
  /** Silero score threshold for "is silence" (0..1). Default 0.35. */
  silenceThreshold?: number
}

/** Lazy import — keeps the bundle small until the user opens /realtime.
 *
 *  vite.config.ts aliases `onnxruntime-web` → `ort.bundle.min.mjs` and
 *  `onnxruntime-web/wasm` → `ort.wasm.bundle.min.mjs`. Both bundles
 *  INLINE the wasm into the JS — but **vad-web bundles its OWN copy of
 *  ORT inside its prebundle** (separate module instance from our
 *  aliased onnxruntime-web), so our `numThreads`/`proxy` settings on the
 *  outer `ort` object don't affect what vad-web uses internally.
 *
 *  vad-web's `MicVAD.new` unconditionally writes its `onnxWASMBasePath`
 *  option into ITS embedded `ort.env.wasm.wasmPaths` (see
 *  vad-web/dist/index.js → the `exports.ort.env.wasm.wasmPaths = ...`
 *  line). The library's default for `onnxWASMBasePath` is `"./"`, which
 *  flips its embedded ORT into externally-fetched-wasm mode and bakes a
 *  RELATIVE path `"./"` into the wasm-loader's `locateFile`. The browser
 *  resolves `"./ort-wasm-simd-threaded.wasm"` against the document URL
 *  (http://localhost:5173/realtime), gets `/ort-wasm-simd-threaded.wasm`,
 *  hits Vite's SPA fallback, receives index.html, and tries to compile
 *  HTML as wasm → "expected magic word 00 61 73 6d, found 3c 21 64 6f"
 *  (3c 21 64 6f = `<!do…`).
 *
 *  Fix: pass an explicit `onnxWASMBasePath` pointing at a URL Vite will
 *  serve as a module. `/node_modules/onnxruntime-web/dist/` is handled
 *  by the `ortWasmWorkerPlugin` middleware in vite.config.ts which
 *  resolves the wasm/.mjs files to their real on-disk locations. We
 *  cannot use `/vad/` (where the files also exist) because Vite's dev
 *  server refuses to serve files from `/public/` with the `?import`
 *  query suffix that dynamic imports get — it returns a 500 with HTML.
 *
 *  Single-thread + no-proxy is still useful — keeps the binary small
 *  and avoids spawning a worker that the browser sandbox might block.
 */
async function loadVADLib() {
  const ort = await import('onnxruntime-web')
  // Do NOT set ort.env.wasm.wasmPaths — the bundled .mjs already has wasm.
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const mod = await import('@ricky0123/vad-web')
  return mod
}

export async function startVAD(opts: StartOptions): Promise<TurnDetectorHandle> {
  const {
    onSpeechStart,
    onSpeechEnd,
    onFrameProcessed,
    onError,
    silenceThresholdMs = 800,
    minSpeechMs = 250,
    speechThreshold = 0.5,
    silenceThreshold = 0.35,
  } = opts

  // Silero v5 runs at 16 kHz with 32 ms (512-sample) frames.
  const FRAME_MS = 32
  const minSpeechFrames = Math.max(1, Math.round(minSpeechMs / FRAME_MS))
  const redemptionFrames = Math.max(1, Math.round(silenceThresholdMs / FRAME_MS))

  let level = 0
  let running = false

  const lib = await loadVADLib()

  // Create the MicVAD instance. We use the modern v5 model (smaller +
  // more accurate than legacy). The `MicVAD` helper handles getUserMedia
  // + AudioWorkletNode wiring for us.
  let vad: { start: () => Promise<void>; pause: () => void; destroy: () => void } | null = null
  try {
    vad = await lib.MicVAD.new({
      // baseAssetPath: where MicVAD fetches the silero_vad_v5.onnx model
      // file. That's a plain HTTP fetch of a static asset (NOT a JS module
      // import), so /public/vad/ is fine for it.
      baseAssetPath: VAD_BASE_PATH,
      // Point vad-web's embedded ORT at the directory that holds the
      // ort-wasm-simd-threaded.mjs worker entrypoint.
      //
      // DEV: Vite's dev server refuses to serve /public/ files with the
      // `?import` suffix that dynamic-import requests carry (500 error).
      // Instead we point at /node_modules/onnxruntime-web/dist/, which
      // the ortWasmWorkerPlugin middleware in vite.config.ts intercepts
      // and streams from the real on-disk location.
      //
      // PROD: /node_modules/ doesn't exist under FastAPI's StaticFiles
      // mount — only dist/ is served. The /vad/ directory (copied from
      // public/vad/ by Vite) already contains the .mjs + .wasm files,
      // so we point there instead.  Static file fetches (no ?import
      // suffix) work fine from /public/ in production.
      onnxWASMBasePath: import.meta.env.DEV
        ? '/node_modules/onnxruntime-web/dist/'
        : '/vad/',
      model: 'v5',
      positiveSpeechThreshold: speechThreshold,
      negativeSpeechThreshold: silenceThreshold,
      minSpeechFrames,
      redemptionFrames,
      onSpeechStart: () => {
        onSpeechStart?.()
      },
      onSpeechEnd: (audio: Float32Array) => {
        onSpeechEnd?.(audio)
      },
      onFrameProcessed: (probs: { isSpeech: number; notSpeech: number }) => {
        level = probs.isSpeech
        onFrameProcessed?.(probs)
      },
      // VAD ships a mis-configuration warning callback; surface as error.
      onVADMisfire: () => {
        // Not actually an error — fired when speech started but didn't
        // last long enough to count. Silently ignore in v1.
      },
    } as Parameters<typeof lib.MicVAD.new>[0])
  } catch (e) {
    const err = e instanceof Error ? e : new Error(String(e))
    onError?.(new Error(`Silero VAD load failed: ${err.message}`))
    throw err
  }

  return {
    start: async () => {
      if (!vad) throw new Error('VAD not initialised')
      await vad.start()
      running = true
    },
    pause: () => {
      if (vad && running) vad.pause()
    },
    resume: async () => {
      if (vad && running) await vad.start()
    },
    stop: async () => {
      if (vad) {
        try { vad.destroy() } catch { /* */ }
      }
      vad = null
      running = false
    },
    isRunning: () => running,
    level: () => level,
  }
}
