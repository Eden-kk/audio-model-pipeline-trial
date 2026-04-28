import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Slice O6.3 — Silero VAD support.
  //
  // Two related dep quirks to handle:
  //
  // 1. `onnxruntime-web` does internal `import('./ort-wasm-*.mjs?import')`
  //    calls; Vite's optimizer + ESM transform tries to follow them and
  //    fails with "Failed to fetch dynamically imported module".
  //    Excluding from optimizeDeps avoids the transform.  We additionally
  //    configure `ort.env.wasm.numThreads=1` + `proxy=false` in
  //    src/lib/turnDetector.ts to opt into the single-file wasm path.
  //
  // 2. `@ricky0123/vad-web` ships CommonJS only (its dist/index.js is
  //    `exports.MicVAD = ...`, `require(...)`). The browser loads ESM, so
  //    if Vite leaves the package un-prebundled the runtime explodes with
  //    "exports is not defined". The fix is the OPPOSITE of (1) — INCLUDE
  //    it in optimizeDeps.include so esbuild pre-bundles it (CJS → ESM
  //    during prebundling).
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
    include: ['@ricky0123/vad-web'],
  },
  assetsInclude: ['**/*.onnx'],
})
