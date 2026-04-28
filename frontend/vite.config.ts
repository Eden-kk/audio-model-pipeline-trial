import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Force every import of `onnxruntime-web` (whether from our
  // turnDetector.ts or transitively from vad-web's CJS chain) to resolve
  // to a self-contained ESM bundle. Without this alias, esbuild's
  // prebundling picks the `require` condition for vad-web's CJS, which
  // resolves to `dist/ort.min.js` — the non-bundled build that loads
  // `ort-wasm-simd-threaded.mjs` externally from /public/vad/. Vite
  // refuses to serve module imports from /public, so the runtime would
  // explode with "Failed to load url /vad/ort-wasm-...mjs". The bundled
  // .mjs inlines the wasm — no external fetch needed.
  //
  // Use regex aliases (exact-match) for both the bare specifier and the
  // ./wasm subpath. A plain string alias would prefix-match and rewrite
  // `onnxruntime-web/wasm` → `ort.bundle.min.mjs/wasm` (a bogus path).
  resolve: {
    alias: [
      {
        find: /^onnxruntime-web$/,
        replacement: fileURLToPath(
          new URL(
            './node_modules/onnxruntime-web/dist/ort.bundle.min.mjs',
            import.meta.url,
          ),
        ),
      },
      {
        find: /^onnxruntime-web\/wasm$/,
        replacement: fileURLToPath(
          new URL(
            './node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs',
            import.meta.url,
          ),
        ),
      },
    ],
  },
  // Slice O6.3 — Silero VAD support.
  //
  // Both `@ricky0123/vad-web` and `onnxruntime-web` ship CommonJS that
  // `require()`s into each other. Without prebundling, the browser ESM
  // runtime errors out twice:
  //   * "exports is not defined"   (vad-web's CJS dist)
  //   * "Calling require for onnxruntime-web/wasm in an environment that
  //      doesn't expose require"   (vad-web's runtime call into ORT)
  //
  // Fix: pre-bundle the whole CJS island via esbuild.  optimizeDeps.include
  // forces esbuild to follow every require(...) and produce an ESM-shaped
  // artifact for the browser.  We include both the `.` entry and the
  // `./wasm` subpath because vad-web's chain ends up requiring both.
  //
  // (Historical comment claimed onnxruntime-web's internal
  //  `import('./ort-wasm-*.mjs?import')` calls broke the optimizer; that
  //  was real on older versions but the modern 1.24.x ESM bundles
  //  (ort.bundle.min.mjs) are self-contained and prebundle cleanly.)
  //
  // We additionally set `ort.env.wasm.numThreads=1` + `proxy=false` in
  // src/lib/turnDetector.ts to opt into the single-file wasm path.
  optimizeDeps: {
    // Only vad-web needs prebundling now — `onnxruntime-web` is aliased
    // above to a single self-contained .mjs and resolves directly.
    include: ['@ricky0123/vad-web'],
  },
  assetsInclude: ['**/*.onnx'],
})
