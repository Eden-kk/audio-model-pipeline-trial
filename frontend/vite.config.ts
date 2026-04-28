import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'

// Vite plugin: serve ORT's wasm worker .mjs files from
// node_modules/onnxruntime-web/dist/ when the prebundled chunks request
// them via /node_modules/.vite/deps/ort-wasm-simd-threaded(.jsep)?.mjs.
//
// Why we need this: esbuild's optimizeDeps prebundles vad-web + ORT but
// leaves the wasm worker .mjs files behind — they're not statically
// imported, only dynamically. The prebundled chunk does
// `import('./ort-wasm-simd-threaded.mjs?import')` against `.vite/deps/`,
// the file isn't there, and we get a 404 ("Failed to fetch dynamically
// imported module"). This plugin makes those URLs resolve to the real
// files inside node_modules.
function ortWasmWorkerPlugin(): Plugin {
  const ortDistDir = fileURLToPath(
    new URL('./node_modules/onnxruntime-web/dist/', import.meta.url),
  )
  return {
    name: 'ort-wasm-worker',
    enforce: 'pre',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url || '').split('?')[0]
        // Match the worker .mjs / .wasm file by basename, regardless of
        // where in node_modules the path resolved to. esbuild's prebundle
        // rewrites the dynamic-import URL to the actual resolved location
        // (under .pnpm/onnxruntime-web@x.y.z/... when pnpm is the package
        // manager), so a regex tied to .vite/deps/ would miss it.
        const m = url.match(
          /\/(ort-wasm-simd-threaded(?:\.(?:jsep|jspi|asyncify))?\.(?:mjs|wasm))$/,
        )
        if (!m) return next()
        // Only intercept when the request is somewhere under node_modules
        // (so we don't accidentally hijack a same-named file the app code
        // copied elsewhere). Both /node_modules/.vite/deps/... and
        // /node_modules/.pnpm/... and plain /node_modules/onnxruntime-web/...
        // are valid.
        if (!url.includes('/node_modules/')) return next()
        const filePath = path.join(ortDistDir, m[1])
        if (!existsSync(filePath)) return next()
        readFile(filePath)
          .then((buf) => {
            res.setHeader(
              'Content-Type',
              filePath.endsWith('.wasm')
                ? 'application/wasm'
                : 'application/javascript',
            )
            res.end(buf)
          })
          .catch(next)
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), ortWasmWorkerPlugin()],
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
