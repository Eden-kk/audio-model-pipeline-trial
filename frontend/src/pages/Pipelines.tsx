import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listRecipes, type Recipe } from '../lib/api'
import { cx } from '../lib/cx'

const CATEGORY_BADGE: Record<string, string> = {
  asr: 'bg-blue-100 text-blue-800 border-blue-200',
  tts: 'bg-purple-100 text-purple-800 border-purple-200',
  speaker_verify: 'bg-amber-100 text-amber-800 border-amber-200',
  vad: 'bg-green-100 text-green-800 border-green-200',
  diarization: 'bg-indigo-100 text-indigo-800 border-indigo-200',
  intent_llm: 'bg-pink-100 text-pink-800 border-pink-200',
  realtime_omni: 'bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200',
  // Slow-loop categories (Slice 9.2)
  lid: 'bg-cyan-100 text-cyan-800 border-cyan-200',
  dispatch: 'bg-emerald-100 text-emerald-800 border-emerald-200',
}

function CategoryPill({ category }: { category: string }) {
  return (
    <span
      className={cx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono border',
        CATEGORY_BADGE[category] ?? 'bg-gray-100 text-gray-700 border-gray-200',
      )}
    >
      {category}
    </span>
  )
}

// Color of the SVG arrow stroke per upstream category (so an `audio_path`
// edge from an ASR stage looks different from a `speaker_segments` edge).
// Falls back to gray if the category isn't in the table.
const EDGE_STROKE: Record<string, string> = {
  asr: '#3b82f6',            // blue-500
  tts: '#a855f7',            // purple-500
  speaker_verify: '#f59e0b', // amber-500
  lid: '#06b6d4',            // cyan-500
  intent_llm: '#ec4899',     // pink-500
  vad: '#10b981',            // green-500
  diarization: '#6366f1',    // indigo-500
  realtime_omni: '#d946ef',  // fuchsia-500
  dispatch: '#10b981',       // emerald-500
}

/** Topological-wave layout for a recipe DAG.
 *
 * Stages with no incoming edge from each other land in the same wave
 * (column); waves chain left-to-right exactly as the runner executes them.
 * For slow-loop this gives `[asr | speaker_tag] → intent → dispatch` —
 * which mirrors the parallel execution the backend now does. */
function layoutWaves(stages: Recipe['stages'], edges: Recipe['edges']) {
  const ids = new Set(stages.map((s) => s.id))
  const deps: Record<string, Set<string>> = {}
  for (const s of stages) deps[s.id] = new Set()
  for (const e of edges) if (ids.has(e.to) && ids.has(e.from)) deps[e.to].add(e.from)

  const placed = new Set<string>()
  const waves: Recipe['stages'][] = []
  while (placed.size < stages.length) {
    const wave = stages.filter(
      (s) => !placed.has(s.id) && [...deps[s.id]].every((d) => placed.has(d)),
    )
    const next = wave.length > 0 ? wave : stages.filter((s) => !placed.has(s.id)).slice(0, 1)
    waves.push(next)
    for (const s of next) placed.add(s.id)
  }
  return waves
}

/** Compact SVG flow diagram: wave-laid-out category pills + per-edge
 *  arrows with port labels at the midpoint. The whole diagram is one SVG
 *  so arrows can route through the gaps between rows without colliding
 *  with the foreground HTML. */
function RecipeDag({ stages, edges }: { stages: Recipe['stages']; edges: Recipe['edges'] }) {
  const waves = layoutWaves(stages, edges)

  // Fixed geometry — easy to tweak. The diagram's bounding box scales with
  // the number of waves (cols) and the tallest wave (rows).
  const COL_W = 150        // horizontal pitch between waves
  const ROW_H = 56         // vertical pitch within a wave
  const PILL_W = 116       // pill bounding box for arrow endpoints
  const PILL_H = 30
  const PAD_X = 8
  const PAD_Y = 8

  // Record each stage's center coordinate so we can wire SVG arrows between
  // them. Arrow tail = right edge of source pill; arrow head = left edge of
  // target pill — gives a clean "data flows out the right, in the left".
  const pos: Record<string, { cx: number; cy: number; col: number; row: number }> = {}
  waves.forEach((wave, col) => wave.forEach((s, row) => {
    pos[s.id] = {
      cx: PAD_X + col * COL_W + PILL_W / 2,
      cy: PAD_Y + row * ROW_H + PILL_H / 2,
      col,
      row,
    }
  }))
  const stageById = Object.fromEntries(stages.map((s) => [s.id, s]))

  const tallest = Math.max(...waves.map((w) => w.length))
  const width = PAD_X * 2 + waves.length * COL_W - (COL_W - PILL_W)
  const height = PAD_Y * 2 + tallest * ROW_H - (ROW_H - PILL_H)

  return (
    <div className="relative" style={{ width, height }}>
      {/* SVG layer: arrows + port labels behind the pills */}
      <svg
        width={width}
        height={height}
        className="absolute inset-0 pointer-events-none"
        // Distinct arrowhead per category — cheaper than redefining per-arrow.
        // We register one marker per category we actually use.
      >
        <defs>
          {Object.entries(EDGE_STROKE).map(([cat, color]) => (
            <marker
              key={cat}
              id={`arrow-${cat}`}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
            </marker>
          ))}
          <marker
            id="arrow-default"
            viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
          </marker>
        </defs>

        {edges.map((e, i) => {
          const a = pos[e.from]
          const b = pos[e.to]
          if (!a || !b) return null
          const upstream = stageById[e.from]
          const stroke = (upstream && EDGE_STROKE[upstream.category]) || '#9ca3af'
          const markerId = (upstream && EDGE_STROKE[upstream.category]) ? `arrow-${upstream.category}` : 'arrow-default'

          // Tail/head sit on the pill's right/left edges so the arrow doesn't
          // visually overlap the rounded-rect text.
          const x1 = a.cx + PILL_W / 2
          const y1 = a.cy
          const x2 = b.cx - PILL_W / 2
          const y2 = b.cy

          // Cubic bezier with horizontal control handles — gives a smooth
          // curve when source/target are on different rows, and degrades to
          // a straight line when same-row.
          const dx = Math.max(20, (x2 - x1) / 2)
          const path = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`

          // Label position — midpoint of the chord, nudged above the curve.
          const lx = (x1 + x2) / 2
          const ly = (y1 + y2) / 2 - 6

          return (
            <g key={i}>
              <path
                d={path}
                stroke={stroke}
                strokeWidth={1.5}
                fill="none"
                markerEnd={`url(#${markerId})`}
                opacity={0.85}
              />
              {e.port && (
                <g>
                  {/* Pill background so the label is readable when crossing other arrows */}
                  <rect
                    x={lx - e.port.length * 3.2 - 4}
                    y={ly - 9}
                    width={e.port.length * 6.4 + 8}
                    height={13}
                    rx={3}
                    fill="#f9fafb"
                    stroke="#e5e7eb"
                    strokeWidth={0.5}
                  />
                  <text
                    x={lx}
                    y={ly}
                    textAnchor="middle"
                    fontSize="10"
                    fontFamily="ui-monospace, monospace"
                    fill="#4b5563"
                  >
                    {e.port}
                  </text>
                </g>
              )}
            </g>
          )
        })}
      </svg>

      {/* HTML layer: category pills positioned absolutely over the SVG */}
      {stages.map((s) => {
        const p = pos[s.id]
        if (!p) return null
        return (
          <div
            key={s.id}
            className="absolute flex items-center justify-center"
            style={{
              left: p.cx - PILL_W / 2,
              top: p.cy - PILL_H / 2,
              width: PILL_W,
              height: PILL_H,
            }}
            title={`${s.id} · ${s.category}`}
          >
            <CategoryPill category={s.category} />
          </div>
        )
      })}
    </div>
  )
}

function StageChain({ stages, edges }: { stages: Recipe['stages']; edges: Recipe['edges'] }) {
  // Single-stage recipes (asr-only / tts-only / speaker-verify-only) have
  // no edges to render — just show the lone pill so the diagram doesn't
  // collapse to an empty box.
  if (stages.length <= 1 || edges.length === 0) {
    return (
      <div className="flex items-center gap-2 flex-wrap">
        {stages.map((s, i) => (
          <div key={s.id} className="flex items-center gap-2">
            <CategoryPill category={s.category} />
            {i < stages.length - 1 && (
              <svg className="w-4 h-4 text-gray-400" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth={2}>
                <path d="M5 12h14M13 5l7 7-7 7" />
              </svg>
            )}
          </div>
        ))}
      </div>
    )
  }
  // Multi-stage with real edges → draw the wave-laid-out flow diagram with
  // SVG arrows + port labels.
  return (
    <div className="overflow-x-auto">
      <RecipeDag stages={stages} edges={edges} />
    </div>
  )
}

export default function Pipelines() {
  const [recipes, setRecipes] = useState<Recipe[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    listRecipes()
      .then((r) => { setRecipes(r); setLoading(false) })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
  }, [])

  return (
    <div className="flex flex-col h-full bg-gray-50">
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">Pipelines</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Pre-built recipe pipelines. Each declares an ordered list of stages
          by category — pick a concrete adapter per stage when you run.
          Drag-and-drop composer lands in P1.5.
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6 flex flex-col gap-6">
        {loading && (
          <div className="text-sm text-gray-500 italic">Loading recipes…</div>
        )}
        {error && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
            <span className="font-medium">Backend not reachable.</span>{' '}
            <span className="text-xs text-amber-700">{error}</span>
          </div>
        )}
        {!loading && !error && recipes.length === 0 && (
          <div className="text-sm text-gray-500 italic">No recipes yet.</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {recipes.map((r) => (
            <div key={r.id} className="card flex flex-col gap-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-base font-semibold text-gray-900">{r.name}</h3>
                  <p className="text-xs text-gray-500 mt-0.5 font-mono">{r.id}</p>
                </div>
                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200 text-xs">
                  {r.stages.length} stage{r.stages.length === 1 ? '' : 's'}
                </span>
              </div>

              <p className="text-sm text-gray-600 leading-relaxed">{r.description}</p>

              <div className="pt-2 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-2">Data flow</p>
                <StageChain stages={r.stages} edges={r.edges} />
              </div>

              <div className="flex items-center gap-2 pt-3">
                <Link to={`/run?recipe=${r.id}`} className="btn-pill-dark text-xs">
                  Open in Run →
                </Link>
                <button
                  type="button"
                  onClick={() => { void navigator.clipboard.writeText(JSON.stringify(r, null, 2)) }}
                  className="btn-pill-outline text-xs"
                  title="Copy recipe JSON"
                >
                  Copy JSON
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
