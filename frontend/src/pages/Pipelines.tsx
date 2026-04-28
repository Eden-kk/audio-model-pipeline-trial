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

function StageChain({ stages }: { stages: Recipe['stages'] }) {
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
                <p className="text-xs text-gray-500 mb-2">Stage chain</p>
                <StageChain stages={r.stages} />
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
