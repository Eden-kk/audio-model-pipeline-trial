import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactFlow, {
  Background, Controls, MarkerType, Position,
  type Edge, type Node, type NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  listRecipes, listAdapters, listClips, startRecipeRun,
  clipAudioUrl,
  type Recipe, type Adapter, type Clip, type RecipeRun, type StageRun,
} from '../lib/api'
import { cx } from '../lib/cx'
import SegmentTimeline, { type Segment } from '../components/SegmentTimeline'

const CATEGORY_BADGE: Record<string, string> = {
  asr: 'bg-blue-100 text-blue-800 border-blue-200',
  tts: 'bg-purple-100 text-purple-800 border-purple-200',
  speaker_verify: 'bg-amber-100 text-amber-800 border-amber-200',
  vad: 'bg-green-100 text-green-800 border-green-200',
  diarization: 'bg-indigo-100 text-indigo-800 border-indigo-200',
  intent_llm: 'bg-pink-100 text-pink-800 border-pink-200',
  realtime_omni: 'bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200',
  lid: 'bg-cyan-100 text-cyan-800 border-cyan-200',
  dispatch: 'bg-emerald-100 text-emerald-800 border-emerald-200',
}

type StageState = 'idle' | 'running' | 'done' | 'error'

interface StageNodeData {
  stageId: string
  category: string
  adapterId: string | null
  state: StageState
  latencyMs?: number
  outputPreview?: string
  error?: string
  onAdapterChange?: (adapterId: string) => void
  adapterChoices?: Adapter[]
}

// ─── Custom node ─────────────────────────────────────────────────────────────

function StageNode({ data }: NodeProps<StageNodeData>) {
  const { category, stageId, adapterId, state, latencyMs, outputPreview, error,
          onAdapterChange, adapterChoices } = data
  const choices = (adapterChoices ?? []).filter((a) => a.category === category)

  const ringColor =
    state === 'done'    ? 'ring-green-300'
  : state === 'running' ? 'ring-amber-300 animate-pulse'
  : state === 'error'   ? 'ring-red-300'
  :                       'ring-transparent'

  return (
    <div className={cx(
      'bg-white rounded-xl border border-gray-200 px-4 py-3 w-[260px] shadow-sm ring-2 ring-offset-1',
      ringColor,
    )}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="font-mono text-xs text-gray-500">{stageId}</span>
        <span className={cx(
          'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono border',
          CATEGORY_BADGE[category] ?? 'bg-gray-100 text-gray-700 border-gray-200',
        )}>
          {category}
        </span>
      </div>

      {onAdapterChange && choices.length > 0 ? (
        <select
          className="field text-xs w-full mb-2"
          value={adapterId ?? ''}
          onChange={(e) => onAdapterChange(e.target.value)}
          disabled={state === 'running'}
        >
          <option value="">— pick adapter —</option>
          {choices.map((a) => (
            <option key={a.id} value={a.id}>{a.display_name}</option>
          ))}
        </select>
      ) : (
        <p className="text-xs font-mono text-gray-700 mb-2 truncate">
          {adapterId ?? 'no adapter'}
        </p>
      )}

      {state === 'done' && latencyMs != null && (
        <div className="text-xs flex items-center gap-2">
          <span className="text-green-700">✓ {latencyMs.toFixed(0)} ms</span>
        </div>
      )}
      {state === 'running' && (
        <div className="text-xs text-amber-700">running…</div>
      )}
      {state === 'error' && (
        <div className="text-xs text-red-700 truncate" title={error}>✗ {error}</div>
      )}

      {outputPreview && state === 'done' && (
        <div className="text-xs text-gray-600 mt-2 line-clamp-2">{outputPreview}</div>
      )}
    </div>
  )
}

const nodeTypes = { stage: StageNode }

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Run() {
  const [params] = useSearchParams()
  const recipeIdParam = params.get('recipe')
  const clipIdParam = params.get('clip')

  const [recipes, setRecipes] = useState<Recipe[]>([])
  const [clips, setClips] = useState<Clip[]>([])
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [recipeId, setRecipeId] = useState<string>(recipeIdParam ?? '')
  const [clipId, setClipId] = useState<string>(clipIdParam ?? '')
  const [stageAdapters, setStageAdapters] = useState<Record<string, string>>({})
  const [stageStates, setStageStates] = useState<Record<string, StageState>>({})
  const [stageResults, setStageResults] = useState<Record<string, StageRun>>({})
  const [busy, setBusy] = useState(false)
  const [runResult, setRunResult] = useState<RecipeRun | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Initial load
  useEffect(() => {
    Promise.all([listRecipes(), listClips(), listAdapters()])
      .then(([rs, cs, as_]) => {
        setRecipes(rs); setClips(cs); setAdapters(as_)
        if (!recipeId && rs.length > 0) setRecipeId(rs[0].id)
        if (!clipId && cs.length > 0) setClipId(cs[0].id)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const recipe = useMemo(() => recipes.find((r) => r.id === recipeId), [recipes, recipeId])
  const clip = useMemo(() => clips.find((c) => c.id === clipId), [clips, clipId])

  // Reset stage state whenever recipe changes
  useEffect(() => {
    if (!recipe) return
    const initialAdapters: Record<string, string> = {}
    const initialStates: Record<string, StageState> = {}
    for (const s of recipe.stages) {
      // Auto-pick first adapter that matches the category
      const candidates = adapters.filter((a) => a.category === s.category)
      if (candidates.length > 0) initialAdapters[s.id] = candidates[0].id
      initialStates[s.id] = 'idle'
    }
    setStageAdapters(initialAdapters)
    setStageStates(initialStates)
    setStageResults({})
    setRunResult(null)
  }, [recipeId, recipe, adapters])

  // Build react-flow nodes + edges from the current recipe + state
  const { nodes, edges } = useMemo(() => {
    if (!recipe) return { nodes: [] as Node[], edges: [] as Edge[] }
    const nodes: Node[] = recipe.stages.map((s, i) => ({
      id: s.id,
      type: 'stage',
      position: { x: 80 + i * 320, y: 80 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      data: {
        stageId: s.id,
        category: s.category,
        adapterId: stageAdapters[s.id] ?? null,
        state: stageStates[s.id] ?? 'idle',
        latencyMs: stageResults[s.id]?.latency_ms,
        outputPreview: stageResults[s.id]?.output_preview,
        error: stageResults[s.id]?.error ?? undefined,
        adapterChoices: adapters,
        onAdapterChange: (adapterId: string) =>
          setStageAdapters((prev) => ({ ...prev, [s.id]: adapterId })),
      } satisfies StageNodeData,
    }))
    const edges: Edge[] = recipe.edges.map((e, i) => ({
      id: `e${i}`,
      source: e.from,
      target: e.to,
      label: e.port,
      labelStyle: { fontSize: 10, fill: '#6b7280' },
      labelBgStyle: { fill: '#f9fafb' },
      style: { stroke: '#9ca3af', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#9ca3af' },
    }))
    return { nodes, edges }
  }, [recipe, adapters, stageAdapters, stageStates, stageResults])

  async function handleRun() {
    if (!recipe || !clip) return
    setBusy(true); setError(null); setRunResult(null)
    setStageStates(Object.fromEntries(recipe.stages.map((s) => [s.id, 'running'])))
    setStageResults({})
    try {
      const res = await startRecipeRun(clip.id, recipe.id, stageAdapters)
      setRunResult(res)
      const newStates: Record<string, StageState> = {}
      const newResults: Record<string, StageRun> = {}
      for (const s of res.stages) {
        newStates[s.stage_id] = s.error ? 'error' : 'done'
        newResults[s.stage_id] = s
      }
      // Stages that never ran (because an earlier one failed) → idle
      for (const stg of recipe.stages) {
        if (!(stg.id in newStates)) newStates[stg.id] = 'idle'
      }
      setStageStates(newStates)
      setStageResults(newResults)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
      setStageStates(Object.fromEntries(recipe.stages.map((s) => [s.id, 'error'])))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">Run</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Visualize a multi-stage pipeline as a port-typed DAG. Pick an adapter
          per stage; nodes light up as the run progresses.
        </p>
      </div>

      {/* Controls strip */}
      <div className="border-b border-gray-200 bg-white px-6 py-3 flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Pipeline (recipe)</label>
          <select
            value={recipeId}
            onChange={(e) => setRecipeId(e.target.value)}
            disabled={busy}
            className="field text-sm min-w-[220px]"
          >
            {recipes.map((r) => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Clip</label>
          <select
            value={clipId}
            onChange={(e) => setClipId(e.target.value)}
            disabled={busy}
            className="field text-sm min-w-[280px]"
          >
            {clips.length === 0 && <option value="">(no clips — record one)</option>}
            {clips.map((c) => (
              <option key={c.id} value={c.id}>
                {c.original_filename || c.filename} · {c.duration_s.toFixed(1)}s
              </option>
            ))}
          </select>
        </div>

        {clip && (
          <audio controls preload="metadata" className="h-9" src={clipAudioUrl(clip.id)}>
            <track kind="captions" />
          </audio>
        )}

        <button
          type="button"
          onClick={() => void handleRun()}
          disabled={!recipe || !clip || busy
            || recipe.stages.some((s) => !stageAdapters[s.id])}
          className="btn-pill-dark ml-auto"
        >
          {busy ? 'Running…' : 'Run pipeline'}
        </button>
      </div>

      {error && (
        <div className="mx-6 mt-4 bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* DAG canvas */}
      <div className="flex-1 m-6 mt-4 bg-white border border-gray-200 rounded-xl overflow-hidden">
        {recipe ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.25, minZoom: 0.5, maxZoom: 1.5 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#e5e7eb" gap={16} />
            <Controls position="bottom-left" showInteractive={false} />
          </ReactFlow>
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500 text-sm">
            Pick a recipe to render its DAG.
          </div>
        )}
      </div>

      {/* Run result summary */}
      {runResult && (
        <div className="mx-6 mb-3 card text-sm flex flex-wrap items-center gap-4">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Run summary</span>
          <span className="font-mono text-gray-900">
            {runResult.total_latency_ms.toFixed(0)} ms total
          </span>
          {runResult.total_cost_usd > 0 && (
            <span className="font-mono text-gray-700">
              ${runResult.total_cost_usd.toFixed(5)}
            </span>
          )}
          {runResult.error ? (
            <span className="text-red-700 text-xs">error: {runResult.error}</span>
          ) : (
            <span className="text-green-700 text-xs">✓ all stages OK</span>
          )}
          <span className="text-xs text-gray-500 ml-auto font-mono">
            {runResult.id.slice(0, 12)}…
          </span>
        </div>
      )}

      {/* Slow-loop drill-in: per-segment user-tag timeline + envelope JSON */}
      {runResult && runResult.stages.map((s) => {
        const result = s.result as Record<string, unknown> | null | undefined
        // speaker_tag stage with verify_segments() output
        if (s.category === 'speaker_verify' && result && Array.isArray((result as { segments?: unknown[] }).segments)) {
          const segs = (result as { segments: Segment[] }).segments
          const dur = (result as { duration_s?: number }).duration_s
          const thr = (result as { threshold?: number }).threshold
          return (
            <div key={`${s.stage_id}-tl`} className="mx-6 mb-3 card">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs text-gray-500 uppercase tracking-wider">
                  Stage drill-in · {s.stage_id} ({s.adapter})
                </span>
                <span className="text-xs text-gray-500 font-mono ml-auto">
                  {s.latency_ms?.toFixed(0)} ms
                </span>
              </div>
              <SegmentTimeline segments={segs} duration_s={dur} threshold={thr} />
            </div>
          )
        }
        // intent_llm stage envelope
        if (s.category === 'intent_llm' && result) {
          const r = result as { memory_doc?: string; tool_calls?: unknown[]; salient_facts?: string[] }
          return (
            <div key={`${s.stage_id}-env`} className="mx-6 mb-3 card">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs text-gray-500 uppercase tracking-wider">
                  Envelope · {s.stage_id} ({s.adapter})
                </span>
                <span className="text-xs text-gray-500 font-mono ml-auto">
                  {s.latency_ms?.toFixed(0)} ms
                </span>
              </div>
              {r.memory_doc && (
                <div className="mb-2">
                  <p className="text-xs text-gray-500">memory_doc</p>
                  <p className="text-sm text-gray-900 whitespace-pre-wrap">{r.memory_doc}</p>
                </div>
              )}
              {r.tool_calls && r.tool_calls.length > 0 && (
                <div className="mb-2">
                  <p className="text-xs text-gray-500">tool_calls</p>
                  <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-auto max-h-40">
{JSON.stringify(r.tool_calls, null, 2)}
                  </pre>
                </div>
              )}
              {r.salient_facts && r.salient_facts.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500">salient_facts</p>
                  <ul className="text-sm text-gray-900 list-disc list-inside">
                    {r.salient_facts.map((f, i) => <li key={i}>{f}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )
        }
        return null
      })}
    </div>
  )
}
