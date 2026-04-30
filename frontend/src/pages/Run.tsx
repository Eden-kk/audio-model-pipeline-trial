import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactFlow, {
  Background, Controls, Handle, MarkerType, Position,
  type Edge, type Node, type NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  listRecipes, listAdapters, listClips, listEnrollments, startRecipeRun,
  clipAudioUrl,
  type Recipe, type Adapter, type Clip, type Enrollment, type RecipeRun, type StageRun,
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

// Edge stroke color per upstream stage's category — kept in sync with the
// pill palette in Pipelines.tsx so an `asr → intent` arrow on Run looks the
// same blue as it does in the recipe gallery.
const EDGE_STROKE: Record<string, string> = {
  asr: '#3b82f6',
  tts: '#a855f7',
  speaker_verify: '#f59e0b',
  lid: '#06b6d4',
  intent_llm: '#ec4899',
  vad: '#10b981',
  diarization: '#6366f1',
  realtime_omni: '#d946ef',
  dispatch: '#10b981',
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
  /** Pre-run hint shown under the adapter dropdown — currently used for
   *  speaker_verify ↔ enrolled-profile adapter mismatches. */
  warning?: string
  onAdapterChange?: (adapterId: string) => void
  adapterChoices?: Adapter[]
}

// ─── Custom node ─────────────────────────────────────────────────────────────

function StageNode({ data }: NodeProps<StageNodeData>) {
  const { category, stageId, adapterId, state, latencyMs, outputPreview, error,
          warning, onAdapterChange, adapterChoices } = data
  const choices = (adapterChoices ?? []).filter((a) => a.category === category)

  const ringColor =
    state === 'done'    ? 'ring-green-300'
  : state === 'running' ? 'ring-amber-300 animate-pulse'
  : state === 'error'   ? 'ring-red-300'
  :                       'ring-transparent'

  return (
    <div className={cx(
      'bg-white rounded-xl border border-gray-200 px-4 py-3 w-[260px] shadow-sm ring-2 ring-offset-1 relative',
      ringColor,
    )}>
      {/* Handles are the anchor points react-flow uses to draw edges. We
          register both a target (left) and a source (right) on every node;
          stages with no incoming/outgoing edge in the recipe just leave
          their unused handle un-referenced. Without these, react-flow
          silently drops every edge. */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#9ca3af', width: 8, height: 8, border: '2px solid white' }}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#9ca3af', width: 8, height: 8, border: '2px solid white' }}
      />
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

      {(adapterChoices ?? []).find((a) => a.id === adapterId)?.multilang && (
        <span className="inline-flex self-start px-2 py-0.5 rounded-full text-[11px] bg-indigo-50 text-indigo-700 border border-indigo-200 mb-2">
          Multi-lang
        </span>
      )}

      {warning && state !== 'done' && state !== 'error' && (
        <div
          className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1 mb-2"
          title={warning}
        >
          ⚠ {warning}
        </div>
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
  const [enrollments, setEnrollments] = useState<Enrollment[]>([])
  const [recipeId, setRecipeId] = useState<string>(recipeIdParam ?? '')
  const [clipId, setClipId] = useState<string>(clipIdParam ?? '')
  const [stageAdapters, setStageAdapters] = useState<Record<string, string>>({})
  const [stageStates, setStageStates] = useState<Record<string, StageState>>({})
  const [stageResults, setStageResults] = useState<Record<string, StageRun>>({})
  const [busy, setBusy] = useState(false)
  const [runResult, setRunResult] = useState<RecipeRun | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Initial load — also pulls enrollments so the speaker_verify stage's
  // adapter dropdown can default to whatever the wearer was enrolled with.
  // Mismatching adapter ↔ enrollment causes a 256-vs-512 dim error otherwise.
  useEffect(() => {
    Promise.all([listRecipes(), listClips(), listAdapters(), listEnrollments()])
      .then(([rs, cs, as_, es]) => {
        setRecipes(rs); setClips(cs); setAdapters(as_); setEnrollments(es)
        if (!recipeId && rs.length > 0) setRecipeId(rs[0].id)
        if (!clipId && cs.length > 0) setClipId(cs[0].id)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const recipe = useMemo(() => recipes.find((r) => r.id === recipeId), [recipes, recipeId])
  const clip = useMemo(() => clips.find((c) => c.id === clipId), [clips, clipId])

  // Reset stage state whenever recipe changes. For speaker_verify stages,
  // bias the default toward whichever adapter the wearer's profile was
  // enrolled with — Resemblyzer (256-d) and pyannote_verify (512-d)
  // embeddings are NOT cross-comparable, and using the wrong one throws a
  // numpy shape mismatch deep in the pipeline.
  useEffect(() => {
    if (!recipe) return
    const wearerAdapter = enrollments.find((e) => e.profile_id === 'wearer')?.adapter
    const initialAdapters: Record<string, string> = {}
    const initialStates: Record<string, StageState> = {}
    for (const s of recipe.stages) {
      const candidates = adapters.filter((a) => a.category === s.category)
      if (candidates.length === 0) { initialStates[s.id] = 'idle'; continue }

      let pick = candidates[0].id
      if (s.category === 'speaker_verify' && wearerAdapter) {
        const match = candidates.find((a) => a.id === wearerAdapter)
        if (match) pick = match.id
      }
      initialAdapters[s.id] = pick
      initialStates[s.id] = 'idle'
    }
    setStageAdapters(initialAdapters)
    setStageStates(initialStates)
    setStageResults({})
    setRunResult(null)
  }, [recipeId, recipe, adapters, enrollments])

  // Build react-flow nodes + edges from the current recipe + state
  const { nodes, edges } = useMemo(() => {
    if (!recipe) return { nodes: [] as Node[], edges: [] as Edge[] }

    // Topological-wave layout: stages with no edge between them sit in the
    // same wave (column), so e.g. slow-loop's `asr` and `speaker_tag` —
    // which both read the clip and run concurrently in the backend — render
    // stacked in one column rather than strung out left-to-right.
    const deps: Record<string, Set<string>> = {}
    for (const s of recipe.stages) deps[s.id] = new Set()
    for (const e of recipe.edges) {
      if (deps[e.to]) deps[e.to].add(e.from)
    }
    const placed = new Set<string>()
    const waves: typeof recipe.stages[] = []
    while (placed.size < recipe.stages.length) {
      const wave = recipe.stages.filter(
        (s) => !placed.has(s.id) && [...deps[s.id]].every((d) => placed.has(d)),
      )
      const next = wave.length > 0 ? wave : recipe.stages.filter((s) => !placed.has(s.id)).slice(0, 1)
      waves.push(next)
      for (const s of next) placed.add(s.id)
    }
    const wavePos: Record<string, { col: number; row: number }> = {}
    waves.forEach((wave, col) => wave.forEach((s, row) => { wavePos[s.id] = { col, row } }))

    // Column pitch must leave enough horizontal slack for edge labels to
    // sit between two 260-px-wide nodes without overlapping either. With a
    // 260 px node and ~140 px gap, "memory_doc" / "speaker_segments" /
    // "language" all fit on a white pill in the middle of the arrow.
    const COL_W = 400
    const ROW_H = 200
    // Wearer's enrolled adapter — used to flag speaker_verify stages whose
    // adapter dropdown the user has flipped to a non-matching choice.
    const wearer = enrollments.find((e) => e.profile_id === 'wearer')

    const nodes: Node[] = recipe.stages.map((s) => {
      const { col, row } = wavePos[s.id]
      const picked = stageAdapters[s.id] ?? null

      let warning: string | undefined
      if (s.category === 'speaker_verify' && wearer && picked && picked !== wearer.adapter) {
        warning = (
          `Wearer enrolled with ${wearer.adapter} (${wearer.embedding_dim}-d). `
          + `Switch back, or re-enroll with the new adapter — running this will fail.`
        )
      } else if (s.category === 'speaker_verify' && !wearer) {
        warning = 'No wearer enrolled — go to Settings → Wearer enrollment first.'
      }

      return {
        id: s.id,
        type: 'stage',
        position: { x: 80 + col * COL_W, y: 80 + row * ROW_H },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        data: {
          stageId: s.id,
          category: s.category,
          adapterId: picked,
          state: stageStates[s.id] ?? 'idle',
          latencyMs: stageResults[s.id]?.latency_ms,
          outputPreview: stageResults[s.id]?.output_preview,
          error: stageResults[s.id]?.error ?? undefined,
          warning,
          adapterChoices: adapters,
          onAdapterChange: (adapterId: string) =>
            setStageAdapters((prev) => ({ ...prev, [s.id]: adapterId })),
        } satisfies StageNodeData,
      }
    })
    // Color the arrow by the upstream stage's category — so the audio→ASR
    // hop looks distinct from speaker_segments→intent. Animate the arrow
    // while the upstream stage is running so the user sees data "in flight".
    const edges: Edge[] = recipe.edges.map((e, i) => {
      const fromStage = recipe.stages.find((s) => s.id === e.from)
      const cat = fromStage?.category ?? ''
      const stroke = EDGE_STROKE[cat] ?? '#9ca3af'
      const upstreamState = stageStates[e.from] ?? 'idle'
      const downstreamState = stageStates[e.to] ?? 'idle'
      // Live animation: arrow pulses while upstream is running OR downstream
      // just started consuming. Stops once both are settled.
      const animated = upstreamState === 'running'
        || (upstreamState === 'done' && downstreamState === 'running')
      return {
        id: `e${i}`,
        source: e.from,
        target: e.to,
        type: 'smoothstep',
        animated,
        label: e.port,
        labelStyle: { fontSize: 11, fill: '#374151', fontFamily: 'ui-monospace, monospace' },
        // Fully opaque background + thin border so the label always reads
        // cleanly even if it lands close to a node's rounded corner. Larger
        // padding gives a real "pill" look.
        labelBgStyle: { fill: '#ffffff', fillOpacity: 1, stroke: '#e5e7eb', strokeWidth: 1 },
        labelBgPadding: [8, 4] as [number, number],
        labelBgBorderRadius: 6,
        labelShowBg: true,
        style: { stroke, strokeWidth: 2.25 },
        markerEnd: { type: MarkerType.ArrowClosed, color: stroke, width: 18, height: 18 },
      }
    })
    return { nodes, edges }
  }, [recipe, adapters, enrollments, stageAdapters, stageStates, stageResults])

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

      {/* Slow-loop drill-in: ASR transcript + per-segment user-tag timeline + envelope JSON */}
      {runResult && runResult.stages.map((s) => {
        const result = s.result as Record<string, unknown> | null | undefined
        // ASR stage — explicit, full transcript card so the user never has
        // to expand a tooltip to see what was heard.
        if (s.category === 'asr' && result && typeof (result as { text?: unknown }).text === 'string') {
          const r = result as { text: string; words?: unknown[]; language?: string }
          const wordCount = Array.isArray(r.words) ? r.words.length : 0
          return (
            <div key={`${s.stage_id}-asr`} className="mx-6 mb-3 card">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs text-gray-500 uppercase tracking-wider">
                  Transcript · {s.stage_id} ({s.adapter})
                </span>
                <span className="text-xs text-gray-500 font-mono ml-auto flex items-center gap-3">
                  {r.language && <span>lang={r.language}</span>}
                  {wordCount > 0 && <span>{wordCount} words</span>}
                  <span>{s.latency_ms?.toFixed(0)} ms</span>
                </span>
              </div>
              {r.text.trim().length > 0 ? (
                <p className="text-sm text-gray-900 whitespace-pre-wrap leading-relaxed">
                  {r.text}
                </p>
              ) : (
                <p className="text-sm text-gray-500 italic">(empty transcript)</p>
              )}
            </div>
          )
        }
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
