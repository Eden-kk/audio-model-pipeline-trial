import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  listClips,
  updateClipTags,
  deleteClip,
  clipAudioUrl,
  autotagClip,
  autotagAllClips,
  type Clip,
  type AutoTagResult,
} from '../lib/api'
import { cx } from '../lib/cx'

const SCENARIO_PALETTE = [
  'outdoor-traffic', 'indoor-cafe', 'multi-speaker', 'code-switch',
  'accented', 'whisper-voice', 'phone-call', 'noisy-restaurant',
  'quiet-office', 'wearer-walking',
]

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtDuration(s: number): string {
  if (!s || !isFinite(s)) return '—'
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60); const sec = Math.floor(s % 60)
  return `${m}m ${sec}s`
}

function fmtTimestamp(iso: string): string {
  if (!iso) return ''
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

function ChipToggle({
  label, active, onClick, disabled,
}: { label: string; active: boolean; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cx(
        'px-2.5 py-1 rounded-full text-xs border transition-colors',
        active
          ? 'bg-gray-900 text-white border-gray-900'
          : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100',
        disabled && 'opacity-50 cursor-not-allowed',
      )}
    >
      {label}
    </button>
  )
}

function ModalityBadge({ modality }: { modality: string }) {
  return (
    <span className={cx(
      'inline-flex items-center px-2 py-0.5 rounded-full text-xs border',
      modality === 'video'
        ? 'bg-purple-100 text-purple-800 border-purple-200'
        : 'bg-blue-100 text-blue-800 border-blue-200',
    )}>
      {modality}
    </span>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Corpus() {
  const [clips, setClips] = useState<Clip[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filterScenarios, setFilterScenarios] = useState<Set<string>>(new Set())
  const [filterModality, setFilterModality] = useState<'all' | 'audio' | 'video'>('all')
  const [search, setSearch] = useState('')
  const [busyClipId, setBusyClipId] = useState<string | null>(null)
  // Auto-tagger UX: per-clip evidence for tooltips, and a global "running"
  // flag while the batch button is in flight so the user gets a progress hint.
  const [autoEvidence, setAutoEvidence] = useState<Record<string, AutoTagResult>>({})
  const [autoRunning, setAutoRunning] = useState<'none' | 'all' | 'one'>('none')
  const [autoProgress, setAutoProgress] = useState<string>('')

  function refresh() {
    setLoading(true)
    listClips()
      .then((c) => { setClips(c); setLoading(false); setError(null) })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
  }

  useEffect(() => { refresh() }, [])

  // Sort newest first
  const sorted = useMemo(() => {
    return [...clips].sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
  }, [clips])

  // Apply filters
  const filtered = useMemo(() => {
    return sorted.filter((c) => {
      if (filterModality !== 'all' && c.modality !== filterModality) return false
      if (filterScenarios.size > 0) {
        const has = c.scenarios.some((s) => filterScenarios.has(s))
        if (!has) return false
      }
      if (search.trim()) {
        const q = search.trim().toLowerCase()
        const hay = `${c.original_filename} ${c.user_tags.join(' ')} ${c.scenarios.join(' ')} ${c.language_detected ?? ''}`
        if (!hay.toLowerCase().includes(q)) return false
      }
      return true
    })
  }, [sorted, filterScenarios, filterModality, search])

  function toggleFilterScenario(tag: string) {
    setFilterScenarios((prev) => {
      const next = new Set(prev)
      if (next.has(tag)) next.delete(tag); else next.add(tag)
      return next
    })
  }

  async function toggleClipScenario(clip: Clip, tag: string) {
    setBusyClipId(clip.id)
    const next = clip.scenarios.includes(tag)
      ? clip.scenarios.filter((t) => t !== tag)
      : [...clip.scenarios, tag]
    try {
      const updated = await updateClipTags(clip.id, { scenarios: next })
      setClips((cs) => cs.map((c) => (c.id === clip.id ? updated : c)))
    } catch (e: unknown) {
      console.error(e)
    } finally {
      setBusyClipId(null)
    }
  }

  async function handleAutoTagOne(clip: Clip, replace = false) {
    setBusyClipId(clip.id); setAutoRunning('one')
    try {
      const res = await autotagClip(clip.id, { replace })
      setAutoEvidence((prev) => ({ ...prev, [clip.id]: res }))
      setClips((cs) => cs.map((c) => (
        c.id === clip.id ? { ...c, scenarios: res.final_scenarios } : c
      )))
    } catch (e: unknown) {
      console.error(e)
      alert(`Auto-tag failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusyClipId(null); setAutoRunning('none')
    }
  }

  async function handleAutoTagAll(replace = false) {
    if (clips.length === 0) return
    const confirmMsg = replace
      ? `Replace scenarios on all ${clips.length} clips with auto-detected tags? Existing manual tags will be lost.`
      : `Add auto-detected scenarios to all ${clips.length} clips? Existing tags are preserved.`
    if (!confirm(confirmMsg)) return
    setAutoRunning('all')
    setAutoProgress(`Processing ${clips.length} clips…`)
    try {
      const res = await autotagAllClips({ replace })
      const evMap: Record<string, AutoTagResult> = {}
      for (const r of res.results) evMap[r.clip_id] = r
      setAutoEvidence((prev) => ({ ...prev, ...evMap }))
      // Patch local clip state with new scenarios in one pass.
      setClips((cs) => cs.map((c) => {
        const r = evMap[c.id]
        return r ? { ...c, scenarios: r.final_scenarios } : c
      }))
      setAutoProgress(
        `Auto-tagged ${res.succeeded}/${res.total} clip${res.total === 1 ? '' : 's'}`
        + (res.failed > 0 ? ` · ${res.failed} failed` : ''),
      )
    } catch (e: unknown) {
      console.error(e)
      setAutoProgress(`Auto-tag-all failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setAutoRunning('none')
      // Clear the toast after a few seconds so it doesn't linger.
      window.setTimeout(() => setAutoProgress(''), 6000)
    }
  }

  async function handleDelete(clip: Clip) {
    if (!confirm(`Delete clip "${clip.original_filename}"?`)) return
    setBusyClipId(clip.id)
    try {
      await deleteClip(clip.id)
      setClips((cs) => cs.filter((c) => c.id !== clip.id))
    } catch (e: unknown) {
      console.error(e)
    } finally {
      setBusyClipId(null)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Corpus</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Stored clips from upload + recording. Click <span className="font-mono text-xs bg-gray-100 px-1 rounded">Auto-tag all</span> to
            run heuristic scenario detection (SNR, spectrum, sample-rate,
            optional Whisper-LID + speaker-spread) over every clip.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {autoProgress && (
            <span className="text-xs text-gray-600 font-mono mr-2">{autoProgress}</span>
          )}
          <button
            type="button"
            onClick={() => void handleAutoTagAll(false)}
            disabled={autoRunning !== 'none' || clips.length === 0}
            className="btn-pill-dark text-xs"
            title="Detect scenarios for all clips and merge with existing tags"
          >
            {autoRunning === 'all' ? '⟳ Auto-tagging…' : '✨ Auto-tag all'}
          </button>
          <button
            type="button"
            onClick={() => void handleAutoTagAll(true)}
            disabled={autoRunning !== 'none' || clips.length === 0}
            className="btn-pill-outline text-xs"
            title="Replace existing scenarios with auto-detected ones"
          >
            replace mode
          </button>
          <button onClick={refresh} className="btn-pill-outline text-xs">
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Filter bar */}
      <div className="border-b border-gray-200 bg-white px-6 py-3 flex flex-wrap gap-3 items-center">
        <input
          type="search"
          placeholder="Search filename, tag, language…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="field text-sm w-64"
        />
        <div className="flex gap-1 ml-auto">
          {(['all', 'audio', 'video'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setFilterModality(m)}
              className={cx(
                'px-3 py-1.5 rounded-full text-xs border transition-colors',
                filterModality === m
                  ? 'bg-gray-900 text-white border-gray-900'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-100',
              )}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Scenario filter chips */}
      <div className="border-b border-gray-200 bg-white px-6 py-3 flex flex-wrap items-center gap-2">
        <span className="text-xs text-gray-500 uppercase tracking-wider mr-2">Filter by scenario</span>
        {SCENARIO_PALETTE.map((s) => (
          <ChipToggle
            key={s}
            label={s}
            active={filterScenarios.has(s)}
            onClick={() => toggleFilterScenario(s)}
          />
        ))}
        {filterScenarios.size > 0 && (
          <button
            type="button"
            onClick={() => setFilterScenarios(new Set())}
            className="text-xs text-gray-500 underline ml-2"
          >
            clear
          </button>
        )}
      </div>

      {/* Clip grid */}
      <div className="flex-1 overflow-auto p-6 flex flex-col gap-4">
        {loading && (
          <div className="text-sm text-gray-500 italic">Loading clips…</div>
        )}
        {error && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
            <span className="font-medium">Backend not reachable.</span>{' '}
            <span className="text-xs text-amber-700">{error}</span>
          </div>
        )}
        {!loading && !error && filtered.length === 0 && (
          <div className="text-sm text-gray-500 italic">
            {clips.length === 0
              ? 'No clips yet — record one in Playground or upload via the API.'
              : `No clips match the active filters (${clips.length} total).`}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((c) => (
            <div key={c.id} className="card flex flex-col gap-3">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-gray-900 truncate">
                    {c.original_filename || c.filename}
                  </h3>
                  <p className="text-xs text-gray-500 font-mono truncate">{c.id.slice(0, 12)}…</p>
                </div>
                <ModalityBadge modality={c.modality} />
              </div>

              {/* Audio preview */}
              <audio controls preload="metadata" className="w-full h-9" src={clipAudioUrl(c.id)}>
                <track kind="captions" />
              </audio>

              {/* Auto-extracted metadata grid */}
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <div className="text-gray-500">duration</div>
                <div className="font-mono text-gray-900">{fmtDuration(c.duration_s)}</div>
                <div className="text-gray-500">sample rate</div>
                <div className="font-mono text-gray-900">{c.sample_rate || '—'} Hz</div>
                <div className="text-gray-500">channels</div>
                <div className="font-mono text-gray-900">{c.channels || '—'}</div>
                {c.language_detected && (
                  <>
                    <div className="text-gray-500">language</div>
                    <div className="font-mono text-gray-900">{c.language_detected}</div>
                  </>
                )}
                {c.snr_db != null && (
                  <>
                    <div className="text-gray-500">SNR</div>
                    <div className="font-mono text-gray-900">{c.snr_db.toFixed(1)} dB</div>
                  </>
                )}
                <div className="text-gray-500">created</div>
                <div className="font-mono text-gray-900">{fmtTimestamp(c.created_at)}</div>
              </div>

              {/* Scenario chips (clickable to toggle, ✨ auto-detect button) */}
              <div className="border-t border-gray-100 pt-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-500">
                    Scenarios
                    {autoEvidence[c.id] && (
                      <span
                        className="ml-2 text-[10px] text-gray-400"
                        title={Object.entries(autoEvidence[c.id].evidence)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join('\n') || 'no evidence'}
                      >
                        (auto-detected · hover for evidence)
                      </span>
                    )}
                  </p>
                  <button
                    type="button"
                    onClick={() => void handleAutoTagOne(c, false)}
                    disabled={busyClipId === c.id || autoRunning === 'all'}
                    className="text-[11px] px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200 hover:bg-amber-100 disabled:opacity-50"
                    title="Run heuristic auto-tagger on this clip and merge with existing tags"
                  >
                    {busyClipId === c.id && autoRunning === 'one' ? '⟳' : '✨ auto-detect'}
                  </button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {SCENARIO_PALETTE.map((s) => {
                    const wasAutoDetected = autoEvidence[c.id]?.detected.includes(s)
                    return (
                      <span key={s} className="relative">
                        <ChipToggle
                          label={s}
                          active={c.scenarios.includes(s)}
                          onClick={() => void toggleClipScenario(c, s)}
                          disabled={busyClipId === c.id}
                        />
                        {wasAutoDetected && (
                          <span
                            className="absolute -top-1 -right-1 w-2 h-2 bg-amber-400 rounded-full ring-1 ring-white"
                            title="Auto-detected"
                          />
                        )}
                      </span>
                    )
                  })}
                </div>
                {autoEvidence[c.id] && (
                  <details className="mt-2">
                    <summary className="text-[10px] text-gray-400 cursor-pointer">
                      acoustic features
                    </summary>
                    <pre className="text-[10px] text-gray-600 bg-gray-50 border border-gray-200 rounded p-1.5 mt-1 overflow-x-auto">
{JSON.stringify(autoEvidence[c.id].features, null, 2)}
                    </pre>
                  </details>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 pt-2 border-t border-gray-100">
                <Link
                  to={`/playground?clip=${c.id}`}
                  className="btn-pill-dark text-xs"
                >
                  Run in Playground
                </Link>
                <Link
                  to={`/run?clip=${c.id}`}
                  className="btn-pill-outline text-xs"
                >
                  Pipeline
                </Link>
                <button
                  type="button"
                  onClick={() => void handleDelete(c)}
                  className="btn-pill-danger text-xs ml-auto"
                  disabled={busyClipId === c.id}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        {filtered.length > 0 && (
          <p className="text-xs text-gray-500 text-center mt-2">
            Showing {filtered.length} of {clips.length} clip{clips.length === 1 ? '' : 's'}
          </p>
        )}
      </div>
    </div>
  )
}
