import { useEffect, useState, useCallback } from 'react'
import MicRecorder from '../components/MicRecorder'
import AudioFileDrop from '../components/AudioFileDrop'
import {
  enrollWearer, listEnrollments, deleteEnrollment, getSettings,
  listAdapters,
  type Enrollment, type EnrollResult, type BackendSettings, type Adapter,
} from '../lib/api'
import { cx } from '../lib/cx'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function StatusPill({ ok, label, hint }: { ok: boolean; label: string; hint?: string }) {
  return (
    <span
      title={hint}
      className={cx(
        'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs border',
        ok
          ? 'bg-green-100 text-green-800 border-green-200'
          : 'bg-gray-100 text-gray-600 border-gray-200',
      )}
    >
      <span className={cx('w-1.5 h-1.5 rounded-full',
                           ok ? 'bg-green-600' : 'bg-gray-400')} />
      {label}
    </span>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Settings() {
  // Wearer enrollment
  const [enrollments, setEnrollments] = useState<Enrollment[]>([])
  const [enrollAdapter, setEnrollAdapter] = useState<string>('pyannote_verify')
  const [enrollProfileId, setEnrollProfileId] = useState<string>('wearer')
  const [pendingBlob, setPendingBlob] = useState<{ blob: Blob; mime: string } | null>(null)
  const [enrolling, setEnrolling] = useState(false)
  const [enrollResult, setEnrollResult] = useState<EnrollResult | null>(null)
  const [enrollError, setEnrollError] = useState<string | null>(null)
  const [inputMode, setInputMode] = useState<'record' | 'upload'>('record')

  const [verifyAdapters, setVerifyAdapters] = useState<Adapter[]>([])

  // Settings status
  const [settings, setSettings] = useState<BackendSettings | null>(null)
  const [settingsErr, setSettingsErr] = useState<string | null>(null)

  function refresh() {
    listEnrollments().then(setEnrollments).catch(() => setEnrollments([]))
    getSettings().then(setSettings).catch(
      (e) => setSettingsErr(e instanceof Error ? e.message : String(e)),
    )
    listAdapters('speaker_verify').then(setVerifyAdapters).catch(() => setVerifyAdapters([]))
  }
  useEffect(() => { refresh() }, [])

  const handleBlob = useCallback((blob: Blob, mime: string) => {
    setPendingBlob({ blob, mime })
    setEnrollResult(null); setEnrollError(null)
  }, [])

  async function handleEnroll() {
    if (!pendingBlob) return
    setEnrolling(true); setEnrollError(null); setEnrollResult(null)
    try {
      const r = await enrollWearer(pendingBlob.blob, pendingBlob.mime, {
        adapter: enrollAdapter, profile_id: enrollProfileId,
      })
      setEnrollResult(r)
      refresh()
    } catch (e: unknown) {
      setEnrollError(e instanceof Error ? e.message : String(e))
    } finally {
      setEnrolling(false)
    }
  }

  async function handleDelete(profile_id: string) {
    if (!confirm(`Delete enrolled profile "${profile_id}"?`)) return
    await deleteEnrollment(profile_id)
    refresh()
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Wearer enrollment + read-only environment status. Edit{' '}
          <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">backend/.env</code>{' '}
          and restart uvicorn to change keys / URLs.
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ─── Wearer enrollment ────────────────────────────────────────── */}
        <div className="card flex flex-col gap-4">
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              🎙 Enroll wearer voice
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Record ~10 s of speech to learn your voice. The slow-loop
              pipeline's <span className="font-mono">speaker_tag</span> stage
              auto-loads this embedding on every run.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500">Adapter</span>
              <select
                value={enrollAdapter}
                onChange={(e) => setEnrollAdapter(e.target.value)}
                disabled={enrolling}
                className="field text-sm"
              >
                {verifyAdapters.length === 0 && (
                  <>
                    <option value="pyannote_verify">pyannote_verify</option>
                    <option value="resemblyzer">resemblyzer</option>
                  </>
                )}
                {verifyAdapters.map((a) => (
                  <option key={a.id} value={a.id}>{a.display_name}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500">Profile id</span>
              <input
                type="text"
                value={enrollProfileId}
                onChange={(e) => setEnrollProfileId(e.target.value)}
                disabled={enrolling}
                className="field text-sm font-mono"
              />
            </label>
          </div>

          {/* Source selector */}
          <div className="flex gap-2">
            {(['record', 'upload'] as const).map((m) => (
              <button key={m} type="button"
                disabled={enrolling}
                onClick={() => setInputMode(m)}
                className={cx(
                  inputMode === m ? 'btn-pill-dark' : 'btn-pill-outline',
                  'text-xs',
                )}>
                {m === 'record' ? 'Mic record' : 'File upload'}
              </button>
            ))}
          </div>

          {inputMode === 'record'
            ? <MicRecorder onBlob={handleBlob} disabled={enrolling} />
            : <AudioFileDrop onBlob={handleBlob} disabled={enrolling} />}

          {pendingBlob && (
            <p className="text-xs text-green-700">
              Audio ready — {(pendingBlob.blob.size / 1024).toFixed(1)} KB
            </p>
          )}

          <button
            type="button"
            onClick={() => void handleEnroll()}
            disabled={!pendingBlob || enrolling || !enrollProfileId.trim()}
            className="btn-pill-dark"
          >
            {enrolling ? 'Enrolling…' : 'Enroll'}
          </button>

          {enrollError && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-700">
              {enrollError}
            </div>
          )}
          {enrollResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-xs text-green-800">
              ✓ Enrolled <span className="font-mono">{enrollResult.profile_id}</span>{' '}
              — {enrollResult.embedding_dim}-dim {enrollResult.embedding_dtype}
              {enrollResult.duration_s != null && (
                <> from {enrollResult.duration_s.toFixed(1)} s of audio</>
              )}
              <br /><span className="text-gray-500">{enrollResult.saved_to}</span>
            </div>
          )}

          {/* Existing enrollments */}
          <div className="border-t border-gray-100 pt-3">
            <p className="text-xs text-gray-500 mb-2">Saved profiles</p>
            {enrollments.length === 0 ? (
              <p className="text-xs text-gray-500 italic">None yet.</p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {enrollments.map((e) => (
                  <li key={e.profile_id} className="flex items-center gap-2 text-xs">
                    <span className="font-mono text-gray-900 font-semibold">{e.profile_id}</span>
                    <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-700 text-[10px]">
                      {e.adapter}
                    </span>
                    <span className="text-gray-500">{e.embedding_dim}-d</span>
                    <span className="text-gray-400 ml-auto font-mono">
                      {e.enrolled_at.slice(0, 10)}
                    </span>
                    <button
                      type="button"
                      onClick={() => void handleDelete(e.profile_id)}
                      className="text-red-600 hover:text-red-700 underline"
                    >
                      delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* ─── Intent LLM status + how-to ──────────────────────────────── */}
        <div className="card flex flex-col gap-3">
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              🧠 Intent LLM
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              The slow-loop's <span className="font-mono">intent</span> stage hits
              an OpenAI-compatible <span className="font-mono">/v1/chat/completions</span>{' '}
              endpoint to produce the memory + tool-call envelope.
            </p>
          </div>

          {settings ? (
            <div className="flex items-center gap-2 flex-wrap">
              <StatusPill
                ok={settings.intent_llm.url_configured}
                label={settings.intent_llm.url_configured ? 'INTENT_LLM_URL set' : 'INTENT_LLM_URL not set'}
              />
              <StatusPill
                ok={settings.intent_llm.key_configured}
                label={settings.intent_llm.key_configured ? 'INTENT_LLM_KEY set' : 'no key (open endpoint)'}
              />
              <span className="text-xs text-gray-500 font-mono ml-auto">
                model: {settings.intent_llm.default_model}
              </span>
            </div>
          ) : settingsErr ? (
            <p className="text-xs text-red-600">{settingsErr}</p>
          ) : (
            <p className="text-xs text-gray-500 italic">Loading…</p>
          )}

          <div className="border-t border-gray-100 pt-3">
            <p className="text-xs text-gray-500 mb-2">Quick-start options (pick one)</p>

            <details open>
              <summary className="text-sm font-medium text-gray-900 cursor-pointer hover:text-gray-700">
                Option A — Qwen-2.5-7B on Modal (recommended, zero per-call cost)
              </summary>
              <pre className="mt-2 text-[11px] text-gray-700 bg-gray-50 border border-gray-200 rounded p-3 overflow-auto">
{`# ambient-deploy's exp06-qwen25-openai-server (vLLM Qwen-2.5-7B on A100).
# Validated end-to-end: 376 in / 219 out tokens on a 6-segment slow-loop
# call; cold-start ~45s, warm calls 1-3s.
INTENT_LLM_URL=https://hao-ai-lab--exp06-qwen25-openai-server-server-fastapi-app.modal.run/v1
INTENT_LLM_KEY=sk-dummy-ambient-deploy
INTENT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# then restart uvicorn`}
              </pre>
            </details>

            <details className="mt-2">
              <summary className="text-sm font-medium text-gray-900 cursor-pointer hover:text-gray-700">
                Option B — Groq (fast, free tier — Llama-3.3, NOT Qwen)
              </summary>
              <pre className="mt-2 text-[11px] text-gray-700 bg-gray-50 border border-gray-200 rounded p-3 overflow-auto">
{`# Uses your existing GROQ_API_KEY. Groq doesn't host Qwen models, so
# this is the fastest non-Qwen alternative.
INTENT_LLM_URL=https://api.groq.com/openai/v1
INTENT_LLM_KEY=$GROQ_API_KEY
INTENT_LLM_MODEL=llama-3.3-70b-versatile`}
              </pre>
            </details>

            <details className="mt-2">
              <summary className="text-sm font-medium text-gray-900 cursor-pointer hover:text-gray-700">
                Option C — OpenAI directly
              </summary>
              <pre className="mt-2 text-[11px] text-gray-700 bg-gray-50 border border-gray-200 rounded p-3 overflow-auto">
{`INTENT_LLM_URL=https://api.openai.com/v1
INTENT_LLM_KEY=sk-...
INTENT_LLM_MODEL=gpt-4o-mini`}
              </pre>
            </details>

            <p className="text-[11px] text-gray-500 italic mt-3">
              The QwenIntentAdapter sends a JSON-mode chat completion with the
              slow-loop envelope schema in the system prompt. Any OpenAI-compatible
              endpoint that supports{' '}
              <code className="font-mono">response_format=json_object</code> (Groq,
              OpenAI, vLLM ≥0.5, llama.cpp server) will work.
            </p>
          </div>
        </div>

        {/* ─── API key status (full grid) ─────────────────────────────── */}
        {settings && (
          <div className="card lg:col-span-2 flex flex-col gap-3">
            <div>
              <h2 className="text-base font-semibold text-gray-900">🔑 Backend env</h2>
              <p className="text-xs text-gray-500 mt-0.5">
                Read-only. Edit <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">backend/.env</code> + restart uvicorn to change.
              </p>
            </div>

            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">API keys</p>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(settings.api_keys).map(([k, v]) => (
                  <StatusPill key={k} ok={v === 'set'} label={k} />
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Service URLs</p>
              <div className="flex flex-col gap-1">
                {Object.entries(settings.service_urls).map(([k, v]) => (
                  <div key={k} className="flex items-center gap-2 text-xs">
                    <span className="font-mono text-gray-700 w-44">{k}</span>
                    {v.configured
                      ? <span className="font-mono text-gray-900 truncate">{v.value}</span>
                      : <span className="text-gray-400 italic">unset</span>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
