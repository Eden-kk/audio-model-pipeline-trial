// ---------------------------------------------------------------------------
// API client — typed wrapper around fetch + WebSocket
// Base URL from VITE_API_URL. Falls back to same-origin in production
// builds so the SPA works on whatever host:port serves it (no rebuild
// needed when the public origin changes). Dev (`vite`) keeps the
// localhost:8000 default for the split frontend/backend workflow.
// ---------------------------------------------------------------------------

const BASE = (import.meta.env.VITE_API_URL as string | undefined)
  ?? (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin)

// ---------------------------------------------------------------------------
// Shared types (mirrors backend data model)
// ---------------------------------------------------------------------------

export type ModelCategory =
  | 'vad'
  | 'asr'
  | 'speaker_verify'
  | 'diarization'
  | 'tts'
  | 'intent_llm'
  | 'realtime_omni'

export interface Adapter {
  id: string
  category: ModelCategory
  display_name: string
  hosting: 'cloud' | 'modal' | 'edge'
  vendor: string
  inputs: { name: string; type: string }[]
  outputs: { name: string; type: string }[]
  config_schema: Record<string, unknown>
  cost_per_call_estimate_usd: number | null
  /** True if the adapter implements transcribe_stream() and the runner
   *  emits stage.progress events as partial transcripts arrive. The
   *  Playground renders a typing cursor for these; batch adapters get
   *  a single-shot final transcript. */
  is_streaming?: boolean
}

export interface Clip {
  id: string
  source: 'record' | 'upload' | 'live-mic'
  modality: 'audio' | 'video'
  filename: string
  format: string
  original_filename: string
  original_format: string
  duration_s: number
  sample_rate: number
  channels: number
  language_detected: string | null
  snr_db: number | null
  speaker_count_estimate: number | null
  user_tags: string[]
  scenarios: string[]
  uploaded_by: string
  created_at: string
  /** Plan D A2 — populated for clips captured from /ws/mic with ?save=1.
   *  The vendor's streaming transcript is stored as a ground-truth seed
   *  for the AR-glass benchmark. null/empty for upload + record blobs. */
  captured_transcript?: string | null
  captured_transcript_segments?: Array<{
    start: number
    end: number
    text: string
    is_final: boolean
  }>
}

/**
 * Mirrors backend.main.RunOut. The synchronous run response carries the full
 * adapter result inside `result`; WS events are only emitted for streaming
 * runs (Slice 3+). Always prefer this response as the source of truth.
 */
export interface Run {
  id: string
  clip_id: string
  adapter: string
  config: Record<string, unknown>
  started_at: string
  finished_at: string | null
  latency_ms: number | null
  cost_usd: number | null
  input_preview: string
  output_preview: string
  result: {
    text?: string
    words?: Array<{ word: string; start: number; end: number; confidence?: number; speaker?: string | null }>
    language?: string
    duration_s?: number
    cost_usd?: number
    wall_time_s?: number
    raw_response?: unknown
    [k: string]: unknown
  }
  error: string | null
}

// WebSocket event shapes — backend emits the legacy event names
// (StageStarted / StageProgress / StagePartial / StageCompleted /
// StageFailed); the runner currently uses the `event` key.  Keep the
// new `type` field too for forward-compat once the runner migrates.
export type StageEventName =
  | 'StageStarted'
  | 'StageProgress'   // streaming-only — partial transcripts
  | 'StagePartial'    // batch-shaped final preview
  | 'StageCompleted'
  | 'StageFailed'
  | 'keepalive'
  | 'pong'

export interface RunEvent {
  event?: StageEventName
  type?: StageEventName     // forward-compat
  run_id?: string
  adapter?: string
  partial_text?: string
  partial_index?: number
  text?: string
  latency_ms?: number
  cost_usd?: number
  result?: Record<string, unknown>
  error?: string
  is_streaming?: boolean
  timestamp?: string
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

/** List all adapters, optionally filter by category */
export async function listAdapters(category?: ModelCategory): Promise<Adapter[]> {
  const url = category ? `/api/adapters?category=${category}` : '/api/adapters'
  return apiFetch<Adapter[]>(url)
}

// ---------------------------------------------------------------------------
// Clips
// ---------------------------------------------------------------------------

/** Audio URL for a stored clip — the canonical extracted audio. */
export function clipAudioUrl(clipId: string): string {
  return `${BASE}/api/clips/${clipId}/audio`
}

/** GET /api/clips — full clip library, freshest first server-side. */
export async function listClips(): Promise<Clip[]> {
  const res = await apiFetch<{ clips: Clip[] }>('/api/clips')
  return res.clips
}

/** PATCH /api/clips/{id} — update user_tags or scenarios. */
export async function updateClipTags(
  clipId: string,
  patch: { user_tags?: string[]; scenarios?: string[] },
): Promise<Clip> {
  return apiFetch<Clip>(`/api/clips/${clipId}`, {
    method: 'PATCH',
    body: JSON.stringify(patch),
  })
}

/** DELETE /api/clips/{id} — purge clip + manifest from disk. */
export async function deleteClip(clipId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/clips/${clipId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete ${res.status}`)
}

// ─── Auto-tagger ──────────────────────────────────────────────────────────

export interface AutoTagOptions {
  use_lid?: boolean
  use_speaker_spread?: boolean
  /** If true, replace existing scenarios entirely; otherwise union with
   *  what the user already set. Default false (union — never clobber). */
  replace?: boolean
}

export interface AutoTagResult {
  clip_id: string
  detected: string[]
  final_scenarios: string[]
  features: Record<string, unknown>
  evidence: Record<string, string>
}

export interface AutoTagAllResult {
  total: number
  succeeded: number
  failed: number
  results: AutoTagResult[]
  errors: Record<string, string>
}

/** POST /api/clips/{id}/autotag — heuristic scenario detection for one clip. */
export async function autotagClip(
  clipId: string,
  opts: AutoTagOptions = {},
): Promise<AutoTagResult> {
  return apiFetch<AutoTagResult>(`/api/clips/${clipId}/autotag`, {
    method: 'POST',
    body: JSON.stringify(opts),
  })
}

/** POST /api/clips/autotag-all — run the auto-tagger over the entire corpus. */
export async function autotagAllClips(opts: AutoTagOptions = {}): Promise<AutoTagAllResult> {
  return apiFetch<AutoTagAllResult>('/api/clips/autotag-all', {
    method: 'POST',
    body: JSON.stringify(opts),
  })
}

/** Upload a raw audio Blob. Returns the created Clip. */
export async function uploadClip(blob: Blob, mime: string): Promise<Clip> {
  const form = new FormData()
  const ext = mime.includes('wav') ? 'wav' : mime.includes('ogg') ? 'ogg' : 'webm'
  form.append('file', blob, `recording.${ext}`)
  form.append('source', 'record')

  const res = await fetch(`${BASE}/api/clips`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`Upload ${res.status}: ${text}`)
  }
  return res.json() as Promise<Clip>
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export interface StartRunPayload {
  clip_id: string
  adapter: string
  config?: Record<string, unknown>
}

/** POST /api/runs — create a new run */
export async function startRun(
  clipId: string,
  adapter: string,
  config: Record<string, unknown> = {},
): Promise<Run> {
  return apiFetch<Run>('/api/runs', {
    method: 'POST',
    body: JSON.stringify({ clip_id: clipId, adapter, config }),
  })
}

/** GET /api/runs/{id} — fetch a saved run record. Used as a defensive
 *  fallback after a streaming run's StageCompleted, in case the WS event
 *  arrived without a full result.text (race vs server-side buffer). */
export async function getRun(runId: string): Promise<Run> {
  return apiFetch<Run>(`/api/runs/${runId}`)
}

// ---------------------------------------------------------------------------
// Enrollment (Slice 9.1e + 9.4)
// ---------------------------------------------------------------------------

export interface Enrollment {
  profile_id: string
  adapter: string
  embedding_dim: number
  saved_to: string
  enrolled_at: string
}

export interface EnrollResult {
  profile_id: string
  adapter: string
  embedding_dim: number
  embedding_dtype: string
  duration_s: number | null
  saved_to: string
}

/** POST /api/enroll — upload a reference clip → save embedding to disk. */
export async function enrollWearer(
  blob: Blob,
  mime: string,
  opts: { adapter?: string; profile_id?: string } = {},
): Promise<EnrollResult> {
  const form = new FormData()
  const ext = mime.includes('wav') ? 'wav' : mime.includes('webm') ? 'webm'
            : mime.includes('ogg') ? 'ogg' : 'wav'
  form.append('file', blob, `enroll.${ext}`)
  form.append('adapter', opts.adapter ?? 'pyannote_verify')
  form.append('profile_id', opts.profile_id ?? 'wearer')
  const res = await fetch(`${BASE}/api/enroll`, { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`Enroll ${res.status}: ${text}`)
  }
  return res.json() as Promise<EnrollResult>
}

export async function listEnrollments(): Promise<Enrollment[]> {
  const r = await apiFetch<{ enrollments: Enrollment[] }>('/api/enroll')
  return r.enrollments
}

export async function deleteEnrollment(profileId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/enroll/${profileId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete ${res.status}`)
}

// ---------------------------------------------------------------------------
// Settings — read-only env status
// ---------------------------------------------------------------------------

export interface BackendSettings {
  api_keys: Record<string, 'set' | 'unset'>
  service_urls: Record<string, { value: string | null; configured: boolean }>
  intent_llm: {
    url_configured: boolean
    default_model: string
    key_configured: boolean
  }
  realtime_omni?: {
    minicpm_o: {
      url_configured: boolean
      url: string
    }
    gemini_live: {
      vertex_adc_configured: boolean
      vertex_project: string
      vertex_location: string
      api_key_configured: boolean
      status_note: string
    }
  }
}

export async function getSettings(): Promise<BackendSettings> {
  return apiFetch<BackendSettings>('/api/settings')
}

// ---------------------------------------------------------------------------
// Recipes (multi-stage pipelines)
// ---------------------------------------------------------------------------

export interface RecipeStage {
  id: string
  category: ModelCategory
  adapter: string | null   // null = placeholder, user picks per-stage
  config: Record<string, unknown>
}

export interface Recipe {
  id: string
  name: string
  description: string
  is_recipe: true
  stages: RecipeStage[]
  edges: { from: string; to: string; port: string }[]
}

export interface StageRun {
  stage_id: string
  category: ModelCategory
  adapter: string | null
  started_at: string
  finished_at: string
  latency_ms: number
  cost_usd: number
  input_preview: string
  output_preview: string
  result: Record<string, unknown>
  error: string | null
}

export interface RecipeRun {
  id: string
  clip_id: string
  recipe_id: string
  started_at: string
  finished_at: string
  stages: StageRun[]
  total_latency_ms: number
  total_cost_usd: number
  error: string | null
}

export async function listRecipes(): Promise<Recipe[]> {
  return apiFetch<Recipe[]>('/api/recipes')
}

export async function startRecipeRun(
  clipId: string,
  recipeId: string,
  stageAdapters: Record<string, string>,
  stageConfigs: Record<string, Record<string, unknown>> = {},
): Promise<RecipeRun> {
  return apiFetch<RecipeRun>('/api/runs/recipe', {
    method: 'POST',
    body: JSON.stringify({
      clip_id: clipId,
      recipe_id: recipeId,
      stage_adapters: stageAdapters,
      stage_configs: stageConfigs,
    }),
  })
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

/** Open a WebSocket to /ws/run/{runId} and call onEvent for each message.
 *  Returns a cleanup function that closes the socket. */
export function connectRunWS(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onClose?: () => void,
  onError?: (err: Event) => void,
): () => void {
  const wsBase = BASE.replace(/^http/, 'ws')
  const ws = new WebSocket(`${wsBase}/ws/run/${runId}`)

  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data as string) as RunEvent
      onEvent(event)
    } catch {
      // ignore malformed frames
    }
  }

  ws.onclose = () => onClose?.()
  ws.onerror = (err) => onError?.(err)

  return () => ws.close()
}
