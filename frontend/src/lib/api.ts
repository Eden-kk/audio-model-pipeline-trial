// ---------------------------------------------------------------------------
// API client — typed wrapper around fetch + WebSocket
// Base URL from VITE_API_URL (default http://localhost:8000)
// ---------------------------------------------------------------------------

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

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
  source: 'record' | 'upload'
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

// WebSocket event shapes
export type RunEventType =
  | 'run.started'
  | 'stage.started'
  | 'stage.progress'
  | 'stage.finished'
  | 'run.finished'
  | 'run.error'

export interface RunEvent {
  type: RunEventType
  run_id: string
  stage_id?: string
  data: Record<string, unknown>
  ts: number
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
