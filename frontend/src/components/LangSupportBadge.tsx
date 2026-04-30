import type { Adapter } from '../lib/api'

export type LangSupport = 'auto' | 'fixed-multi' | 'single'

// Category derived from what the user can actually pick in the language
// dropdown:
//   auto         → list contains "auto"  (adapter exposes auto-detect)
//   fixed-multi  → 2+ codes, no "auto"   (must pick explicitly)
//   single       → 0 or 1 codes
// The `multilang_realtime` flag is preserved on the Adapter type as a
// stricter "true mid-stream LID" signal for docs/metrics, but the UI
// keys off the picker itself so the badge always matches what the
// dropdown shows.
export function langSupport(a: Adapter): LangSupport {
  const langs = a.supported_languages ?? []
  if (langs.length <= 1) return 'single'
  if (langs.includes('auto')) return 'auto'
  return 'fixed-multi'
}

const META: Record<LangSupport, { text: (a: Adapter) => string; cls: string; tip: string }> = {
  auto: {
    text: () => '🌐 Auto multi-lang',
    cls: 'bg-green-50 text-green-700 border-green-200',
    tip: 'Supports auto-detect — pick "auto" and the model handles the language for you. No need to specify up-front.',
  },
  'fixed-multi': {
    text: () => '🔒 Pick lang at start',
    cls: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    tip: 'Supports multiple languages, but you must pick one up-front. Locked at session start.',
  },
  single: {
    text: (a) => {
      const code = (a.supported_languages?.[0] ?? 'en').toUpperCase()
      return `${code} only`
    },
    cls: 'bg-gray-100 text-gray-600 border-gray-200',
    tip: 'Single-language model. Cannot transcribe other languages.',
  },
}

export default function LangSupportBadge({ adapter }: { adapter: Adapter }) {
  const kind = langSupport(adapter)
  const m = META[kind]
  return (
    <span
      title={m.tip}
      className={`px-2 py-0.5 rounded-full text-xs border ${m.cls}`}
    >
      {m.text(adapter)}
    </span>
  )
}
