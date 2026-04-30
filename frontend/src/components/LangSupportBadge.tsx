import type { Adapter } from '../lib/api'

export type LangSupport = 'realtime' | 'fixed-multi' | 'single'

export function langSupport(a: Adapter): LangSupport {
  if (a.multilang_realtime) return 'realtime'
  if (a.multilang) return 'fixed-multi'
  return 'single'
}

const META: Record<LangSupport, { text: (a: Adapter) => string; cls: string; tip: string }> = {
  realtime: {
    text: () => '🌐 Auto multi-lang',
    cls: 'bg-green-50 text-green-700 border-green-200',
    tip: 'Detects language switches during streaming. No need to pick a language up-front.',
  },
  'fixed-multi': {
    text: () => '🔒 Pick lang at start',
    cls: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    tip: 'Supports multiple languages, but you must choose one before starting. Cannot switch mid-session.',
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
