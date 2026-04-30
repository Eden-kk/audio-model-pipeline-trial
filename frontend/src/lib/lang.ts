/** Human-readable BCP-47 language name with code suffix.
 *  "es" → "Spanish (es)" · "auto" → "Auto-detect" · unknown code → the code itself. */
let displayNames: Intl.DisplayNames | null = null
try { displayNames = new Intl.DisplayNames(['en'], { type: 'language' }) }
catch { /* very old browser; fall back to code-only */ }

export function formatLanguage(code: string): string {
  if (code === 'auto') return 'Auto-detect'
  const name = displayNames?.of(code)
  return name && name !== code ? `${name} (${code})` : code
}
