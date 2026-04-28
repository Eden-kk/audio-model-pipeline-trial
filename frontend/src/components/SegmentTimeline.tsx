import { cx } from '../lib/cx'

export interface Segment {
  start: number
  end: number
  score: number
  is_user: boolean
  embedding_b64?: string
  embedding_dim?: number
}

interface Props {
  segments: Segment[]
  duration_s?: number
  threshold?: number
  /** Optional: highlight a particular t-position (ms) by drawing a vertical line. */
  cursor_s?: number
}

/**
 * Per-segment user-tag timeline for the slow-loop speaker_tag stage.
 *
 * Renders a horizontal bar where each segment is a colored block:
 *   green = is_user (wearer)
 *   gray  = not user (other speakers / silence / ambient)
 * Block opacity tracks the cosine score so low-confidence is_user blocks
 * look slightly washed out.
 *
 * Hover any block → tooltip shows {start, end, score, embedding_dim}.
 */
export default function SegmentTimeline({
  segments,
  duration_s,
  threshold,
  cursor_s,
}: Props) {
  if (!segments || segments.length === 0) {
    return (
      <p className="text-xs text-gray-500 italic">
        No segments — speaker_tag stage didn't run or returned an empty result.
      </p>
    )
  }

  const total = duration_s ?? Math.max(...segments.map((s) => s.end), 1)
  const userCount = segments.filter((s) => s.is_user).length
  const otherCount = segments.length - userCount
  const userPct = (userCount / segments.length) * 100

  return (
    <div className="flex flex-col gap-2">
      {/* Summary row */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <span className="text-gray-700">
            <span className="font-semibold">{segments.length}</span> segments ·
            <span className="ml-1 text-green-700 font-semibold">{userCount} wearer</span> ·
            <span className="ml-1 text-gray-600">{otherCount} other</span>
          </span>
          <span className="text-gray-500 font-mono">
            {userPct.toFixed(0)}% wearer
          </span>
        </div>
        <span className="text-gray-500 font-mono">
          {total.toFixed(1)}s · thr={threshold ?? '?'}
        </span>
      </div>

      {/* Timeline bar */}
      <div
        className="relative w-full h-6 rounded-md bg-gray-100 border border-gray-200 overflow-hidden"
        style={{ minHeight: '1.5rem' }}
      >
        {segments.map((s, i) => {
          const left = (s.start / total) * 100
          const width = ((s.end - s.start) / total) * 100
          // Score 0..1; opacity range 0.4..1 so low-conf still visible
          const opacity = 0.4 + Math.min(Math.max(s.score, 0), 1) * 0.6
          return (
            <div
              key={i}
              className={cx(
                'absolute top-0 bottom-0 transition-opacity',
                s.is_user ? 'bg-green-500' : 'bg-gray-400',
              )}
              style={{
                left: `${left}%`,
                width: `${width}%`,
                opacity,
              }}
              title={
                `t=${s.start.toFixed(2)}-${s.end.toFixed(2)}s\n` +
                `score=${s.score.toFixed(3)} ${s.is_user ? '(wearer)' : '(other)'}` +
                (s.embedding_dim ? `\nemb dim=${s.embedding_dim}` : '')
              }
            />
          )
        })}

        {/* Optional playhead */}
        {cursor_s != null && cursor_s >= 0 && cursor_s <= total && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-red-500 pointer-events-none"
            style={{ left: `${(cursor_s / total) * 100}%` }}
          />
        )}
      </div>

      {/* Time axis */}
      <div className="flex justify-between text-[10px] text-gray-500 font-mono">
        <span>0s</span>
        <span>{(total / 2).toFixed(1)}s</span>
        <span>{total.toFixed(1)}s</span>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-gray-600">
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 bg-green-500 rounded-sm" />
          wearer (is_user=true)
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 bg-gray-400 rounded-sm" />
          other / silence
        </div>
        <span className="ml-auto text-gray-500 italic">
          opacity ∝ score
        </span>
      </div>
    </div>
  )
}
