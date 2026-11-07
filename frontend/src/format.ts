export const money = (v: number | null | undefined, digits = 0) =>
  v == null ? '—' : (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })

export const compact = (v: number | null | undefined) =>
  v == null ? '—' : (v < 0 ? '-$' : '$') + Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(Math.abs(v))

export const pct = (v: number | null | undefined, digits = 1, sign = false) =>
  v == null ? '—' : `${sign && v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`

export const num = (v: number | null | undefined, digits = 2) => (v == null ? '—' : v.toFixed(digits))

export const label = (ch: string) =>
  ch.replace(/^meta_/, '').replace(/^google_/, 'google ').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
    .replace('Pmax', 'PMax').replace(/^Tv$/, 'TV')

const PALETTE = ['#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6']
export const colorFor = (ch: string, all: string[]) => PALETTE[Math.max(all.indexOf(ch), 0) % PALETTE.length]
