/* eslint-disable @typescript-eslint/no-explicit-any */
export type AppState = {
  has_data: boolean; weeks: number; start: string | null; end: string | null; channels: string[]
  model: null | { trained_weeks: number; r2: number; mape: number; stale: boolean }
  gross_margin: number | null; breakeven: number; openai: boolean; openai_model: string
  demo_weeks_remaining: number; scenarios: string[]; last_goal: any; real_data_available: boolean
}

async function req<T = any>(path: string, body?: unknown, method?: string): Promise<T> {
  const opts: RequestInit = { method: method ?? (body === undefined ? 'GET' : 'POST') }
  if (body instanceof FormData) opts.body = body
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers = { 'Content-Type': 'application/json' } }
  const r = await fetch(`/api${path}`, opts)
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try { const j = await r.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* ignore */ }
    throw new Error(msg)
  }
  return r.json()
}

export const api = {
  state: () => req<AppState>('/state'),
  settings: (b: any) => req<AppState>('/settings', b),
  demo: (seed = 7) => req<AppState>('/data/demo', { seed }),
  real: () => req<AppState>('/data/real', {}),
  upload: (f: File) => { const fd = new FormData(); fd.append('file', f); return req<AppState>('/data/upload', fd) },
  data: () => req('/data'),
  fit: (b: any = {}) => req('/model/fit', b),
  model: () => req('/model'),
  optimize: (b: any) => req('/optimize', b),
  whatif: (b: any) => req('/whatif', b),
  scenarios: () => req('/scenarios'),
  plan: (scenario = 'recommended') => req('/plan', { scenario }),
  plans: () => req<any[]>('/plans'),
  ingest: (demo_weeks: number) => req('/ingest', { demo_weeks }),
  ingestUpload: (f: File) => { const fd = new FormData(); fd.append('file', f); return req('/ingest/upload', fd) },
  alerts: () => req('/alerts', {}),
  history: () => req('/history'),
  evaluate: (budget: number) => req('/evaluate', { budget }),
  autopilot: (total_budget: number, n_weeks: number) => req('/autopilot', { total_budget, n_weeks }),
  chatReset: () => req('/chat/reset', {}),
  async chat(message: string, onEvent: (e: any) => void) {
    const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message }) })
    if (!r.ok || !r.body) {
      let msg = `${r.status}`
      try { msg = (await r.json()).detail } catch { /* ignore */ }
      throw new Error(msg)
    }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let i
      while ((i = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 1); if (line.trim()) onEvent(JSON.parse(line)) }
    }
  },
}

export const fileUrl = (name: string) => `/api/plans/${encodeURIComponent(name)}`
