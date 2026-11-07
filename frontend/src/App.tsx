import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Bot, Database, BrainCircuit, Target, SlidersHorizontal, Activity, ShieldCheck, Settings, X, Sparkles, KeyRound,
  CircleDollarSign, Menu,
} from 'lucide-react'
import { api } from './api'
import type { AppState } from './api'
import { Button, Field, inputCls, ErrorBox } from './components/ui'
import { pct } from './format'
import AgentPage from './pages/Agent'
import DataPage from './pages/Data'
import ModelPage from './pages/Model'
import OptimizerPage from './pages/Optimizer'
import WhatIfPage from './pages/WhatIf'
import MonitoringPage from './pages/Monitoring'
import EvaluationPage from './pages/Evaluation'

type Ctx = { state: AppState | null; refresh: () => Promise<void>; go: (p: Page) => void }
const AppCtx = createContext<Ctx>({ state: null, refresh: async () => {}, go: () => {} })
export const useApp = () => useContext(AppCtx)

export type Page = 'agent' | 'data' | 'model' | 'optimizer' | 'whatif' | 'monitoring' | 'evaluation'
const NAV: { id: Page; label: string; icon: ReactNode; group: string }[] = [
  { id: 'agent', label: 'Agent', icon: <Bot className="h-4 w-4" />, group: 'Assistant' },
  { id: 'data', label: 'Data', icon: <Database className="h-4 w-4" />, group: 'Analyse' },
  { id: 'model', label: 'Channel insights', icon: <BrainCircuit className="h-4 w-4" />, group: 'Analyse' },
  { id: 'optimizer', label: 'Optimizer', icon: <Target className="h-4 w-4" />, group: 'Plan' },
  { id: 'whatif', label: 'What-if', icon: <SlidersHorizontal className="h-4 w-4" />, group: 'Plan' },
  { id: 'monitoring', label: 'Monitoring', icon: <Activity className="h-4 w-4" />, group: 'Operate' },
  { id: 'evaluation', label: 'Evaluation', icon: <ShieldCheck className="h-4 w-4" />, group: 'Operate' },
]

export default function App() {
  const [state, setState] = useState<AppState | null>(null)
  const [page, setPage] = useState<Page>(() => (location.hash.slice(1) as Page) || 'agent')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [navOpen, setNavOpen] = useState(false)
  const [offline, setOffline] = useState(false)

  const refresh = useCallback(async () => {
    try { setState(await api.state()); setOffline(false) } catch { setOffline(true) }
  }, [])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => {
    const onHash = () => { const h = location.hash.slice(1) as Page; if (NAV.some((n) => n.id === h)) setPage(h) }
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])
  const go = (p: Page) => { setPage(p); location.hash = p; setNavOpen(false) }

  const groups = [...new Set(NAV.map((n) => n.group))]
  const Current = { agent: AgentPage, data: DataPage, model: ModelPage, optimizer: OptimizerPage, whatif: WhatIfPage,
    monitoring: MonitoringPage, evaluation: EvaluationPage }[page]

  return (
    <AppCtx.Provider value={{ state, refresh, go }}>
      <div className="flex h-full">
        {/* Sidebar */}
        <aside className={`fixed inset-y-0 left-0 z-40 w-64 shrink-0 bg-slate-950 text-slate-300 transition-transform lg:static lg:translate-x-0 ${navOpen ? 'translate-x-0' : '-translate-x-full'}`}>
          <div className="flex h-16 items-center gap-3 px-5">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-lg shadow-indigo-500/30">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white">Budget Agent</div>
              <div className="text-[11px] text-slate-500">Marketing mix optimizer</div>
            </div>
          </div>
          <nav className="mt-4 space-y-6 px-3">
            {groups.map((g) => (
              <div key={g}>
                <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{g}</div>
                {NAV.filter((n) => n.group === g).map((n) => (
                  <button key={n.id} onClick={() => go(n.id)}
                    className={`mb-0.5 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${page === n.id ? 'bg-white/10 font-medium text-white' : 'hover:bg-white/5 hover:text-white'}`}>
                    <span className={page === n.id ? 'text-indigo-400' : ''}>{n.icon}</span>{n.label}
                  </button>
                ))}
              </div>
            ))}
          </nav>
          <div className="absolute inset-x-0 bottom-0 p-4">
            <button onClick={() => setSettingsOpen(true)} className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm hover:bg-white/5 hover:text-white">
              <Settings className="h-4 w-4" />Settings
            </button>
          </div>
        </aside>
        {navOpen && <div className="fixed inset-0 z-30 bg-slate-950/40 lg:hidden" onClick={() => setNavOpen(false)} />}

        {/* Main */}
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-slate-200/80 bg-white/80 px-4 backdrop-blur sm:px-8">
            <button className="rounded-lg p-2 hover:bg-slate-100 lg:hidden" onClick={() => setNavOpen(true)}><Menu className="h-5 w-5" /></button>
            <div className="flex flex-1 items-center gap-2 overflow-x-auto text-xs [scrollbar-width:none] sm:flex-wrap">
              <Chip ok={!!state?.has_data} label={state?.has_data ? `${state.weeks} weeks · ${state.start} → ${state.end}` : 'No data'} />
              <Chip ok={!!state?.model && !state.model.stale}
                label={state?.model ? `Model R² ${state.model.r2.toFixed(2)}${state.model.stale ? ' · needs refit' : ''}` : 'Model not trained'} />
              <Chip ok={!!state?.gross_margin} label={state?.gross_margin ? `Margin ${pct(state.gross_margin, 0)}` : 'Revenue mode'} neutral />
              <Chip ok={!!state?.openai} label={state?.openai ? `AI: ${state.openai_model}` : 'No OpenAI key'} />
            </div>
            <Button variant="secondary" onClick={() => setSettingsOpen(true)} icon={<Settings className="h-4 w-4" />} className="hidden sm:inline-flex">Settings</Button>
          </header>
          <main className="flex-1 overflow-y-auto px-4 py-8 sm:px-8">
            <div className="mx-auto max-w-7xl fade-in" key={page}>
              {offline ? <ErrorBox error="Cannot reach the API. Start it with: python -m budget_agent.server" /> : state ? <Current /> : null}
            </div>
          </main>
        </div>
      </div>
      {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} />}
    </AppCtx.Provider>
  )
}

function Chip({ ok, label, neutral }: { ok: boolean; label: string; neutral?: boolean }) {
  const dot = ok ? 'bg-emerald-500' : neutral ? 'bg-slate-400' : 'bg-amber-500'
  return <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 font-medium text-slate-600"><span className={`h-1.5 w-1.5 rounded-full ${dot}`} />{label}</span>
}

function SettingsPanel({ onClose }: { onClose: () => void }) {
  const { state, refresh } = useApp()
  const [key, setKey] = useState('')
  const [model, setModel] = useState(state?.openai_model ?? '')
  const [margin, setMargin] = useState(state?.gross_margin ? String(Math.round(state.gross_margin * 100)) : '')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const save = async () => {
    setBusy(true); setErr(null)
    try {
      const m = parseFloat(margin)
      await api.settings({
        openai_api_key: key || undefined, openai_model: model || undefined,
        ...(margin.trim() === '' ? { clear_margin: true } : { gross_margin: m / 100 }),
      })
      await refresh(); onClose()
    } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30 backdrop-blur-[2px]" onClick={onClose}>
      <div className="h-full w-full max-w-md overflow-y-auto bg-white p-6 shadow-2xl fade-in" onClick={(e) => e.stopPropagation()}>
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Settings</h2>
          <button onClick={onClose} className="rounded-lg p-1.5 hover:bg-slate-100"><X className="h-5 w-5" /></button>
        </div>
        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-200 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-900"><KeyRound className="h-4 w-4 text-indigo-600" />OpenAI</div>
            <div className="space-y-3">
              <Field label="API key" hint={state?.openai ? 'A key is set. Leave empty to keep it.' : 'Kept in memory only. To keep it, put OPENAI_API_KEY in .env.'}>
                <input type="password" className={inputCls} placeholder="sk-…" value={key} onChange={(e) => setKey(e.target.value)} />
              </Field>
              <Field label="Model" hint="Any OpenAI model that supports tool calling.">
                <input className={inputCls} value={model} onChange={(e) => setModel(e.target.value)} />
              </Field>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-900"><CircleDollarSign className="h-4 w-4 text-emerald-600" />Profit</div>
            <Field label="Gross margin (%)" hint="Share of each sale kept after product costs. Leave empty if unknown: results stay in revenue.">
              <input type="number" min={1} max={100} className={inputCls} placeholder="e.g. 55" value={margin} onChange={(e) => setMargin(e.target.value)} />
            </Field>
            {state?.gross_margin ? <p className="mt-2 text-xs text-slate-500">Break-even: ${state.breakeven.toFixed(2)} of sales per $1 of ads.</p> : null}
          </div>
          <ErrorBox error={err} />
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={onClose}>Cancel</Button>
            <Button onClick={save} loading={busy}>Save</Button>
          </div>
        </div>
      </div>
    </div>
  )
}
