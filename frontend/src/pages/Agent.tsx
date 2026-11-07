/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import {
  ArrowUp, Bot, CheckCircle2, ChevronRight, CircleAlert, Download, FileText, ListChecks, Loader2, RotateCcw,
  ShieldCheck, Sparkles, User, Wand2, Wrench,
} from 'lucide-react'
import { useApp } from '../App'
import { api, fileUrl } from '../api'
import { Badge, Button, Card, ErrorBox, Field, inputCls } from '../components/ui'

type Step = { tool: string; args: any; seconds: number; error?: string | null; result: any }
type Msg = {
  role: 'user' | 'assistant'; content: string; steps?: Step[]; plan?: { goal: string; steps: string[] } | null
  grounding?: { grounded: string[]; ungrounded: string[]; numbers_checked: number; passed: boolean } | null
  revised?: boolean; pending?: boolean; error?: string; log?: string[]
}

const SUGGESTIONS = [
  'I have $100K for Q1. How should I spend it?',
  'What if I cut TV by 30%?',
  'Which channels are saturated? Alert the team.',
  'New weekly results are in: ingest 4 demo weeks and tell me what changed.',
  'Plan $250K over 26 weeks, at least $40K on TV, no channel changing more than 40%.',
]

const TOOL_LABEL: Record<string, string> = {
  make_plan: 'Writing a plan', check_data: 'Checking data quality', fit_model: 'Training the model',
  get_channel_performance: 'Reading channel performance', optimize_budget: 'Optimizing the budget',
  run_what_if: 'Testing a scenario', compare_scenarios: 'Comparing scenarios', check_saturation_alerts: 'Checking saturation & sending alerts',
  generate_budget_plan: 'Writing the budget plan file', ingest_new_data: 'Learning from new data', get_history: 'Reading memory',
  set_gross_margin: 'Setting the margin',
}

export default function AgentPage() {
  const { state, refresh, go } = useApp()
  const [msgs, setMsgs] = useState<Msg[]>(() => { try { return JSON.parse(sessionStorage.getItem('chat') || '[]') } catch { return [] } })
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => { try { sessionStorage.setItem('chat', JSON.stringify(msgs.filter((m) => !m.pending))) } catch { /* ignore */ } }, [msgs])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const patchLast = (f: (m: Msg) => Msg) => setMsgs((ms) => [...ms.slice(0, -1), f(ms[ms.length - 1])])

  const send = async (text: string) => {
    if (!text.trim() || busy) return
    setInput(''); setBusy(true)
    setMsgs((ms) => [...ms, { role: 'user', content: text }, { role: 'assistant', content: '', steps: [], pending: true }])
    try {
      await api.chat(text, (ev) => {
        if (ev.type === 'step') {
          patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), ev], plan: ev.tool === 'make_plan' ? { goal: ev.args.goal, steps: ev.args.steps } : m.plan }))
        } else if (ev.type === 'answer') {
          patchLast((m) => ({ ...m, content: ev.answer, plan: ev.plan ?? m.plan, grounding: ev.grounding, revised: ev.revised, pending: false }))
        } else if (ev.type === 'error') {
          patchLast((m) => ({ ...m, pending: false, error: ev.error }))
        }
      })
    } catch (e) {
      patchLast((m) => ({ ...m, pending: false, error: (e as Error).message }))
    } finally { setBusy(false); refresh() }
  }

  const reset = async () => { setMsgs([]); try { await api.chatReset() } catch { /* ignore */ } }

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex min-h-[calc(100vh-10rem)] min-w-0 flex-col rounded-2xl border border-slate-200/80 bg-white">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-50 text-indigo-600"><Bot className="h-5 w-5" /></div>
            <div>
              <div className="text-sm font-semibold text-slate-900">Budget Agent</div>
              <div className="text-xs text-slate-500">Plans, runs the model, acts, and checks its own numbers</div>
            </div>
          </div>
          {msgs.length > 0 && <Button variant="ghost" onClick={reset} icon={<RotateCcw className="h-4 w-4" />}>New chat</Button>}
        </div>

        <div className="flex-1 space-y-6 overflow-y-auto px-5 py-6">
          {!state?.openai && (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <span>Add your OpenAI key to chat. The Autopilot on the right works without it.</span>
            </div>
          )}
          {msgs.length === 0 ? (
            <div className="mx-auto max-w-xl py-10 text-center">
              <div className="mx-auto mb-5 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-lg shadow-indigo-500/30"><Sparkles className="h-7 w-7" /></div>
              <h2 className="text-xl font-semibold text-slate-900">Where should your next marketing dollar go?</h2>
              <p className="mt-2 text-sm text-slate-500">Ask in plain language. The agent plans its steps, runs the model and optimizer, and explains every recommendation.</p>
              {!state?.has_data && <p className="mt-3 text-sm text-slate-500">Tip: <button className="font-medium text-indigo-600 hover:underline" onClick={() => go('data')}>load data</button> first.</p>}
              <div className="mt-6 grid gap-2 text-left">
                {SUGGESTIONS.map((s) => (
                  <button key={s} disabled={!state?.openai || busy} onClick={() => send(s)}
                    className="group flex items-center justify-between rounded-xl border border-slate-200 px-4 py-3 text-sm text-slate-700 transition hover:border-indigo-300 hover:bg-indigo-50/50 disabled:opacity-50">
                    {s}<ChevronRight className="h-4 w-4 text-slate-400 group-hover:text-indigo-500" />
                  </button>
                ))}
              </div>
            </div>
          ) : msgs.map((m, i) => <Message key={i} m={m} />)}
          <div ref={endRef} />
        </div>

        <form onSubmit={(e) => { e.preventDefault(); send(input) }} className="border-t border-slate-100 p-4">
          <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-2 focus-within:border-indigo-300 focus-within:ring-4 focus-within:ring-indigo-100">
            <textarea rows={1} value={input} onChange={(e) => setInput(e.target.value)} disabled={!state?.openai}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
              placeholder={state?.openai ? 'e.g. I have $100K for Q1, how should I spend it?' : 'Add an OpenAI key in Settings to chat'}
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-none placeholder:text-slate-400" />
            <button type="submit" disabled={!input.trim() || busy || !state?.openai}
              className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-500 disabled:opacity-40">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
            </button>
          </div>
        </form>
      </div>

      <div className="space-y-6">
        <AutopilotCard onLog={(log) => setMsgs((ms) => [...ms, { role: 'assistant', content: '', log }])} />
        <PlansCard />
      </div>
    </div>
  )
}

function Message({ m }: { m: Msg }) {
  if (m.role === 'user') {
    return (
      <div className="flex justify-end gap-3 fade-in">
        <div className="max-w-[80%] rounded-2xl rounded-tr-md bg-indigo-600 px-4 py-2.5 text-sm text-white shadow-sm">{m.content}</div>
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-slate-200 text-slate-600"><User className="h-4 w-4" /></div>
      </div>
    )
  }
  return (
    <div className="flex gap-3 fade-in">
      <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-indigo-50 text-indigo-600"><Bot className="h-4 w-4" /></div>
      <div className="min-w-0 flex-1 space-y-3">
        {m.log && (
          <div className="rounded-2xl border border-slate-200 bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-200">
            <div className="mb-2 flex items-center gap-2 font-sans text-[11px] font-semibold uppercase tracking-wider text-indigo-300"><Wand2 className="h-3.5 w-3.5" />Autopilot run</div>
            <pre className="whitespace-pre-wrap">{m.log.join('\n')}</pre>
          </div>
        )}
        {m.plan && (
          <div className="rounded-2xl border border-indigo-100 bg-indigo-50/50 p-4">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-700"><ListChecks className="h-3.5 w-3.5" />Plan</div>
            <ol className="space-y-1 text-sm text-slate-700">
              {m.plan.steps.map((s, i) => (
                <li key={i} className="flex gap-2"><span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-white text-[11px] font-semibold text-indigo-600 ring-1 ring-indigo-200">{i + 1}</span>{s}</li>
              ))}
            </ol>
          </div>
        )}
        {!!m.steps?.length && (
          <div className="space-y-1.5">
            {m.steps.filter((s) => s.tool !== 'make_plan').map((s, i) => <ToolStep key={i} s={s} />)}
          </div>
        )}
        {m.pending && (
          <div className="flex items-center gap-2 text-sm text-slate-500"><Loader2 className="h-4 w-4 animate-spin text-indigo-500" />
            {m.steps?.length ? 'Working…' : 'Thinking…'}</div>
        )}
        {m.error && <ErrorBox error={m.error} />}
        {m.content && <div className="prose-chat rounded-2xl rounded-tl-md bg-slate-50 px-4 py-3 text-sm text-slate-700"><Markdown>{m.content}</Markdown></div>}
        {m.grounding && m.content && (
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
            {m.grounding.passed
              ? <Badge tone="green"><ShieldCheck className="h-3 w-3" />{m.grounding.grounded.length}/{m.grounding.numbers_checked} figures verified against the model</Badge>
              : <Badge tone="amber"><CircleAlert className="h-3 w-3" />Unverified: {m.grounding.ungrounded.join(', ')}</Badge>}
            {m.revised && <Badge tone="indigo">Self-corrected</Badge>}
          </div>
        )}
      </div>
    </div>
  )
}

function ToolStep({ s }: { s: Step }) {
  const [open, setOpen] = useState(false)
  const file = s.tool === 'generate_budget_plan' && s.result?.csv_path ? s.result : null
  return (
    <div className="rounded-xl border border-slate-200 bg-white">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm">
        {s.error ? <CircleAlert className="h-4 w-4 text-rose-500" /> : <CheckCircle2 className="h-4 w-4 text-emerald-500" />}
        <Wrench className="h-3.5 w-3.5 text-slate-400" />
        <span className="font-medium text-slate-700">{TOOL_LABEL[s.tool] ?? s.tool}</span>
        <code className="hidden truncate text-xs text-slate-400 sm:inline">{s.tool}</code>
        <span className="ml-auto text-xs tabular-nums text-slate-400">{s.seconds.toFixed(1)}s</span>
        <ChevronRight className={`h-4 w-4 text-slate-400 transition ${open ? 'rotate-90' : ''}`} />
      </button>
      {file && (
        <div className="flex flex-wrap gap-2 px-3 pb-2">
          <a className="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 hover:underline" href={fileUrl(file.csv_path.split('/').pop())}><Download className="h-3 w-3" />Weekly plan CSV</a>
          <a className="inline-flex items-center gap-1 text-xs font-medium text-indigo-600 hover:underline" href={fileUrl(file.brief_path.split('/').pop())}><FileText className="h-3 w-3" />Brief</a>
        </div>
      )}
      {open && (
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-all border-t border-slate-100 bg-slate-50 px-3 py-2 text-[11px] leading-relaxed text-slate-600">
          {s.error ? s.error : JSON.stringify({ args: s.args, result: s.result }, null, 2)}
        </pre>
      )}
    </div>
  )
}

function AutopilotCard({ onLog }: { onLog: (log: string[]) => void }) {
  const { state, refresh } = useApp()
  const [budget, setBudget] = useState(100000)
  const [weeks, setWeeks] = useState(13)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const runIt = async () => {
    setBusy(true); setErr(null)
    try { const r = await api.autopilot(budget, weeks); onLog(r.log); await refresh() }
    catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <Card title={<span className="flex items-center gap-2"><Wand2 className="h-4 w-4 text-indigo-600" />Autopilot</span>}
      subtitle="Runs the whole plan without the AI chat: check data → train → optimize → alert → write plan.">
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Budget ($)"><input type="number" className={inputCls} value={budget} onChange={(e) => setBudget(+e.target.value)} /></Field>
          <Field label="Weeks"><input type="number" className={inputCls} value={weeks} onChange={(e) => setWeeks(+e.target.value)} /></Field>
        </div>
        <Button className="w-full" onClick={runIt} loading={busy} disabled={!state?.has_data}>{busy ? 'Running…' : 'Run autopilot'}</Button>
        <ErrorBox error={err} />
      </div>
    </Card>
  )
}

function PlansCard() {
  const { state } = useApp()
  const [plans, setPlans] = useState<any[]>([])
  useEffect(() => { api.plans().then(setPlans).catch(() => {}) }, [state])
  if (!plans.length) return null
  return (
    <Card title={<span className="flex items-center gap-2"><FileText className="h-4 w-4 text-indigo-600" />Generated plans</span>}>
      <ul className="space-y-2">
        {plans.slice(0, 5).map((p) => (
          <li key={p.brief_name} className="flex items-center justify-between gap-2 rounded-xl border border-slate-100 px-3 py-2 text-sm">
            <span className="truncate text-slate-700">{p.title}<span className="block text-xs text-slate-400">{p.brief_name.match(/\d{8}-\d{6}/)?.[0]}</span></span>
            <span className="flex shrink-0 gap-2">
              <a href={fileUrl(p.csv_name)} className="text-xs font-medium text-indigo-600 hover:underline">CSV</a>
              <a href={fileUrl(p.brief_name)} className="text-xs font-medium text-indigo-600 hover:underline">Brief</a>
            </span>
          </li>
        ))}
      </ul>
    </Card>
  )
}
