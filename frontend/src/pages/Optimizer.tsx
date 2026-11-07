/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from 'react'
import Markdown from 'react-markdown'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ArrowDownRight, ArrowUpRight, Download, FileText, Target } from 'lucide-react'
import { useApp } from '../App'
import { api, fileUrl } from '../api'
import { Guard } from '../components/guards'
import { Button, Card, ErrorBox, Field, PageHeader, Stat, Table, inputCls } from '../components/ui'
import { compact, label, money, num, pct } from '../format'

export default function OptimizerPage() {
  return <Guard needModel><Optimizer /></Guard>
}

function Optimizer() {
  const { state, refresh } = useApp()
  const chans = state!.channels
  const [budget, setBudget] = useState(100000)
  const [weeks, setWeeks] = useState(13)
  const [maxChange, setMaxChange] = useState('')
  const [goal, setGoal] = useState<'max_sales' | 'max_profit'>('max_sales')
  const [mins, setMins] = useState<Record<string, string>>({})
  const [maxs, setMaxs] = useState<Record<string, string>>({})
  const [res, setRes] = useState<any>(null)
  const [plan, setPlan] = useState<any>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const toMap = (o: Record<string, string>) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== '' && !isNaN(+v)).map(([k, v]) => [k, +v]))

  const run = async () => {
    setBusy('opt'); setErr(null); setPlan(null)
    try {
      setRes(await api.optimize({ total_budget: budget, n_weeks: weeks, min_spend: toMap(mins), max_spend: toMap(maxs),
        max_change_pct: maxChange ? +maxChange / 100 : null, goal: state?.gross_margin ? goal : 'max_sales' }))
      await refresh()
    } catch (e) { setErr((e as Error).message) } finally { setBusy(null) }
  }
  const makePlan = async () => {
    setBusy('plan'); setErr(null)
    try { setPlan(await api.plan('recommended')) } catch (e) { setErr((e as Error).message) } finally { setBusy(null) }
  }

  const rec = res?.recommended, cur = res?.current_split
  const chart = rec ? chans.map((c) => ({ name: label(c), current: cur.allocation[c], recommended: rec.allocation[c] })) : []

  return (
    <div className="space-y-6">
      <PageHeader title="Budget optimizer" subtitle="Find the split that brings the most sales (or profit) for your budget and rules" />
      <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
        <Card title="Your goal" className="self-start xl:sticky xl:top-6">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Total budget ($)"><input type="number" className={inputCls} value={budget} onChange={(e) => setBudget(+e.target.value)} /></Field>
              <Field label="Period (weeks)" hint="A quarter = 13"><input type="number" className={inputCls} value={weeks} onChange={(e) => setWeeks(+e.target.value)} /></Field>
            </div>
            <Field label="Max change per channel (%)" hint="Keeps the plan realistic, e.g. 30. Empty = no limit.">
              <input type="number" className={inputCls} placeholder="no limit" value={maxChange} onChange={(e) => setMaxChange(e.target.value)} />
            </Field>
            <div>
              <span className="mb-1.5 block text-[13px] font-medium text-slate-600">Optimize for</span>
              <div className="grid grid-cols-2 gap-1 rounded-xl bg-slate-100 p-1 text-sm">
                {(['max_sales', 'max_profit'] as const).map((g) => (
                  <button key={g} disabled={g === 'max_profit' && !state?.gross_margin} onClick={() => setGoal(g)}
                    className={`rounded-lg px-3 py-1.5 font-medium transition disabled:opacity-40 ${goal === g ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>
                    {g === 'max_sales' ? 'Sales' : 'Profit'}
                  </button>
                ))}
              </div>
              {!state?.gross_margin && <p className="mt-1 text-xs text-slate-400">Set a gross margin in Settings to optimize for profit.</p>}
            </div>
            <details className="rounded-xl border border-slate-200 p-3 text-sm">
              <summary className="cursor-pointer font-medium text-slate-700">Channel limits (optional)</summary>
              <div className="mt-3 space-y-2">
                <div className="grid grid-cols-[1fr_80px_80px] gap-2 text-xs text-slate-500"><span>Channel</span><span>Min $</span><span>Max $</span></div>
                {chans.map((c) => (
                  <div key={c} className="grid grid-cols-[1fr_80px_80px] items-center gap-2">
                    <span className="truncate text-slate-700">{label(c)}</span>
                    <input className={`${inputCls} px-2 py-1`} value={mins[c] ?? ''} onChange={(e) => setMins({ ...mins, [c]: e.target.value })} />
                    <input className={`${inputCls} px-2 py-1`} value={maxs[c] ?? ''} onChange={(e) => setMaxs({ ...maxs, [c]: e.target.value })} />
                  </div>
                ))}
              </div>
            </details>
            <Button className="w-full" onClick={run} loading={busy === 'opt'} icon={<Target className="h-4 w-4" />}>Optimize</Button>
            <ErrorBox error={err} />
          </div>
        </Card>

        <div className="space-y-6">
          {!res ? (
            <div className="grid h-full min-h-72 place-items-center rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center text-sm text-slate-500">
              Set your budget and press Optimize to see the recommended split.
            </div>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-3">
                <Stat label="Expected sales from ads" value={compact(rec.total_incremental_sales)}
                  delta={`${pct(res.uplift_vs_current_pct, 1, true)} vs current split`} tone={res.uplift_vs_current >= 0 ? 'up' : 'down'} />
                <Stat label="Return per $1" value={`$${num(rec.roi)}`} delta={`was $${num(cur.roi)}`} tone={rec.roi >= cur.roi ? 'up' : 'down'} />
                <Stat label="Extra sales" value={money(res.uplift_vs_current)} delta={`${res.n_weeks} weeks from ${res.plan_starts}`} />
              </div>
              {res.recommended_profit != null && (
                <div className="grid gap-4 sm:grid-cols-3">
                  <Stat label="Profit from ads" value={money(res.recommended_profit)} delta={`${res.profit_uplift_vs_current >= 0 ? '+' : ''}${money(res.profit_uplift_vs_current)} vs current`} tone={res.profit_uplift_vs_current >= 0 ? 'up' : 'down'} />
                  <Stat label="Current split profit" value={money(res.current_split_profit)} tone={res.current_split_profit < 0 ? 'down' : 'neutral'} delta={res.current_split_profit < 0 ? 'losing money' : undefined} />
                  <Stat label="Budget left unspent" value={money(res.unspent_budget)} />
                </div>
              )}
              <Card title="Current vs recommended split">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <YAxis tickFormatter={(v) => compact(v)} tick={{ fontSize: 11, fill: '#64748b' }} width={56} />
                    <Tooltip formatter={(v: any) => money(v)} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="current" name="Current split" fill="#cbd5e1" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="recommended" name="Recommended" fill="#4f46e5" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Why" subtitle="The biggest moves and the model's reason for each">
                <ul className="space-y-3">
                  {res.biggest_moves.map((mv: any) => (
                    <li key={mv.channel} className="flex gap-3 rounded-xl border border-slate-100 p-3">
                      <div className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl ${mv.change > 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600'}`}>
                        {mv.change > 0 ? <ArrowUpRight className="h-5 w-5" /> : <ArrowDownRight className="h-5 w-5" />}
                      </div>
                      <div className="text-sm">
                        <div className="font-medium text-slate-900">{mv.change > 0 ? 'Increase' : 'Reduce'} {label(mv.channel)} by {money(Math.abs(mv.change))}</div>
                        <div className="text-slate-500">{mv.reason[0].toUpperCase() + mv.reason.slice(1)}. Next $1 returns ${num(mv.current_marginal_roi)} now → ${num(mv.optimized_marginal_roi)} after.</div>
                      </div>
                    </li>
                  ))}
                </ul>
              </Card>
              <Card title="Plan detail" action={<Button variant="secondary" onClick={makePlan} loading={busy === 'plan'} icon={<FileText className="h-4 w-4" />}>Generate plan file</Button>}>
                <Table head={['Channel', 'Current', 'Recommended', 'Per week', 'Expected sales', 'Next $1', 'Saturation']}
                  rows={chans.map((c) => [label(c), money(cur.allocation[c]), money(rec.allocation[c]), money(rec.allocation[c] / res.n_weeks),
                    money(rec.expected_sales[c]), `$${num(rec.marginal_roi[c])}`, pct(rec.saturation_pct[c], 0)])} />
                {plan && (
                  <div className="mt-5 space-y-3 border-t border-slate-100 pt-5 fade-in">
                    <div className="flex flex-wrap gap-2">
                      <a href={fileUrl(plan.csv_name)}><Button variant="secondary" icon={<Download className="h-4 w-4" />}>Weekly plan (CSV)</Button></a>
                      <a href={fileUrl(plan.brief_name)}><Button variant="secondary" icon={<Download className="h-4 w-4" />}>Brief (Markdown)</Button></a>
                    </div>
                    <div className="prose-chat rounded-xl bg-slate-50 p-4 text-sm"><Markdown>{plan.brief_markdown}</Markdown></div>
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
