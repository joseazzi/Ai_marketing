/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from 'react'
import { SlidersHorizontal } from 'lucide-react'
import { useApp } from '../App'
import { api } from '../api'
import { Guard } from '../components/guards'
import { Button, Card, ErrorBox, Field, PageHeader, Stat, Table, inputCls } from '../components/ui'
import { compact, label, money, num, pct } from '../format'

export default function WhatIfPage() {
  return <Guard needModel><WhatIf /></Guard>
}

function WhatIf() {
  const { state, refresh } = useApp()
  const chans = state!.channels
  const hasRec = state!.scenarios.includes('recommended')
  const [base, setBase] = useState(hasRec ? 'recommended' : 'current split')
  const [budget, setBudget] = useState(state?.last_goal?.total_budget ?? 100000)
  const [changes, setChanges] = useState<Record<string, number>>({})
  const [realloc, setRealloc] = useState(false)
  const [res, setRes] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const active = Object.fromEntries(Object.entries(changes).filter(([, v]) => v !== 0).map(([k, v]) => [k, v / 100]))
  const run = async () => {
    setBusy(true); setErr(null)
    try { setRes(await api.whatif({ changes: active, base, reallocate: realloc, total_budget: budget, n_weeks: state?.last_goal?.n_weeks ?? 13 })); await refresh() }
    catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="What-if scenarios" subtitle='Test ideas like "cut TV by 30%" and see the expected effect on sales right away' />
      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card title="Scenario">
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Start from">
                <select className={inputCls} value={base} onChange={(e) => setBase(e.target.value)}>
                  <option value="current split">Current split</option>
                  {hasRec && <option value="recommended">Recommended plan</option>}
                </select>
              </Field>
              <Field label="Budget ($)" hint="Used if no plan exists yet">
                <input type="number" className={inputCls} value={budget} onChange={(e) => setBudget(+e.target.value)} />
              </Field>
            </div>
            <div className="space-y-4">
              {chans.map((c) => {
                const v = changes[c] ?? 0
                return (
                  <div key={c}>
                    <div className="mb-1 flex justify-between text-sm">
                      <span className="font-medium text-slate-700">{label(c)}</span>
                      <span className={`tabular-nums font-medium ${v > 0 ? 'text-emerald-600' : v < 0 ? 'text-rose-600' : 'text-slate-400'}`}>{v > 0 ? '+' : ''}{v}%</span>
                    </div>
                    <input type="range" min={-100} max={200} step={5} value={v} onChange={(e) => setChanges({ ...changes, [c]: +e.target.value })}
                      className="w-full accent-indigo-600" />
                  </div>
                )
              })}
            </div>
            <label className="flex items-center gap-3 text-sm text-slate-700">
              <input type="checkbox" checked={realloc} onChange={(e) => setRealloc(e.target.checked)} className="h-4 w-4 accent-indigo-600" />
              Move freed money to the other channels (keep the total budget the same)
            </label>
            <div className="flex gap-2">
              <Button onClick={run} loading={busy} disabled={!Object.keys(active).length} icon={<SlidersHorizontal className="h-4 w-4" />}>Run scenario</Button>
              <Button variant="ghost" onClick={() => setChanges({})}>Reset</Button>
            </div>
            <ErrorBox error={err} />
          </div>
        </Card>

        <div className="space-y-6">
          {!res ? (
            <div className="grid h-full min-h-72 place-items-center rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center text-sm text-slate-500">
              Move a slider and run the scenario.
            </div>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <Stat label="Expected sales from ads" value={compact(res.result.total_incremental_sales)}
                  delta={`${res.sales_change_vs_base >= 0 ? '+' : ''}${money(res.sales_change_vs_base)} (${pct(res.sales_change_vs_base_pct, 1, true)}) vs ${res.base}`}
                  tone={res.sales_change_vs_base >= 0 ? 'up' : 'down'} />
                <Stat label="Budget" value={compact(res.result.total_budget)} delta={`${res.budget_change_vs_base >= 0 ? '+' : ''}${money(res.budget_change_vs_base)}`} />
                <Stat label="Return per $1" value={`$${num(res.result.roi)}`} />
                {res.profit != null && <Stat label="Profit from ads" value={money(res.profit)} delta={`${res.profit_change_vs_base >= 0 ? '+' : ''}${money(res.profit_change_vs_base)} vs base`} tone={res.profit_change_vs_base >= 0 ? 'up' : 'down'} />}
              </div>
              <Card title="All scenarios so far">
                <Table head={['Scenario', 'Budget', 'Sales from ads', 'Return per $1', 'vs first']}
                  rows={res.comparison.scenarios.map((s: any) => [s.name, money(s.total_budget), money(s.incremental_sales), `$${num(s.roi)}`,
                    <span className={s.sales_vs_first >= 0 ? 'text-emerald-600' : 'text-rose-600'}>{pct(s.sales_vs_first_pct, 1, true)}</span>])} />
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
