/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ShieldCheck } from 'lucide-react'
import { useApp } from '../App'
import { api } from '../api'
import { Guard } from '../components/guards'
import { Button, Card, ErrorBox, Field, PageHeader, Stat, inputCls } from '../components/ui'
import { compact, label, money, num, pct } from '../format'

export default function EvaluationPage() {
  return <Guard><Evaluation /></Guard>
}

function Evaluation() {
  const { state } = useApp()
  const [budget, setBudget] = useState(100000)
  const [ev, setEv] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const run = async () => {
    setBusy(true); setErr(null)
    try { setEv(await api.evaluate(budget)) } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Evaluation" subtitle="Can we trust the recommendations? Three independent checks." />
      <div className="grid gap-4 md:grid-cols-3">
        <Pillar n="1" title="Accuracy" text="Hide the last 13 weeks, predict them, compare with what really happened (R², MAPE)." />
        <Pillar n="2" title="Business value" text="Same budget: current split vs optimized split. How many extra sales?" />
        <Pillar n="3" title="Validation" text="On simulated data the true effects are known. Does the model find them, and how close to the best plan does it get?" />
      </div>
      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Budget for the business-value test ($)"><input type="number" className={`${inputCls} w-48`} value={budget} onChange={(e) => setBudget(+e.target.value)} /></Field>
          <Button onClick={run} loading={busy} icon={<ShieldCheck className="h-4 w-4" />}>{busy ? 'Running (about 30 s)…' : 'Run evaluation'}</Button>
          {!state?.model && <span className="text-xs text-slate-400">Train the model to include the business-value test.</span>}
        </div>
        <div className="mt-3"><ErrorBox error={err} /></div>
      </Card>

      {ev && (
        <div className="space-y-6 fade-in">
          <h2 className="text-lg font-semibold text-slate-900">1 · Accuracy on weeks the model never saw</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Test R²" value={num(ev.accuracy.test_r2)} tone={ev.accuracy.test_r2 > 0.7 ? 'up' : 'down'} delta={ev.accuracy.test_r2 > 0.7 ? 'reliable' : 'weak'} />
            <Stat label="Test error (MAPE)" value={pct(ev.accuracy.test_mape)} />
            <Stat label="Train R²" value={num(ev.accuracy.train_r2)} />
            <Stat label="Train error" value={pct(ev.accuracy.train_mape)} />
          </div>
          <Card title="Holdout forecast" subtitle="The shaded weeks were hidden from the model">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={ev.accuracy.series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} minTickGap={40} />
                <YAxis tickFormatter={(v) => compact(v)} tick={{ fontSize: 11, fill: '#64748b' }} width={56} />
                <Tooltip formatter={(v: any) => money(v)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {(() => { const t = ev.accuracy.series.filter((r: any) => r.split === 'test'); return t.length
                  ? <ReferenceArea x1={t[0].date} x2={t[t.length - 1].date} fill="#6366f1" fillOpacity={0.08} label={{ value: 'hidden weeks', fontSize: 11, fill: '#4f46e5', position: 'insideTop' }} /> : null })()}
                <Line dataKey="actual" stroke="#0f172a" strokeWidth={2} dot={false} />
                <Line dataKey="predicted" stroke="#4f46e5" strokeWidth={2} strokeDasharray="5 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {ev.business_value && (
            <>
              <h2 className="text-lg font-semibold text-slate-900">2 · Business value</h2>
              <div className="grid gap-4 sm:grid-cols-3">
                <Stat label="Current split" value={compact(ev.business_value.current_sales)} />
                <Stat label="Optimized split" value={compact(ev.business_value.optimized_sales)} />
                <Stat label="Expected uplift" value={money(ev.business_value.uplift)} delta={pct(ev.business_value.uplift_pct, 1, true)} tone="up" />
              </div>
            </>
          )}

          <h2 className="text-lg font-semibold text-slate-900">3 · Validation against known truth (simulated)</h2>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Share of best possible uplift captured" value={pct(ev.validation.share_of_best_uplift_captured, 0)} tone="up" delta="judged on the true response curves" />
            <Stat label="ROI ranking agreement" value={num(ev.validation.roi_rank_correlation)} hint="Rank correlation between true and estimated ROI (1 = perfect)" />
            <Stat label="Average ROI error" value={pct(ev.validation.mean_abs_pct_error, 0)} />
          </div>
          <Card title="True vs estimated ROI per channel">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ev.validation.channels.map((c: any) => ({ ...c, name: label(c.channel) }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} width={40} />
                <Tooltip formatter={(v: any) => num(v)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="true_roi" name="True ROI" fill="#0f172a" radius={[6, 6, 0, 0]} />
                <Bar dataKey="estimated_roi" name="Estimated ROI" fill="#4f46e5" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}
    </div>
  )
}

function Pillar({ n, title, text }: { n: string; title: string; text: string }) {
  return (
    <div className="rounded-2xl border border-slate-200/80 bg-white p-5">
      <div className="mb-2 grid h-8 w-8 place-items-center rounded-lg bg-indigo-50 text-sm font-semibold text-indigo-600">{n}</div>
      <div className="font-semibold text-slate-900">{title}</div>
      <p className="mt-1 text-sm text-slate-500">{text}</p>
    </div>
  )
}
