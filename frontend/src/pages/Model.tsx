/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from 'react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ComposedChart, Tooltip, XAxis, YAxis,
} from 'recharts'
import { RefreshCw } from 'lucide-react'
import { useApp } from '../App'
import { api } from '../api'
import { Guard } from '../components/guards'
import { Badge, Button, Card, ErrorBox, PageHeader, Spinner, Stat, Table } from '../components/ui'
import { colorFor, compact, label, money, num, pct } from '../format'

export default function ModelPage() {
  return <Guard needModel><ModelView /></Guard>
}

function ModelView() {
  const { state, refresh } = useApp()
  const [m, setM] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => { api.model().then(setM).catch((e) => setErr(e.message)) }, [state?.model?.trained_weeks, state?.model?.r2, state?.gross_margin])

  const retrain = async () => {
    setBusy(true); setErr(null)
    try { await api.fit({ from_scratch: true }); await refresh() } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }

  if (!m) return <><ErrorBox error={err} /><Spinner label="Loading model…" /></>
  const perf = m.performance
  const chans: string[] = perf.channels.map((c: any) => c.channel)
  const allChans: string[] = state!.channels
  const breakeven = perf.breakeven_sales_per_ad_dollar

  // response curves → one data array per channel
  const curves: Record<string, any[]> = {}
  for (const r of m.curves) (curves[r.channel] ??= []).push(r)

  return (
    <div className="space-y-6">
      <PageHeader title="Channel insights" subtitle="What the Media Mix Model learned from your history"
        action={<Button variant="secondary" onClick={retrain} loading={busy} icon={<RefreshCw className="h-4 w-4" />}>Retrain</Button>} />
      <ErrorBox error={err} />
      {state?.model?.stale && <ErrorBox error="New data arrived since the last training. Retrain to use it." />}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Fit (R²)" value={num(m.metrics.r2)} delta={m.metrics.r2 > 0.8 ? 'Good fit' : 'Weak fit'} tone={m.metrics.r2 > 0.8 ? 'up' : 'down'} hint="Share of week-to-week sales movement the model explains" />
        <Stat label="Average error (MAPE)" value={pct(m.metrics.mape)} hint="Average weekly prediction error" />
        <Stat label="Best channel" value={label(perf.best_channel_by_roi)} delta={`ROI ${num(perf.channels[0].roi)}`} tone="up" />
        <Stat label="Sales driven by ads" value={pct(perf.media_driven_share_of_sales, 0)} hint="The rest is baseline: brand, seasonality, loyal customers" />
      </div>

      <Card title="Actual vs predicted sales">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={m.fit}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} minTickGap={40} />
            <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v) => compact(v)} width={56} />
            <Tooltip formatter={(v: any) => money(v)} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="actual" stroke="#0f172a" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="predicted" stroke="#4f46e5" strokeWidth={2} strokeDasharray="5 4" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Return per $1 of ads" subtitle={`Historical ROI vs the next dollar. Below the line, ads don't pay off${state?.gross_margin ? ` at a ${pct(state.gross_margin, 0)} margin` : ''}.`}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={perf.channels.map((c: any) => ({ ...c, name: label(c.channel) }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} width={40} />
              <Tooltip formatter={(v: any) => `$${v.toFixed(2)}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine y={breakeven} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'break-even', fontSize: 11, fill: '#ef4444', position: 'insideTopRight' }} />
              <Bar dataKey="roi" name="ROI (historical)" fill="#4f46e5" radius={[6, 6, 0, 0]} />
              <Bar dataKey="marginal_roi_now" name="Next $1 returns" fill="#a5b4fc" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Saturation at current spend" subtitle="How much of each channel's maximum effect is already used">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={perf.channels.map((c: any) => ({ name: label(c.channel), v: c.saturation_pct, ch: c.channel }))} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} width={110} />
              <Tooltip formatter={(v: any) => pct(v, 0)} />
              <ReferenceLine x={0.7} stroke="#f59e0b" strokeDasharray="4 4" />
              <Bar dataKey="v" name="Saturation" radius={[0, 6, 6, 0]}>
                {perf.channels.map((c: any) => <Cell key={c.channel} fill={c.saturation_pct > 0.7 ? '#f59e0b' : '#10b981'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Response curves" subtitle="Weekly incremental sales for each level of weekly spend. Dots mark today's spend. Where a curve flattens, extra money is wasted.">
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" />
            <XAxis type="number" dataKey="weekly_spend" tickFormatter={(v) => compact(v)} tick={{ fontSize: 11, fill: '#64748b' }} allowDuplicatedCategory={false} />
            <YAxis tickFormatter={(v) => compact(v)} tick={{ fontSize: 11, fill: '#64748b' }} width={56} />
            <Tooltip formatter={(v: any) => money(v)} labelFormatter={(v) => `Weekly spend ${money(v as number)}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {chans.map((c) => <Line key={c} data={curves[c]} type="monotone" dataKey="weekly_sales" name={label(c)} stroke={colorFor(c, allChans)} strokeWidth={2} dot={false} />)}
            {chans.map((c) => <Scatter key={`p-${c}`} data={[m.current_points[c]]} dataKey="weekly_sales" fill={colorFor(c, allChans)} legendType="none" name={`${label(c)} now`} />)}
          </ComposedChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Sales breakdown" subtitle="Baseline (sales without ads) and each channel's contribution, week by week">
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={m.decomposition}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} minTickGap={40} />
            <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v) => compact(v)} width={56} />
            <Tooltip formatter={(v: any) => money(v)} />
            <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => label(v)} />
            <Area type="monotone" dataKey="baseline" stackId="1" stroke="#cbd5e1" fill="#e2e8f0" />
            {chans.map((c) => <Area key={c} type="monotone" dataKey={c} stackId="1" stroke={colorFor(c, allChans)} fill={colorFor(c, allChans)} fillOpacity={0.6} />)}
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Channel table">
        <Table head={['Channel', 'ROI', 'Next $1', ...(state?.gross_margin ? ['Profit per $1'] : []), 'Saturation', 'Weekly spend now', 'Stops paying off at', 'Ad effect lasts', 'Status']}
          rows={perf.channels.map((c: any) => [
            label(c.channel), num(c.roi), num(c.marginal_roi_now),
            ...(state?.gross_margin ? [<span className={c.profit_per_ad_dollar < 0 ? 'text-rose-600' : 'text-emerald-600'}>{money(c.profit_per_ad_dollar, 2)}</span>] : []),
            pct(c.saturation_pct, 0), money(c.current_weekly_spend),
            c.weekly_spend_where_next_dollar_stops_paying_off == null ? '—' : c.weekly_spend_where_next_dollar_stops_paying_off === 0 ? 'never pays off' : `${money(c.weekly_spend_where_next_dollar_stops_paying_off)}/wk`,
            `${num(c.ad_effect_lasts_weeks, 1)} wk`,
            <Badge tone={c.status === 'saturated' ? 'amber' : 'green'}>{c.status}</Badge>,
          ])} />
      </Card>
    </div>
  )
}
