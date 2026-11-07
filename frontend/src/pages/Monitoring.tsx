/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { BellRing, CloudDownload, Info, TriangleAlert, Upload } from 'lucide-react'
import { useApp } from '../App'
import { api } from '../api'
import { Guard } from '../components/guards'
import { Button, Card, ErrorBox, Field, PageHeader, Stat, Table, inputCls } from '../components/ui'
import { colorFor, label, money, num, pct } from '../format'

export default function MonitoringPage() {
  return <Guard needModel><Monitoring /></Guard>
}

function Monitoring() {
  const { state, refresh } = useApp()
  const [n, setN] = useState(4)
  const [res, setRes] = useState<any>(null)
  const [alerts, setAlerts] = useState<any>(null)
  const [hist, setHist] = useState<any>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { api.history().then(setHist).catch(() => {}) }, [state?.weeks, res, alerts])

  const act = async (name: string, fn: () => Promise<any>, set: (v: any) => void) => {
    setBusy(name); setErr(null)
    try { set(await fn()); await refresh() } catch (e) { setErr((e as Error).message) } finally { setBusy(null) }
  }

  const allChans = state!.channels
  const roiHistory = (hist?.model_history ?? []).map((h: any, i: number) => ({ run: `${h.data_end}${i ? '' : ' (first)'}`, ...h.roi }))

  return (
    <div className="space-y-6">
      <PageHeader title="Monitoring & learning" subtitle="New weekly results come in → the agent checks its last forecast, updates the model, re-plans, and alerts the team if anything changed" />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Add new weekly results" subtitle="From the demo stream, or upload a CSV of new weeks">
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Demo weeks"><input type="number" min={1} max={Math.max(state!.demo_weeks_remaining, 1)} className={`${inputCls} w-28`} value={n} onChange={(e) => setN(+e.target.value)} /></Field>
            <Button onClick={() => act('ingest', () => api.ingest(n), setRes)} loading={busy === 'ingest'} disabled={!state!.demo_weeks_remaining} icon={<CloudDownload className="h-4 w-4" />}>
              Ingest ({state!.demo_weeks_remaining} left)
            </Button>
            <input ref={fileRef} type="file" accept=".csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) act('up', () => api.ingestUpload(f), setRes); e.target.value = '' }} />
            <Button variant="secondary" onClick={() => fileRef.current?.click()} loading={busy === 'up'} icon={<Upload className="h-4 w-4" />}>Upload new weeks</Button>
          </div>
          {!state!.demo_weeks_remaining && <p className="mt-3 text-xs text-slate-400">The demo stream is only available for the demo company.</p>}
        </Card>
        <Card title="Saturation check" subtitle="Finds channels wasting money (or with room to grow) and sends alerts: log, Slack, webhook or email">
          <Button onClick={() => act('alerts', api.alerts, setAlerts)} loading={busy === 'alerts'} icon={<BellRing className="h-4 w-4" />}>Check & send alerts</Button>
          {alerts && (
            <div className="mt-4 space-y-2 fade-in">
              <p className="text-xs text-slate-500">Delivered via: {alerts.delivered_via.join(', ') || 'none'}</p>
              {alerts.alerts.map((a: any, i: number) => <Alert key={i} a={a} />)}
              {!alerts.alerts.length && <p className="text-sm text-slate-500">No alerts. Every channel is in a healthy range.</p>}
            </div>
          )}
        </Card>
      </div>
      <ErrorBox error={err} />

      {res && (
        <div className="space-y-6 fade-in">
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Weeks added" value={res.weeks_added} delta={`data now ends ${res.data_now_ends}`} />
            {res.previous_model_out_of_sample && <Stat label="Last model's error on the new weeks" value={pct(res.previous_model_out_of_sample.mape)} hint="Forecast made before seeing these weeks" tone="neutral" />}
            <Stat label="Updated fit (R²)" value={num(res.refit.r2)} delta={res.refit.mode} />
          </div>
          <div className="grid gap-6 xl:grid-cols-2">
            {res.recommendation_shift && (
              <Card title="How the new data changed the plan" subtitle="Change in recommended budget per channel">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={Object.entries(res.recommendation_shift).map(([c, v]) => ({ name: label(c), v }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} width={56} tickFormatter={(v) => money(v)} />
                    <Tooltip formatter={(v: any) => money(v)} />
                    <ReferenceLine y={0} stroke="#94a3b8" />
                    <Bar dataKey="v" name="Change" radius={[6, 6, 6, 6]}>
                      {Object.values(res.recommendation_shift).map((v: any, i) => <Cell key={i} fill={v >= 0 ? '#10b981' : '#ef4444'} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )}
            <Card title="What the model learned">
              <Table head={['Channel', 'ROI before', 'ROI after', 'Next $1 before', 'Next $1 after']}
                rows={Object.entries(res.channel_changes).map(([c, v]: any) => [label(c), num(v.roi_before), num(v.roi_after), num(v.marginal_roi_before), num(v.marginal_roi_after)])} />
              {res.alerts.length > 0 && <div className="mt-4 space-y-2">{res.alerts.map((a: any, i: number) => <Alert key={i} a={a} />)}</div>}
            </Card>
          </div>
        </div>
      )}

      {roiHistory.length > 1 && (
        <Card title="How the agent's ROI estimates evolved" subtitle="One point per model update">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={roiHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
              <XAxis dataKey="run" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} width={40} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => label(v)} />
              {allChans.map((c) => <Line key={c} dataKey={c} stroke={colorFor(c, allChans)} strokeWidth={2} dot={{ r: 3 }} />)}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {!!hist?.recent_alerts?.length && (
        <Card title="Alert log">
          <div className="max-h-80 space-y-2 overflow-y-auto">
            {hist.recent_alerts.slice(0, 20).map((a: any, i: number) => <Alert key={i} a={a} time={a.timestamp} />)}
          </div>
        </Card>
      )}
    </div>
  )
}

function Alert({ a, time }: { a: any; time?: string }) {
  const warn = a.severity !== 'info'
  return (
    <div className={`flex items-start gap-2 rounded-xl px-3 py-2 text-sm ${warn ? 'bg-amber-50 text-amber-800' : 'bg-sky-50 text-sky-800'}`}>
      {warn ? <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" /> : <Info className="mt-0.5 h-4 w-4 shrink-0" />}
      <span className="flex-1">{a.message}</span>
      {time && <span className="shrink-0 text-xs opacity-60">{time.replace('T', ' ').slice(0, 16)}</span>}
    </div>
  )
}
