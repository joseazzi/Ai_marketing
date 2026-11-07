/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useRef, useState } from 'react'
import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Building2, CheckCircle2, Download, FlaskConical, Upload, AlertTriangle, XCircle } from 'lucide-react'
import { useApp } from '../App'
import { api } from '../api'
import { Button, Card, ErrorBox, PageHeader, Spinner, Stat } from '../components/ui'
import { colorFor, compact, label, money } from '../format'

export default function DataPage() {
  const { state, refresh } = useApp()
  const [view, setView] = useState<any>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { if (state?.has_data) api.data().then(setView).catch((e) => setErr(e.message)); else setView(null) }, [state?.has_data, state?.weeks, state?.end])

  const act = async (name: string, fn: () => Promise<unknown>) => {
    setBusy(name); setErr(null)
    try { await fn(); await refresh() } catch (e) { setErr((e as Error).message) } finally { setBusy(null) }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Data" subtitle="Weekly sales and spend per channel. About two years of weeks are needed for a reliable model." />
      <div className="grid gap-4 md:grid-cols-3">
        <SourceCard icon={<FlaskConical className="h-5 w-5" />} title="Demo company" tag="Simulated"
          text="3 years with TV, radio, email and social. The true effects are known, so the model can be checked."
          action={<Button className="w-full" variant="secondary" loading={busy === 'demo'} onClick={() => act('demo', () => api.demo())}>Load demo</Button>} />
        <SourceCard icon={<Building2 className="h-5 w-5" />} title="Real online brand" tag="Real data"
          text="146 weeks from an anonymised US apparel brand: Google, Facebook, Instagram (public Conjura dataset)."
          action={<Button className="w-full" variant="secondary" loading={busy === 'real'} disabled={!state?.real_data_available} onClick={() => act('real', () => api.real())}>Load real brand</Button>} />
        <SourceCard icon={<Upload className="h-5 w-5" />} title="Your own CSV" tag="Upload"
          text="Columns: date, sales, spend_<channel>… plus optional controls such as holiday or promo."
          action={
            <div className="flex gap-2">
              <input ref={fileRef} type="file" accept=".csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) act('upload', () => api.upload(f)); e.target.value = '' }} />
              <Button className="flex-1" variant="secondary" loading={busy === 'upload'} onClick={() => fileRef.current?.click()}>Upload CSV</Button>
              <a href="/api/template.csv" className="inline-flex items-center rounded-xl border border-slate-200 px-3 text-slate-600 hover:bg-slate-50" title="Download template"><Download className="h-4 w-4" /></a>
            </div>} />
      </div>
      <ErrorBox error={err} />

      {state?.has_data && !view && <Spinner />}
      {view && <DataView view={view} />}
    </div>
  )
}

function SourceCard({ icon, title, tag, text, action }: { icon: React.ReactNode; title: string; tag: string; text: string; action: React.ReactNode }) {
  return (
    <div className="flex flex-col rounded-2xl border border-slate-200/80 bg-white p-5">
      <div className="mb-3 flex items-center justify-between">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-50 text-indigo-600">{icon}</div>
        <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">{tag}</span>
      </div>
      <div className="font-semibold text-slate-900">{title}</div>
      <p className="mt-1 flex-1 text-sm text-slate-500">{text}</p>
      <div className="mt-4">{action}</div>
    </div>
  )
}

function DataView({ view }: { view: any }) {
  const q = view.quality
  const chans: string[] = q.channels
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Weeks" value={q.weeks} delta={`${q.start} → ${q.end}`} />
        <Stat label="Channels" value={chans.length} delta={chans.map(label).join(' · ')} />
        <Stat label="Total spend" value={compact(q.total_spend)} />
        <Stat label="Total sales" value={compact(q.total_sales)} />
      </div>

      <Card title="Data quality" subtitle="Checked before any model is trusted">
        <div className="space-y-2 text-sm">
          {q.issues.map((i: string) => <Row key={i} icon={<XCircle className="h-4 w-4 text-rose-500" />} text={i} />)}
          {q.warnings.map((w: string) => <Row key={w} icon={<AlertTriangle className="h-4 w-4 text-amber-500" />} text={w} />)}
          {!q.issues.length && !q.warnings.length && <Row icon={<CheckCircle2 className="h-4 w-4 text-emerald-500" />} text="No problems found." />}
          {q.controls.length > 0 && <p className="pt-1 text-xs text-slate-500">Outside factors the model controls for: {q.controls.join(', ')}</p>}
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Weekly spend by channel">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={view.series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} minTickGap={40} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v) => compact(v)} width={56} />
              <Tooltip formatter={(v: any) => money(v)} />
              <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => label(v)} />
              {chans.map((c) => <Area key={c} type="monotone" dataKey={c} stackId="1" stroke={colorFor(c, chans)} fill={colorFor(c, chans)} fillOpacity={0.55} />)}
            </AreaChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Weekly sales">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={view.series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} minTickGap={40} />
              <YAxis tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v) => compact(v)} width={56} />
              <Tooltip formatter={(v: any) => money(v)} />
              <Line type="monotone" dataKey="sales" stroke="#0f172a" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </>
  )
}

function Row({ icon, text }: { icon: React.ReactNode; text: string }) {
  return <div className="flex items-start gap-2 text-slate-700">{icon}<span>{text}</span></div>
}
