import { useState } from 'react'
import type { ReactNode } from 'react'
import { Database, BrainCircuit } from 'lucide-react'
import { useApp } from '../App'
import { api } from '../api'
import { Button, Empty, ErrorBox } from './ui'

/** Renders children only when data is loaded (and, with needModel, a model is trained). */
export function Guard({ needModel, children }: { needModel?: boolean; children: ReactNode }) {
  const { state, refresh, go } = useApp()
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  if (!state?.has_data) {
    return (
      <Empty icon={<Database className="h-6 w-6" />} title="Load data to get started"
        action={<Button onClick={() => go('data')}>Go to Data</Button>}>
        Use the demo company, the real brand dataset, or upload your own weekly CSV.
      </Empty>
    )
  }
  if (needModel && !state.model) {
    const train = async () => {
      setBusy(true); setErr(null)
      try { await api.fit(); await refresh() } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
    }
    return (
      <div className="space-y-4">
        <Empty icon={<BrainCircuit className="h-6 w-6" />} title="Train the model first"
          action={<Button onClick={train} loading={busy}>{busy ? 'Training… (about 10 s)' : 'Train model'}</Button>}>
          The model learns how much each channel adds to sales, how long ads keep working, and where extra spend stops paying off.
        </Empty>
        <ErrorBox error={err} />
      </div>
    )
  }
  return <>{children}</>
}
