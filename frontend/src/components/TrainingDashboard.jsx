import React, { useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { Activity, Target, TrendingDown } from 'lucide-react'

const s = {
  root: { display: 'flex', flexDirection: 'column', gap: 16 },
  kpiRow: { display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 },
  kpi: { background: '#111827', border: '1px solid #1e3a5f', borderRadius: 8, padding: '14px 16px', textAlign: 'center' },
  kpiLabel: { fontSize: 11, color: '#64748b', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: 1 },
  kpiValue: { fontSize: 22, fontWeight: 700, fontFamily: 'IBM Plex Mono', marginTop: 4 },
  chartTitle: { fontSize: 12, color: '#94a3b8', fontFamily: 'IBM Plex Mono', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 },
  logBox: { background: '#0a0e1a', border: '1px solid #1e3a5f', borderRadius: 6, padding: 12, height: 120, overflowY: 'auto', fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#64748b' },
  statusDot: (running) => ({
    width: 8, height: 8, borderRadius: '50%',
    background: running ? '#00ff9d' : '#475569',
    display: 'inline-block', marginRight: 6,
    animation: running ? 'pulse 1s infinite' : 'none',
  }),
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: '#111827', border: '1px solid #1e3a5f', borderRadius: 6, padding: '8px 12px', fontFamily: 'IBM Plex Mono', fontSize: 11 }}>
      <div style={{ color: '#94a3b8', marginBottom: 4 }}>Epoch {label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(4)}</div>
      ))}
    </div>
  )
}

export default function TrainingDashboard({ chartData, logs, isTraining, modelInfo }) {
  const logRef = useRef(null)
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const latest = chartData[chartData.length - 1] || {}
  const acc = latest.accuracy ? (latest.accuracy * 100).toFixed(1) : '—'
  const valLoss = latest.val_loss?.toFixed(4) ?? '—'
  const trainLoss = latest.train_loss?.toFixed(4) ?? '—'

  return (
    <div style={s.root}>
      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>

      {/* KPI */}
      <div style={s.kpiRow}>
        <div style={s.kpi}>
          <div style={s.kpiLabel}>準確率</div>
          <div style={{ ...s.kpiValue, color: '#00ff9d' }}>{acc}%</div>
        </div>
        <div style={s.kpi}>
          <div style={s.kpiLabel}>Val Loss</div>
          <div style={{ ...s.kpiValue, color: '#00d4ff' }}>{valLoss}</div>
        </div>
        <div style={s.kpi}>
          <div style={s.kpiLabel}>Train Loss</div>
          <div style={{ ...s.kpiValue, color: '#a78bfa' }}>{trainLoss}</div>
        </div>
      </div>

      {/* 損失曲線 */}
      <div>
        <div style={s.chartTitle}><TrendingDown size={12} /> 損失曲線（即時）</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
            <XAxis dataKey="epoch" tick={{ fill: '#475569', fontSize: 10 }} />
            <YAxis tick={{ fill: '#475569', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="train_loss" stroke="#a78bfa" dot={false} strokeWidth={1.5} name="Train Loss" />
            <Line type="monotone" dataKey="val_loss" stroke="#00d4ff" dot={false} strokeWidth={1.5} name="Val Loss" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 準確率曲線 */}
      <div>
        <div style={s.chartTitle}><Activity size={12} /> 驗證準確率</div>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
            <XAxis dataKey="epoch" tick={{ fill: '#475569', fontSize: 10 }} />
            <YAxis domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill: '#475569', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="accuracy" stroke="#00ff9d" dot={false} strokeWidth={2} name="Accuracy" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 模型資訊 */}
      {modelInfo && (
        <div style={{ background: '#0a0e1a', border: '1px solid #1e3a5f', borderRadius: 6, padding: '10px 14px', fontFamily: 'IBM Plex Mono', fontSize: 11 }}>
          <div style={{ color: '#64748b', marginBottom: 6 }}>MODEL INFO</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
            {Object.entries(modelInfo).map(([k, v]) => (
              <div key={k}><span style={{ color: '#475569' }}>{k}: </span><span style={{ color: '#00d4ff' }}>{v}</span></div>
            ))}
          </div>
        </div>
      )}

      {/* 訓練日誌 */}
      <div>
        <div style={{ ...s.chartTitle, marginBottom: 6 }}>
          <span style={s.statusDot(isTraining)} />
          訓練日誌
        </div>
        <div style={s.logBox} ref={logRef}>
          {logs.map((l, i) => <div key={i} style={{ marginBottom: 2 }}>{l}</div>)}
          {logs.length === 0 && <span style={{ color: '#334155' }}>等待訓練開始...</span>}
        </div>
      </div>
    </div>
  )
}
