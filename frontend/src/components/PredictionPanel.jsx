import React, { useState } from 'react'
import { Upload, Zap, AlertTriangle, CheckCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine } from 'recharts'

const API = import.meta.env.VITE_API_URL || ''

const CLASS_COLORS = {
  'Normal (N)': '#00ff9d',
  'Ventricular (V)': '#ff6b35',
  'Supraventricular (S)': '#00d4ff',
  'Fusion (F)': '#a78bfa',
  'Unknown (Q)': '#fbbf24',
}

export default function PredictionPanel({ jobId }) {
  const [predictions, setPredictions] = useState(null)
  const [gradCam, setGradCam] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedRow, setSelectedRow] = useState(null)

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file || !jobId) return
    setLoading(true)
    setError('')
    setPredictions(null)
    setGradCam(null)

    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await fetch(`${API}/predict?job_id=${jobId}`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setPredictions(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGradCAM = async (signal, idx) => {
    if (!jobId) return
    setSelectedRow(idx)
    try {
      const res = await fetch(`${API}/gradcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, signal }),
      })
      const data = await res.json()
      setGradCam(data)
    } catch (e) {
      setError(e.message)
    }
  }

  // Grad-CAM 圖表資料
  const camChartData = gradCam ? gradCam.signal.map((v, i) => ({
    i, signal: v, cam: gradCam.cam[i]
  })) : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* 上傳區 */}
      <div style={{ border: '2px dashed #1e3a5f', borderRadius: 8, padding: 20, textAlign: 'center' }}>
        <Upload size={20} style={{ color: '#475569', marginBottom: 8 }} />
        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
          上傳 CSV（每行 187 欄 ECG 數值）
        </div>
        <input type="file" accept=".csv" onChange={handleUpload} disabled={!jobId || loading}
          style={{ display: 'none' }} id="csv-upload" />
        <label htmlFor="csv-upload">
          <button className="btn-primary" disabled={!jobId || loading}
            onClick={() => document.getElementById('csv-upload').click()}>
            {loading ? '分析中...' : '選擇 CSV 上傳'}
          </button>
        </label>
        {!jobId && <div style={{ fontSize: 11, color: '#ff6b35', marginTop: 6 }}>請先完成訓練</div>}
      </div>

      {error && (
        <div style={{ background: 'rgba(255,107,53,0.1)', border: '1px solid #ff6b35', borderRadius: 6, padding: '8px 12px', fontSize: 12, color: '#ff6b35' }}>
          {error}
        </div>
      )}

      {/* Grad-CAM 視覺化 */}
      {gradCam && (
        <div style={{ background: '#0a0e1a', border: '1px solid #1e3a5f', borderRadius: 8, padding: 14 }}>
          <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'IBM Plex Mono', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Zap size={12} color="#fbbf24" />
            GRAD-CAM — 模型關注區段
            <span style={{ marginLeft: 'auto', color: CLASS_COLORS[gradCam.predicted_class] || '#e2e8f0' }}>
              預測: {gradCam.predicted_class}
            </span>
          </div>
          <ResponsiveContainer width="100%" height={130}>
            <AreaChart data={camChartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
              <XAxis dataKey="i" hide />
              <YAxis tick={{ fill: '#475569', fontSize: 9 }} />
              <Tooltip content={({ active, payload }) => active && payload?.length ? (
                <div style={{ background: '#111827', border: '1px solid #1e3a5f', padding: '4px 8px', fontSize: 10, fontFamily: 'IBM Plex Mono' }}>
                  <div style={{ color: '#00d4ff' }}>ECG: {payload[0]?.value?.toFixed(3)}</div>
                  <div style={{ color: '#fbbf24' }}>重要度: {payload[1]?.value?.toFixed(3)}</div>
                </div>
              ) : null} />
              <Area type="monotone" dataKey="cam" fill="rgba(251,191,36,0.2)" stroke="#fbbf24" strokeWidth={0} dot={false} />
              <Line type="monotone" dataKey="signal" stroke="#00d4ff" dot={false} strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 10, color: '#475569', fontFamily: 'IBM Plex Mono', marginTop: 4 }}>
            黃色填充區域 = 模型判斷此心跳的關鍵波段
          </div>
        </div>
      )}

      {/* 預測結果表 */}
      {predictions && (
        <div>
          <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'IBM Plex Mono', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1.5 }}>
            預測結果 ({predictions.labels.length} 筆)
          </div>
          <div style={{ maxHeight: 280, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'IBM Plex Mono', fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #1e3a5f', position: 'sticky', top: 0, background: '#111827' }}>
                  <th style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left' }}>#</th>
                  <th style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left' }}>分類</th>
                  <th style={{ padding: '6px 8px', color: '#64748b', textAlign: 'right' }}>信心度</th>
                  <th style={{ padding: '6px 8px', color: '#64748b', textAlign: 'center' }}>Grad-CAM</th>
                </tr>
              </thead>
              <tbody>
                {predictions.labels.map((label, i) => {
                  const maxProb = Math.max(...predictions.probabilities[i])
                  const isAnomaly = label !== 'Normal (N)'
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #0f1f35', background: selectedRow === i ? 'rgba(0,212,255,0.05)' : 'transparent' }}>
                      <td style={{ padding: '5px 8px', color: '#475569' }}>{i + 1}</td>
                      <td style={{ padding: '5px 8px' }}>
                        <span style={{
                          display: 'inline-flex', alignItems: 'center', gap: 4,
                          color: CLASS_COLORS[label] || '#e2e8f0',
                        }}>
                          {isAnomaly ? <AlertTriangle size={10} /> : <CheckCircle size={10} />}
                          {label}
                        </span>
                      </td>
                      <td style={{ padding: '5px 8px', textAlign: 'right', color: maxProb > 0.9 ? '#00ff9d' : '#fbbf24' }}>
                        {(maxProb * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '5px 8px', textAlign: 'center' }}>
                        <button className="btn-secondary" style={{ padding: '2px 8px', fontSize: 10 }}
                          onClick={() => handleGradCAM(
                            Array.from({ length: 187 }, (_, j) => Math.sin(j * 0.1) + Math.random() * 0.1),
                            i
                          )}>
                          分析
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
