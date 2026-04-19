import React from 'react'

const CLASS_SHORT = ['N', 'S', 'V', 'F', 'Q']
const CLASS_FULL = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

function heatColor(val, max) {
  const ratio = max > 0 ? val / max : 0
  const r = Math.round(0 + ratio * 255)
  const g = Math.round(212 * (1 - ratio * 0.6))
  const b = Math.round(255 * (1 - ratio))
  return `rgba(${r},${g},${b},${0.15 + ratio * 0.7})`
}

export default function MetricsPanel({ confusionMatrix, reportDict }) {
  if (!confusionMatrix || !reportDict) {
    return (
      <div style={{ color: '#475569', fontSize: 12, fontFamily: 'IBM Plex Mono', padding: 20, textAlign: 'center' }}>
        訓練完成後顯示指標
      </div>
    )
  }

  const n = confusionMatrix.length
  const maxVal = Math.max(...confusionMatrix.flat())

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* 混淆矩陣 */}
      <div>
        <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'IBM Plex Mono', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1.5 }}>
          混淆矩陣
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontFamily: 'IBM Plex Mono', fontSize: 11 }}>
            <thead>
              <tr>
                <th style={{ padding: '4px 8px', color: '#475569' }}>真實 ↓ 預測 →</th>
                {CLASS_SHORT.slice(0, n).map(c => (
                  <th key={c} style={{ padding: '4px 8px', color: '#00d4ff', textAlign: 'center', minWidth: 52 }}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, i) => (
                <tr key={i}>
                  <td style={{ padding: '4px 8px', color: '#94a3b8', fontWeight: 600 }}>{CLASS_SHORT[i]}</td>
                  {row.map((val, j) => (
                    <td key={j} style={{
                      padding: '6px 8px',
                      textAlign: 'center',
                      background: heatColor(val, maxVal),
                      color: i === j ? '#00ff9d' : '#e2e8f0',
                      fontWeight: i === j ? 700 : 400,
                      border: '1px solid #1e3a5f',
                      borderRadius: 3,
                    }}>
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 分類報告 */}
      <div>
        <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'IBM Plex Mono', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1.5 }}>
          分類效能報告
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'IBM Plex Mono', fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #1e3a5f' }}>
              {['類別', 'Precision', 'Recall', 'F1', 'Support'].map(h => (
                <th key={h} style={{ padding: '6px 8px', color: '#64748b', textAlign: h === '類別' ? 'left' : 'right' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CLASS_FULL.slice(0, n).map(cls => {
              const r = reportDict[cls] || {}
              const isGood = r['f1-score'] >= 0.85
              return (
                <tr key={cls} style={{ borderBottom: '1px solid #0f1f35' }}>
                  <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{cls}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: '#a78bfa' }}>{(r.precision || 0).toFixed(3)}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: '#00d4ff' }}>{(r.recall || 0).toFixed(3)}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: isGood ? '#00ff9d' : '#ff6b35', fontWeight: 600 }}>{(r['f1-score'] || 0).toFixed(3)}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: '#475569' }}>{r.support || 0}</td>
                </tr>
              )
            })}
            <tr style={{ borderTop: '1px solid #1e3a5f', background: '#0a0e1a' }}>
              <td style={{ padding: '6px 8px', color: '#94a3b8', fontWeight: 700 }}>Macro Avg</td>
              {['precision', 'recall', 'f1-score'].map(k => {
                const v = reportDict['macro avg']?.[k] || 0
                return <td key={k} style={{ padding: '6px 8px', textAlign: 'right', color: '#e2e8f0', fontWeight: 700 }}>{v.toFixed(3)}</td>
              })}
              <td style={{ padding: '6px 8px', textAlign: 'right', color: '#475569' }}>—</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
