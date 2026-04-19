import React from 'react'
import { Settings2, Cpu, Database } from 'lucide-react'

const FIELD_DEFS = [
  { key: 'conv_layers',  label: '卷積層數',  type: 'select', options: [1,2,3,4],      hint: '越多層捕捉越複雜特徵，但訓練更慢' },
  { key: 'kernel_size',  label: '卷積核大小', type: 'select', options: [3,5,7],       hint: '較大的核感受野更寬，適合低頻特徵' },
  { key: 'lr',           label: '學習率 (LR)', type: 'select', options: [0.0001,0.001,0.01], hint: '過大易震盪，過小收斂慢' },
  { key: 'batch_size',   label: 'Batch Size', type: 'select', options: [16,32,64],   hint: '較大批次梯度更穩定但記憶體需求高' },
  { key: 'epochs',       label: 'Epochs',     type: 'number', min: 5, max: 100,      hint: '5-100；先用 20 快速驗證' },
  { key: 'dropout',      label: 'Dropout 比率', type: 'number', min: 0, max: 0.5, step: 0.05, hint: '防止過擬合；建議 0.2-0.4' },
  { key: 'loss_fn',      label: '損失函數',  type: 'select', options: ['CrossEntropy','FocalLoss'], hint: 'FocalLoss 適合類別不平衡' },
  { key: 'optimizer',    label: '優化器',    type: 'select', options: ['Adam','SGD','AdamW'], hint: 'Adam 通常收斂最快' },
]

const DATA_DEFS = [
  { key: 'data_source', label: '資料來源', type: 'select', options: ['synthetic','mitbih'], hint: 'synthetic=合成 ECG；mitbih=真實心律資料' },
  { key: 'n_samples',   label: '訓練樣本數', type: 'number', min: 500, max: 20000, step: 500, hint: '越多越準，但訓練時間增加' },
]

const styles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: 14 },
  section: { display: 'flex', flexDirection: 'column', gap: 12 },
  sectionTitle: { display: 'flex', alignItems: 'center', gap: 6, color: '#94a3b8', fontSize: 11, fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 4 },
  row: { display: 'flex', flexDirection: 'column', gap: 3 },
  label: { fontSize: 12, color: '#94a3b8', display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  hint: { fontSize: 10, color: '#475569', fontStyle: 'italic' },
  value: { fontSize: 12, fontFamily: 'IBM Plex Mono', color: '#00d4ff' },
  divider: { height: 1, background: '#1e3a5f', margin: '4px 0' },
}

export default function ConfigPanel({ config, onChange, isTraining }) {
  const update = (key, val) => onChange({ ...config, [key]: val })

  const renderField = (f) => {
    const v = config[f.key]
    return (
      <div key={f.key} style={styles.row}>
        <label style={styles.label}>
          <span>{f.label}</span>
          <span style={styles.value}>{v}</span>
        </label>
        {f.type === 'select' ? (
          <select value={v} onChange={e => update(f.key, isNaN(e.target.value) ? e.target.value : Number(e.target.value))} disabled={isTraining}>
            {f.options.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        ) : (
          <input type="number" min={f.min} max={f.max} step={f.step || 1} value={v}
            onChange={e => update(f.key, Number(e.target.value))} disabled={isTraining} />
        )}
        <span style={styles.hint}>{f.hint}</span>
      </div>
    )
  }

  return (
    <div style={styles.wrapper}>
      <div style={styles.section}>
        <div style={styles.sectionTitle}><Cpu size={12} /> 模型架構</div>
        {FIELD_DEFS.map(renderField)}
      </div>
      <div style={styles.divider} />
      <div style={styles.section}>
        <div style={styles.sectionTitle}><Database size={12} /> 資料設定</div>
        {DATA_DEFS.map(renderField)}
      </div>
    </div>
  )
}
