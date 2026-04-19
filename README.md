# ECG-CNN Analyzer
> **即時心律分類平台** — FastAPI + PyTorch 1D-CNN + React 儀表板

用於向董事報告的 ECG 心律異常偵測工具。支援即時訓練監控、混淆矩陣、Grad-CAM 視覺化及 .md 報告匯出。

---

## 架構總覽

```
使用者/主管
     │
     ▼
┌─────────────────────────────┐      ┌──────────────────────────────┐
│      React 前端              │      │      FastAPI 後端              │
│  (Netlify / localhost:5173) │ HTTP │  (Render / localhost:8000)    │
│                             │ /WS  │                              │
│  ▪ 訓練儀表板 (即時 loss 曲線) │◄────►│  POST /train → job_id         │
│  ▪ 模型設定面板              │      │  WS   /api/ws/{job_id}        │
│  ▪ 混淆矩陣 + 分類報告        │      │  POST /predict (CSV 上傳)     │
│  ▪ CSV 上傳 + 預測           │      │  POST /gradcam               │
│  ▪ Grad-CAM 波形視覺化       │      │  GET  /report/{job_id}       │
└─────────────────────────────┘      └──────────────────────────────┘
```

---

## 快速啟動（GitHub Codespaces / 本地）

### 1. Clone & 開啟 Codespace

```bash
git clone https://github.com/<你的帳號>/ecg-cnn-analyzer
code ecg-cnn-analyzer  # 或直接在 GitHub 上開 Codespace
```

### 2. 啟動後端

```bash
cd backend
pip install -r requirements.txt

# 啟動 (Codespace 會自動轉發 8000 port)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

瀏覽器開啟 `http://localhost:8000/docs` 驗證 API。

### 3. 啟動前端

```bash
cd frontend
npm install

# 複製環境變數 (指向後端)
cp .env.example .env
# 修改 .env:
#   VITE_API_URL=http://localhost:8000
#   VITE_WS_URL=ws://localhost:8000

npm run dev
```

瀏覽器開啟 `http://localhost:5173`

---

## 環境變數

### 前端 `.env`

```env
VITE_API_URL=http://localhost:8000      # 開發用
# VITE_API_URL=https://你的render網址   # 部署後改這行

VITE_WS_URL=ws://localhost:8000
# VITE_WS_URL=wss://你的render網址      # 部署後改這行
```

---

## 部署（完全免費）

### 後端 → Render.com

1. 在 `render.com` 建立 **Web Service**
2. 連結你的 GitHub repo
3. 設定：
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. 複製生成的 URL（例如 `https://ecg-cnn.onrender.com`）

> ⚠️ Render 免費方案冷啟動約 30 秒，第一次請求需等待。

### 前端 → Netlify

1. 在 `netlify.com` 建立新站台，連結 GitHub
2. 設定：
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
3. 在 **Environment variables** 加入：
   - `VITE_API_URL` = `https://ecg-cnn.onrender.com`
   - `VITE_WS_URL` = `wss://ecg-cnn.onrender.com`

---

## 可調整參數說明

| 參數 | 範圍 | 建議 | 說明 |
|------|------|------|------|
| 卷積層數 | 1-4 | **2** | 越多捕捉越複雜特徵 |
| 核大小 | 3/5/7 | **5** | 7 適合低頻長波形特徵 |
| 學習率 | 1e-4~1e-2 | **1e-3** | 過大易振盪 |
| Batch Size | 16/32/64 | **32** | |
| Epochs | 5-100 | **20** 快速驗證 | |
| Dropout | 0-0.5 | **0.3** | 防止過擬合 |
| 損失函數 | CrossEntropy / FocalLoss | FocalLoss 於類別不均衡時使用 | |
| 資料來源 | synthetic / mitbih | synthetic 無需授權 | |

---

## 資料格式說明（CSV 上傳）

上傳用於預測的 CSV 需符合以下格式之一：

**格式 A**（純訊號）：
```
-0.23, 0.12, 0.45, ... (共 187 個數值，代表一個心跳週期)
0.11, -0.05, 0.33, ...
```

**格式 B**（含標籤，第一欄為真實標籤，後 187 欄為訊號）：
```
0, -0.23, 0.12, 0.45, ...
2, 0.11, -0.05, 0.33, ...
```

採樣率假設為 360Hz，每筆長度 187 點（約 0.52 秒）。

---

## 類別對應（MIT-BIH AAMI EC57）

| 代號 | 全名 | 說明 |
|------|------|------|
| N | Normal | 正常竇性心律 |
| S | Supraventricular | 心室上異位心跳 |
| V | Ventricular | 心室異位心跳 ⚠️ |
| F | Fusion | 融合心跳 |
| Q | Unknown | 未分類 |

---

## 董事報告使用流程

1. **訓練**：左側面板設定參數 → 點「開始訓練」
2. **監控**：「訓練監控」分頁即時查看 loss 曲線
3. **評估**：訓練完成後自動跳轉「模型指標」分頁查看混淆矩陣
4. **上傳資料**：「預測分析」分頁上傳真實 CSV → 查看每筆分類結果
5. **Grad-CAM**：點擊任一預測結果的「分析」查看模型決策依據
6. **匯出**：右上角點「下載 .md 報告」即可得到完整董事報告

---

## 未來擴充路線圖

- [ ] 支援 PhysioNet 2017 挑戰資料集（AF 偵測）
- [ ] 模型版本管理（保存/載入不同實驗）
- [ ] 多筆歷史訓練比較圖
- [ ] PDF 報告直接匯出
- [ ] ONNX 模型匯出（部署至邊緣裝置）
