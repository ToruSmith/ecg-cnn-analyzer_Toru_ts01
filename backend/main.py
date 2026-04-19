"""
FastAPI 主程式 — Render 免費方案穩定版
改用 Polling 取代 WebSocket 長連線，解決 Render worker 重啟問題。

端點:
  GET  /               → 健康確認
  GET  /health         → 健康 + 目前 jobs 數
  POST /train          → 啟動訓練，回傳 job_id
  GET  /poll/{job_id}  → 前端每 3 秒輪詢，回傳新 events（替代 WebSocket）
  GET  /history/{job_id} → 完整訓練歷史
  POST /stop/{job_id}  → 停止訓練
  POST /predict        → CSV 批量預測
  POST /gradcam        → 單筆 Grad-CAM
  GET  /report/{job_id} → .md 報告
"""
import asyncio
import json
import uuid
import tempfile
import os
from typing import Dict, List, Any
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from trainer import Trainer, TrainingConfig
from data_loader import load_csv_segments


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
        print("[STARTUP] matplotlib ready")
    except Exception as e:
        print(f"[STARTUP] matplotlib skip: {e}")
    yield


app = FastAPI(title="ECG-CNN Analyzer API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 狀態管理 ─────────────────────────────────────────────────
jobs: Dict[str, Trainer] = {}

# 每個 job 保存最近 200 條 events（前端輪詢用）
job_events: Dict[str, deque] = {}


# ── Schemas ──────────────────────────────────────────────────
class TrainRequest(BaseModel):
    conv_layers: int   = Field(2, ge=1, le=4)
    kernel_size: int   = Field(5)
    dropout: float     = Field(0.3, ge=0.0, le=0.5)
    lr: float          = Field(1e-3, gt=0)
    batch_size: int    = Field(32)
    epochs: int        = Field(20, ge=1, le=100)
    loss_fn: str       = Field("CrossEntropy")
    optimizer: str     = Field("Adam")
    data_source: str   = Field("synthetic")
    n_samples: int     = Field(5000, ge=100, le=20000)


class GradCAMRequest(BaseModel):
    job_id: str
    signal: list


# ── 路由 ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "ECG-CNN Analyzer API", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "jobs_active": len(jobs)}


@app.post("/train")
async def start_training(req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]

    # event buffer：前端輪詢用
    buf: deque = deque(maxlen=200)
    job_events[job_id] = buf

    async def push_fn(msg: str):
        buf.append(msg)          # 存入 buffer，不用 WebSocket

    config = TrainingConfig(**req.model_dump())
    trainer = Trainer(config, push_fn=push_fn)
    jobs[job_id] = trainer

    asyncio.create_task(trainer.run())
    return {"job_id": job_id}


@app.get("/poll/{job_id}")
def poll(job_id: str, since: int = 0):
    """
    前端每 3 秒呼叫一次，回傳 index >= since 的新 events。
    since=0 表示拿全部（頁面重整後恢復用）。
    """
    buf = job_events.get(job_id)
    trainer = jobs.get(job_id)
    if buf is None:
        raise HTTPException(404, "Job not found")

    all_events = list(buf)
    new_events = all_events[since:]
    return {
        "events": new_events,
        "total": len(all_events),
        "is_running": trainer.is_running if trainer else False,
    }


@app.post("/stop/{job_id}")
def stop_training(job_id: str):
    trainer = jobs.get(job_id)
    if not trainer:
        raise HTTPException(404, "Job not found")
    trainer.stop()
    return {"status": "stopping"}


@app.get("/history/{job_id}")
def get_history(job_id: str):
    trainer = jobs.get(job_id)
    if not trainer:
        raise HTTPException(404, "Job not found")
    return {"history": trainer.history, "is_running": trainer.is_running}


@app.post("/predict")
async def predict(job_id: str, file: UploadFile = File(...)):
    trainer = jobs.get(job_id)
    if not trainer or trainer.model is None:
        raise HTTPException(400, "模型未就緒，請先訓練")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(await file.read())
        tmp_path = f.name
    try:
        X = load_csv_segments(tmp_path)
        return trainer.predict_csv(X)
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/gradcam")
def grad_cam(req: GradCAMRequest):
    trainer = jobs.get(req.job_id)
    if not trainer or trainer.model is None:
        raise HTTPException(400, "模型未就緒")
    import numpy as np
    x = np.array(req.signal, dtype=np.float32)
    if len(x) != 187:
        raise HTTPException(400, "訊號長度須為 187")
    return trainer.grad_cam_single(x)


@app.get("/report/{job_id}", response_class=PlainTextResponse)
def get_report(job_id: str):
    trainer = jobs.get(job_id)
    if not trainer or not trainer.final_metrics:
        raise HTTPException(404, "報告尚未生成")
    return trainer._generate_md_report(trainer.final_metrics)
