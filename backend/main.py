"""
FastAPI 主程式
端點:
  POST /train          → 啟動訓練任務，回傳 job_id
  WS   /api/ws/{job}   → 訂閱訓練進度推播
  POST /predict        → CSV 上傳批量預測
  GET  /report/{job}   → 取得 .md 報告
  POST /gradcam        → 單筆 Grad-CAM 分析
  GET  /health         → 健康檢查
"""
import asyncio
import json
import uuid
import tempfile
import os
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from trainer import Trainer, TrainingConfig
from data_loader import load_csv_segments

app = FastAPI(title="ECG-CNN Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境請改為 Netlify URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 狀態管理 ────────────────────────────────────────────────
jobs: Dict[str, Trainer] = {}          # job_id → Trainer
ws_queues: Dict[str, asyncio.Queue] = {}  # job_id → Queue


# ─── Schemas ────────────────────────────────────────────────
class TrainRequest(BaseModel):
    conv_layers: int = Field(2, ge=1, le=4, description="卷積層數 1-4")
    kernel_size: int = Field(5, description="核大小 3/5/7")
    dropout: float = Field(0.3, ge=0.0, le=0.5)
    lr: float = Field(1e-3, gt=0, description="學習率")
    batch_size: int = Field(32, description="16/32/64")
    epochs: int = Field(20, ge=5, le=100)
    loss_fn: str = Field("CrossEntropy", description="CrossEntropy/FocalLoss")
    optimizer: str = Field("Adam", description="Adam/SGD/AdamW")
    data_source: str = Field("synthetic", description="synthetic/mitbih")
    n_samples: int = Field(5000, ge=500, le=20000)


class GradCAMRequest(BaseModel):
    job_id: str
    signal: list  # 187 個浮點數


# ─── 路由 ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "jobs_active": len(jobs)}


@app.post("/train")
async def start_training(req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()
    ws_queues[job_id] = queue

    async def push_fn(msg: str):
        await queue.put(msg)

    config = TrainingConfig(**req.model_dump())
    trainer = Trainer(config, push_fn=push_fn)
    jobs[job_id] = trainer

    # 背景執行訓練
    asyncio.create_task(trainer.run())
    return {"job_id": job_id}


@app.websocket("/api/ws/{job_id}")
async def ws_training(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue = ws_queues.get(job_id)
    if queue is None:
        await websocket.send_text(json.dumps({"event": "error", "message": "Job not found"}))
        await websocket.close()
        return

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_text(msg)

                # 解析是否已結束
                parsed = json.loads(msg)
                if parsed.get("event") in ("done", "error", "stopped"):
                    break
            except asyncio.TimeoutError:
                # 發送 ping 保持連線
                await websocket.send_text(json.dumps({"event": "ping"}))
    except WebSocketDisconnect:
        pass


@app.post("/stop/{job_id}")
def stop_training(job_id: str):
    trainer = jobs.get(job_id)
    if not trainer:
        raise HTTPException(404, "Job not found")
    trainer.stop()
    return {"status": "stopping"}


@app.post("/predict")
async def predict(
    job_id: str,
    file: UploadFile = File(..., description="CSV 檔案，每行 187 欄 ECG 數值"),
):
    trainer = jobs.get(job_id)
    if not trainer or trainer.model is None:
        raise HTTPException(400, "模型未就緒，請先訓練")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(await file.read())
        tmp_path = f.name

    try:
        X = load_csv_segments(tmp_path)
        result = trainer.predict_csv(X)
        return result
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
        raise HTTPException(404, "報告尚未生成，請等待訓練完成")
    return trainer._generate_md_report(trainer.final_metrics)


@app.get("/history/{job_id}")
def get_history(job_id: str):
    trainer = jobs.get(job_id)
    if not trainer:
        raise HTTPException(404, "Job not found")
    return {
        "history": trainer.history,
        "is_running": trainer.is_running,
    }
