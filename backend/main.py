"""
FastAPI 主程式
端點:
  GET  /               → 根路徑（健康確認）
  GET  /health         → 健康檢查
  POST /train          → 啟動訓練任務，回傳 job_id
  WS   /api/ws/{job}   → 訂閱訓練進度推播
  POST /predict        → CSV 上傳批量預測
  GET  /report/{job}   → 取得 .md 報告
  POST /gradcam        → 單筆 Grad-CAM 分析
"""
import asyncio
import json
import uuid
import tempfile
import os
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from trainer import Trainer, TrainingConfig
from data_loader import load_csv_segments


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時預先觸發 matplotlib font cache，避免第一次訓練超時
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.close("all")
        print("[STARTUP] matplotlib font cache ready")
    except Exception as e:
        print(f"[STARTUP] matplotlib pre-warm skipped: {e}")
    yield


app = FastAPI(title="ECG-CNN Analyzer API", version="1.0.0", lifespan=lifespan)

# ─── CORS：允許所有來源（Netlify 部署後可限縮） ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # credentials=True 與 origins=["*"] 不相容，改 False
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 狀態管理 ────────────────────────────────────────────────
jobs: Dict[str, Trainer] = {}
ws_queues: Dict[str, asyncio.Queue] = {}


# ─── Schemas ────────────────────────────────────────────────
class TrainRequest(BaseModel):
    conv_layers: int = Field(2, ge=1, le=4)
    kernel_size: int = Field(5)
    dropout: float = Field(0.3, ge=0.0, le=0.5)
    lr: float = Field(1e-3, gt=0)
    batch_size: int = Field(32)
    epochs: int = Field(20, ge=5, le=100)
    loss_fn: str = Field("CrossEntropy")
    optimizer: str = Field("Adam")
    data_source: str = Field("synthetic")
    n_samples: int = Field(5000, ge=500, le=20000)


class GradCAMRequest(BaseModel):
    job_id: str
    signal: list


# ─── 路由 ────────────────────────────────────────────────────
@app.get("/")
def root():
    """根路徑：讓瀏覽器和前端確認後端存活"""
    return {"status": "ok", "service": "ECG-CNN Analyzer API", "version": "1.0.0"}


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
                parsed = json.loads(msg)
                if parsed.get("event") in ("done", "error", "stopped"):
                    break
            except asyncio.TimeoutError:
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
    file: UploadFile = File(...),
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
        raise HTTPException(404, "報告尚未生成")
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

