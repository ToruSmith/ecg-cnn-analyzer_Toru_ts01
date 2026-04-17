from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "CNN Trading API running"}

@app.post("/run")
def run_experiment(params: dict):
    # 模擬訓練結果
    acc = round(random.uniform(0.55, 0.75), 3)
    sharpe = round(random.uniform(0.8, 1.8), 2)

    history = {
        "loss": [1.1, 0.9, 0.7, 0.6],
        "val_loss": [1.0, 0.85, 0.8, 0.78]
    }

    equity = [100, 105, 110, 120, 130]

    return {
        "params": params,
        "metrics": {
            "accuracy": acc,
            "sharpe": sharpe
        },
        "history": history,
        "equity_curve": equity
    }
