from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from experiment import run_experiment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
def run(params: dict):
    result = run_experiment(params)

    return {
        "params": params,
        "metrics": {
            "accuracy": result["accuracy"]
        },
        "history": {
            "loss": result["loss"],
            "val_loss": result["val_loss"]
        }
    }
