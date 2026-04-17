from dataset import generate_market_data
from feature import build_dataset
from model import build_model
from backtest import backtest, sharpe_ratio
import numpy as np

def run_experiment(params):

    df = generate_market_data()

    X, y = build_dataset(df)

    split = int(len(X)*0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(
        input_shape=(X.shape[1], X.shape[2]),
        filters=params["filters"],
        lr=params["lr"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=params["batch"],
        verbose=0
    )

    pred = (model.predict(X_val) > 0.5).astype(int).flatten()

    equity = backtest(pred, df["close"].values[-len(pred):])
    sharpe = sharpe_ratio(equity)

    return {
        "accuracy": float(max(history.history["val_accuracy"])),
        "equity": equity,
        "sharpe": float(sharpe),
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    }
