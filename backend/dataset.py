import numpy as np
import pandas as pd

def generate_market(n=1000):
    price = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame({
        "close": price,
        "volume": np.random.randint(100, 1000, n)
    })

    # 技術指標（簡化版）
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()

    df = df.dropna()

    return df
