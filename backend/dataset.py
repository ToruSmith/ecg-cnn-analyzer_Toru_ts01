import numpy as np
import pandas as pd

def generate_market_data(n=300):

    price = 100 + np.cumsum(np.random.normal(0, 1, n))

    df = pd.DataFrame({
        "close": price,
        "volume": np.random.randint(100, 1000, n)
    })

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    df = df.dropna()

    return df
