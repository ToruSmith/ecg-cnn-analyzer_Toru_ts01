import numpy as np

def generate_data(samples=500):
    X = np.random.rand(samples, 5, 30, 1)

    # 模擬 label（3分類）
    y = np.random.randint(0, 3, size=(samples,))
    y = np.eye(3)[y]

    return X, y
