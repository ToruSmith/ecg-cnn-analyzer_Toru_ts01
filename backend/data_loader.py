"""
資料層：MIT-BIH (wfdb) + 合成 ECG (neurokit2)
標籤對應 AAMI EC57 五分類：
  0=N (正常), 1=S (心室上), 2=V (心室), 3=F (融合), 4=Q (未知)
"""
import os
import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

# MIT-BIH 記錄號 (去掉起搏器特殊記錄)
MITBIH_RECORDS = [
    "100", "101", "103", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "203",
    "205", "207", "208", "209", "210", "212", "213", "214",
    "215", "217", "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234",
]

LABEL_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}

SEGMENT_LEN = 187  # 每個心跳樣本長度（約 0.65 秒 @360Hz）
CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]


def load_mitbih(
    data_dir: str = "./mitbih_data",
    max_records: int = 10,
    samples_per_class: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """從本地 MIT-BIH 資料夾讀取，自動下載（wfdb）"""
    try:
        import wfdb
    except ImportError:
        raise ImportError("請安裝 wfdb: pip install wfdb")

    os.makedirs(data_dir, exist_ok=True)
    X_all, y_all = [], []
    class_counts = {i: 0 for i in range(5)}

    records = MITBIH_RECORDS[:max_records]
    for rec in records:
        try:
            record = wfdb.rdrecord(
                rec,
                pn_dir="mitdb",
                channels=[0],
                sampfrom=0,
            )
            annotation = wfdb.rdann(rec, "atr", pn_dir="mitdb")

            signal = record.p_signal[:, 0]
            # 正規化
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)

            for idx, sym in zip(annotation.sample, annotation.symbol):
                label = LABEL_MAP.get(sym)
                if label is None:
                    continue
                if class_counts[label] >= samples_per_class:
                    continue

                start = idx - SEGMENT_LEN // 2
                end = start + SEGMENT_LEN
                if start < 0 or end > len(signal):
                    continue

                segment = signal[start:end].astype(np.float32)
                X_all.append(segment)
                y_all.append(label)
                class_counts[label] += 1

        except Exception as e:
            print(f"[WARN] Record {rec} failed: {e}")
            continue

    if len(X_all) == 0:
        raise ValueError("MIT-BIH 資料載入失敗，切換為合成資料")

    X = np.stack(X_all)
    y = np.array(y_all, dtype=np.int64)
    print(f"[DATA] MIT-BIH loaded: {X.shape}, class dist: {class_counts}")
    return X, y


def make_synthetic_ecg(n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 neurokit2 合成 ECG，模擬 5 類心律：
    若 neurokit2 不可用，退回純 numpy 合成
    """
    np.random.seed(seed)
    X, y = [], []

    # 每類樣本數
    per_class = n_samples // 5
    sr = 360  # 模擬採樣率

    def _normalize(sig):
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        return sig.astype(np.float32)

    try:
        import neurokit2 as nk

        configs = [
            # (heart_rate, noise, label_name)
            (70, 0.02, "N"),
            (90, 0.05, "S"),   # 心率偏高
            (60, 0.08, "V"),   # 寬 QRS
            (80, 0.04, "F"),
            (55, 0.12, "Q"),   # 高雜訊
        ]
        for hr, noise, label_name in configs:
            label = LABEL_MAP[label_name]
            for _ in range(per_class):
                hr_jitter = hr + np.random.randint(-5, 6)
                duration = max(2, int(SEGMENT_LEN / sr * 1.2))
                ecg = nk.ecg_simulate(duration=duration, sampling_rate=sr, heart_rate=hr_jitter, noise=noise)
                # 取中段 SEGMENT_LEN 點
                mid = len(ecg) // 2
                half = SEGMENT_LEN // 2
                seg = ecg[max(0, mid - half): max(0, mid - half) + SEGMENT_LEN]
                if len(seg) < SEGMENT_LEN:
                    seg = np.pad(seg, (0, SEGMENT_LEN - len(seg)))
                X.append(_normalize(seg[:SEGMENT_LEN]))
                y.append(label)

    except Exception:
        # 退回 numpy 合成（簡單正弦+雜訊模擬）
        print("[WARN] neurokit2 not available, using numpy synthetic ECG")
        t = np.linspace(0, SEGMENT_LEN / sr, SEGMENT_LEN)
        for label in range(5):
            freq = 1.0 + label * 0.3
            noise_level = 0.05 + label * 0.03
            for _ in range(per_class):
                base = np.sin(2 * np.pi * freq * t)
                spike = np.zeros_like(t)
                peak = np.random.randint(30, SEGMENT_LEN - 30)
                spike[peak] = 2.0
                sig = base + spike + np.random.randn(SEGMENT_LEN) * noise_level
                X.append(_normalize(sig))
                y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    # 打亂
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def load_csv_segments(filepath: str) -> np.ndarray:
    """
    讀取使用者上傳的 CSV，每行為一個 ECG 片段
    格式 A: 187 欄數值（無標頭）
    格式 B: 第一欄為標籤，後 187 欄為數值
    回傳 shape: (N, 187)
    """
    import pandas as pd
    df = pd.read_csv(filepath, header=None)
    data = df.values.astype(np.float32)

    if data.shape[1] == SEGMENT_LEN:
        X = data
    elif data.shape[1] == SEGMENT_LEN + 1:
        X = data[:, :SEGMENT_LEN]
    else:
        raise ValueError(f"CSV 欄數須為 {SEGMENT_LEN} 或 {SEGMENT_LEN+1}，目前為 {data.shape[1]}")

    # 逐行正規化
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True) + 1e-8
    return (X - means) / stds


def get_dataset(source: str = "synthetic", **kwargs):
    """統一入口"""
    if source == "mitbih":
        return load_mitbih(**kwargs)
    return make_synthetic_ecg(**kwargs)
