"""
訓練引擎：
- 每個 epoch 結束後透過 WebSocket callback 推播進度
- 訓練完成後生成：混淆矩陣、分類報告、.md 報告
"""
import asyncio
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report
from typing import Callable, Optional, Dict, Any
from datetime import datetime

from model import ECG_CNN, get_loss_fn
from data_loader import get_dataset, CLASS_NAMES, SEGMENT_LEN


class TrainingConfig:
    def __init__(self, **kwargs):
        self.conv_layers: int = kwargs.get("conv_layers", 2)
        self.kernel_size: int = kwargs.get("kernel_size", 5)
        self.dropout: float = kwargs.get("dropout", 0.3)
        self.lr: float = kwargs.get("lr", 1e-3)
        self.batch_size: int = kwargs.get("batch_size", 32)
        self.epochs: int = kwargs.get("epochs", 20)
        self.loss_fn: str = kwargs.get("loss_fn", "CrossEntropy")
        self.optimizer: str = kwargs.get("optimizer", "Adam")
        self.data_source: str = kwargs.get("data_source", "synthetic")
        self.n_samples: int = kwargs.get("n_samples", 5000)


class Trainer:
    def __init__(self, config: TrainingConfig, push_fn: Optional[Callable] = None):
        self.config = config
        self.push = push_fn or (lambda msg: None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[ECG_CNN] = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "accuracy": []}
        self.final_metrics: Optional[Dict] = None
        self.is_running = False

    async def run(self):
        self.is_running = True
        cfg = self.config
        await self._push("status", "loading_data", "正在載入資料...")

        # 載入資料
        try:
            X, y = get_dataset(
                source=cfg.data_source,
                n_samples=cfg.n_samples,
            )
        except Exception as e:
            await self._push("error", str(e))
            return

        await self._push("status", "building_model", f"資料載入完成，共 {len(X)} 筆。建構模型中...")

        # 資料集切分
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)

        n_val = max(int(len(dataset) * 0.2), 1)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        # 建模型
        num_classes = int(y.max()) + 1
        self.model = ECG_CNN(
            input_len=SEGMENT_LEN,
            num_classes=num_classes,
            conv_layers=cfg.conv_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        ).to(self.device)

        # 優化器 & 損失
        optimizer_cls = {"Adam": optim.Adam, "SGD": optim.SGD, "AdamW": optim.AdamW}.get(
            cfg.optimizer, optim.Adam
        )
        # SGD 需要 momentum
        opt_kwargs = {"lr": cfg.lr}
        if cfg.optimizer == "SGD":
            opt_kwargs["momentum"] = 0.9
        optimizer = optimizer_cls(self.model.parameters(), **opt_kwargs)

        # 計算 class weights 用於不平衡資料
        counts = np.bincount(y, minlength=num_classes).astype(float)
        class_weights = (counts.sum() / (num_classes * counts)).tolist()
        criterion = get_loss_fn(cfg.loss_fn, num_classes, class_weights)

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        await self._push("model_info", {
            "param_count": param_count,
            "num_classes": num_classes,
            "train_size": n_train,
            "val_size": n_val,
            "device": str(self.device),
        })

        # 訓練迴圈
        self.history = {"train_loss": [], "val_loss": [], "accuracy": []}

        for epoch in range(1, cfg.epochs + 1):
            if not self.is_running:
                await self._push("status", "stopped", "訓練已中止")
                return

            t0 = time.time()

            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= n_train

            # Validate
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    val_loss += criterion(out, yb).item() * len(xb)
                    correct += (out.argmax(1) == yb).sum().item()
                    total += len(yb)
            val_loss /= n_val
            accuracy = correct / total

            elapsed = time.time() - t0
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["accuracy"].append(round(accuracy, 4))

            await self._push("epoch", {
                "epoch": epoch,
                "total_epochs": cfg.epochs,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "accuracy": round(accuracy, 4),
                "elapsed_sec": round(elapsed, 2),
            })

            # 讓 async 事件迴圈有機會處理其他請求
            await asyncio.sleep(0)

        # 最終評估
        await self._push("status", "evaluating", "訓練完成，計算最終指標...")
        metrics = self._evaluate(val_loader, n_val)
        self.final_metrics = metrics

        await self._push("done", {
            "status": "done",
            "final_accuracy": round(metrics["accuracy"], 4),
            "history": self.history,
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["report_dict"],
            "report_md": self._generate_md_report(metrics),
        })
        self.is_running = False

    def _evaluate(self, val_loader, n_val):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                preds = self.model(xb).argmax(1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(yb.numpy().tolist())

        n_cls = self.model.config["num_classes"]
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_cls))).tolist()
        report_dict = classification_report(
            all_labels, all_preds,
            labels=list(range(n_cls)),
            target_names=CLASS_NAMES[:n_cls],
            output_dict=True,
            zero_division=0,
        )
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "report_dict": report_dict,
            "all_preds": all_preds,
            "all_labels": all_labels,
        }

    def predict_csv(self, X: np.ndarray) -> Dict[str, Any]:
        """對 CSV 上傳的片段批量預測"""
        if self.model is None:
            raise ValueError("模型尚未訓練")
        self.model.eval()
        n_cls = self.model.config["num_classes"]
        xb = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

        return {
            "labels": [CLASS_NAMES[p] for p in preds],
            "label_indices": preds.tolist(),
            "probabilities": probs.tolist(),
            "class_names": CLASS_NAMES[:n_cls],
        }

    def grad_cam_single(self, x: np.ndarray) -> Dict:
        """單筆 Grad-CAM 分析"""
        if self.model is None:
            raise ValueError("模型尚未訓練")
        xt = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        cam, pred = self.model.grad_cam(xt)
        return {
            "signal": x.tolist(),
            "cam": cam,
            "predicted_class": CLASS_NAMES[pred],
            "predicted_index": pred,
        }

    def _generate_md_report(self, metrics: Dict) -> str:
        cfg = self.config
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_cls = self.model.config["num_classes"]
        rpt = metrics["report_dict"]

        header = f"""# ECG CNN 分析報告
**生成時間**: {now}  
**資料來源**: {cfg.data_source}  
**模型架構**: {cfg.conv_layers} 卷積層，Kernel={cfg.kernel_size}，Dropout={cfg.dropout}  
**訓練設定**: LR={cfg.lr}，Batch={cfg.batch_size}，Epochs={cfg.epochs}，Loss={cfg.loss_fn}

---

## 訓練摘要

| 指標 | 數值 |
|------|------|
| 最終驗證準確率 | **{metrics['accuracy']:.2%}** |
| 訓練 Epochs | {cfg.epochs} |
| 模型參數量 | {sum(p.numel() for p in self.model.parameters()):,} |

## 分類效能

| 類別 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
"""
        for cls_name in CLASS_NAMES[:n_cls]:
            r = rpt.get(cls_name, {})
            header += f"| {cls_name} | {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | {int(r.get('support', 0))} |\n"

        macro = rpt.get("macro avg", {})
        header += f"| **Macro Avg** | {macro.get('precision', 0):.3f} | {macro.get('recall', 0):.3f} | {macro.get('f1-score', 0):.3f} | — |\n"

        header += f"""
## 混淆矩陣

```
預測 →    {' '.join(f'{c[:3]:>6}' for c in CLASS_NAMES[:n_cls])}
"""
        for i, row in enumerate(metrics["confusion_matrix"]):
            header += f"{CLASS_NAMES[i][:3]:>6}    {'  '.join(f'{v:6d}' for v in row)}\n"
        header += "```\n"

        header += f"""
## 損失曲線摘要（最後 5 Epochs）

| Epoch | Train Loss | Val Loss | Accuracy |
|-------|-----------|---------|----------|
"""
        h = self.history
        start = max(0, len(h["train_loss"]) - 5)
        for i in range(start, len(h["train_loss"])):
            header += f"| {i+1} | {h['train_loss'][i]} | {h['val_loss'][i]} | {h['accuracy'][i]:.2%} |\n"

        header += """
---
## 結論與建議

本次 ECG 心律分類模型訓練完成。建議：
1. 若 Ventricular (V) 類別 Recall < 0.85，可增加卷積層數或改用 Focal Loss 解決類別不平衡。
2. 若 Validation Loss > Train Loss × 1.5，建議提高 Dropout 或減少層數。
3. 建議使用 MIT-BIH 真實資料集進行最終驗證後再向董事會報告。

> *此報告由 ECG-CNN Analyzer 自動生成*
"""
        return header

    async def _push(self, event: str, data: Any = None, message: str = ""):
        payload = {"event": event, "data": data, "message": message}
        await self.push(json.dumps(payload))

    def stop(self):
        self.is_running = False
