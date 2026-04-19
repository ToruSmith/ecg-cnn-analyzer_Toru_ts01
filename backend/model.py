"""
ECG 1D-CNN Model — 可動態調整層數、核大小、Dropout
base_filters 預設改為 16（原 32），Render 免費方案每 epoch ~30s
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def get_loss_fn(name: str, num_classes: int, class_weights=None):
    w = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
    if name == "FocalLoss":
        return FocalLoss(gamma=2.0, weight=w)
    return nn.CrossEntropyLoss(weight=w)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ECG_CNN(nn.Module):
    """
    動態 1D-CNN
    base_filters=16 → Render 免費方案每 epoch ~25-35s，20 epochs 約 10 分鐘
    base_filters=32 → 每 epoch ~180s，不適合免費方案
    """
    def __init__(
        self,
        input_len: int = 187,
        num_classes: int = 5,
        conv_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.3,
        base_filters: int = 16,   # ← 從 32 改為 16
    ):
        super().__init__()
        self.config = dict(
            input_len=input_len,
            num_classes=num_classes,
            conv_layers=conv_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            base_filters=base_filters,
        )

        layers = []
        in_ch, out_ch = 1, base_filters
        for _ in range(conv_layers):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 128)   # 上限從 256 降到 128

        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),   # 從 128 降到 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def grad_cam(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.clone().detach().requires_grad_(True)

        activations, gradients = [], []

        last_conv = list(self.conv.children())[-1].block[0]
        fh = last_conv.register_forward_hook(lambda *a: activations.append(a[2]))
        bh = last_conv.register_full_backward_hook(lambda *a: gradients.append(a[2][0]))

        logits = self.forward(x)
        pred = logits.argmax(dim=1)
        logits[0, pred[0]].backward()

        fh.remove()
        bh.remove()

        act  = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=-1, keepdim=True)
        cam = F.relu((weights * act).sum(dim=1))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_full = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False
        ).squeeze()
        return cam_full.detach().numpy().tolist(), int(pred[0])
