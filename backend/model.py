"""
ECG 1D-CNN Model — 可動態調整層數、核大小、Dropout
支援 CrossEntropy 與 Focal Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def get_loss_fn(name: str, num_classes: int, class_weights=None):
    w = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
    if name == "FocalLoss":
        return FocalLoss(gamma=2.0, weight=w)
    return nn.CrossEntropyLoss(weight=w)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ECG_CNN(nn.Module):
    """
    動態 1D-CNN:
      - conv_layers: 1–4
      - kernel_size: 3 / 5 / 7
      - dropout: 0–0.5
      - num_classes: 依資料集決定 (MIT-BIH 預設 5 類)
    """

    def __init__(
        self,
        input_len: int = 187,
        num_classes: int = 5,
        conv_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.3,
        base_filters: int = 32,
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
        in_ch = 1
        out_ch = base_filters
        for _ in range(conv_layers):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)

        self.conv = nn.Sequential(*layers)

        # 計算 flatten 後尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, input_len) → (B, 1, input_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def grad_cam(self, x: torch.Tensor):
        """Grad-CAM 1D：回傳與輸入等長的 saliency map"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x.requires_grad_(True)

        activations = []
        gradients = []

        def fwd_hook(_, __, output):
            activations.append(output)

        def bwd_hook(_, __, grad_output):
            gradients.append(grad_output[0])

        last_conv = list(self.conv.children())[-1].block[0]  # 最後一層 Conv1d
        fh = last_conv.register_forward_hook(fwd_hook)
        bh = last_conv.register_backward_hook(bwd_hook)

        logits = self.forward(x)
        pred = logits.argmax(dim=1)
        logits[0, pred[0]].backward()

        fh.remove()
        bh.remove()

        act = activations[0].detach()        # (1, C, L)
        grad = gradients[0].detach()         # (1, C, L)
        weights = grad.mean(dim=-1, keepdim=True)
        cam = (weights * act).sum(dim=1)     # (1, L)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # 內插到原始長度
        cam_full = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False
        ).squeeze()
        return cam_full.detach().numpy().tolist(), int(pred[0])
