import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 2

        def cnn_block(in_size, out_size):
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            cnn_block(1, 32),
            cnn_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.10),
            cnn_block(64, 128),
            cnn_block(128, 256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.15),
            cnn_block(256, 512),
            cnn_block(512, 512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.20),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AASISTLite(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        channels: int = 128,
        attn_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels % attn_heads != 0:
            raise ValueError("channels must be divisible by attn_heads")
        self.stem = nn.Sequential(
            _ConvBlock(1, 32),
            _ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            _ConvBlock(32, 64),
            _ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout * 0.75),
            _ConvBlock(64, channels),
            _ConvBlock(channels, channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=attn_heads,
            dropout=dropout,
        )
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=attn_heads,
            dropout=dropout,
        )
        self.temporal_norm = nn.LayerNorm(channels)
        self.freq_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, F, T]
        feats = self.stem(x)  # [B, C, F', T']

        temporal = feats.mean(dim=2)  # [B, C, T']
        temporal = temporal.permute(2, 0, 1)  # [T', B, C]
        temporal, _ = self.temporal_attn(
            temporal, temporal, temporal, need_weights=False
        )
        temporal = self.temporal_norm(temporal)
        temporal = temporal.mean(dim=0)  # [B, C]

        freq = feats.mean(dim=3)  # [B, C, F']
        freq = freq.permute(2, 0, 1)  # [F', B, C]
        freq, _ = self.freq_attn(freq, freq, freq, need_weights=False)
        freq = self.freq_norm(freq)
        freq = freq.mean(dim=0)  # [B, C]

        pooled = torch.cat([temporal, freq], dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def build_model(name: str) -> nn.Module:
    key = (name or "baseline_cnn").strip().lower()
    if key in {"baseline", "baseline_cnn", "cnn"}:
        return BaselineCNN()
    if key in {"aasist", "aasist_lite"}:
        return AASISTLite()
    raise ValueError(f"Unknown model '{name}'.")
