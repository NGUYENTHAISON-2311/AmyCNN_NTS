"""1D CNN that classifies a centred ``2k+1`` residue window as core / non-core.

Input channels default to 480 (ESM2-35M embedding dim). The network produces 2 output
logits (non-core, core); softmax is applied only at inference time. The window is
collapsed with adaptive max pooling so the model is agnostic to the exact window width.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .esm_embed import EMBED_DIM


class ResidueCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = EMBED_DIM,
        hidden_channels=(128),
        kernel_size: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in hidden_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, width) -> logits (batch, num_classes)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


if __name__ == "__main__":
    model = ResidueCNN()
    dummy = torch.randn(4, EMBED_DIM, 20)
    out = model(dummy)
    print("output shape:", tuple(out.shape))  # expect (4, 2)
    assert out.shape == (4, 2)
