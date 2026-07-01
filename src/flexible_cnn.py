"""A fully configurable 1D CNN for peptide-level classification.

Every architectural choice is a constructor argument so it can be driven from the
command line by ``src/train_tune.py``:

* number of conv layers      -> length of ``conv_channels``
* channels per layer         -> values in ``conv_channels``
* kernel size                -> ``kernel_sizes`` (a single size)
* multi-kernel (inception)   -> ``kernel_sizes`` with several sizes, run in parallel
                                branches per block and concatenated
* batch-norm on/off          -> ``batchnorm``
* activation                 -> ``activation``
* global pooling             -> ``pool`` (max / mean / both)
* dense head                 -> ``fc_hidden`` (extra Linear layers before output)
* dropout                    -> ``dropout``

The model stores its own ``config`` so a checkpoint can be rebuilt at inference time
via :meth:`FlexibleCNN.from_config`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .esm_embed import EMBED_DIM

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}


class MultiKernelConvBlock(nn.Module):
    """One conv block: parallel Conv1d branches (one per kernel size) concatenated.

    Output channel count is exactly ``out_ch`` regardless of the number of kernels:
    ``out_ch`` is split as evenly as possible across the branches. With a single
    kernel size this is a plain Conv1d -> (BN) -> activation block.
    """

    def __init__(self, in_ch, out_ch, kernel_sizes, batchnorm, activation):
        super().__init__()
        n = len(kernel_sizes)
        if out_ch < n:
            raise ValueError(f"out_channels ({out_ch}) must be >= number of kernels ({n})")
        # Split out_ch across branches (earlier branches absorb the remainder).
        sizes = [out_ch // n] * n
        for i in range(out_ch - sum(sizes)):
            sizes[i] += 1
        self.branches = nn.ModuleList(
            nn.Conv1d(in_ch, oc, ks, padding=ks // 2) for ks, oc in zip(kernel_sizes, sizes)
        )
        self.bn = nn.BatchNorm1d(out_ch) if batchnorm else nn.Identity()
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.act(self.bn(x))


class GlobalPool(nn.Module):
    """Collapse the length dimension. mode: 'max', 'mean', or 'maxmean' (concat)."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.maxp = nn.AdaptiveMaxPool1d(1)
        self.avgp = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if self.mode == "max":
            return self.maxp(x).squeeze(-1)
        if self.mode == "mean":
            return self.avgp(x).squeeze(-1)
        if self.mode == "maxmean":
            return torch.cat([self.maxp(x).squeeze(-1), self.avgp(x).squeeze(-1)], dim=1)
        raise ValueError(f"unknown pool mode {self.mode!r}")

    def out_multiplier(self) -> int:
        return 2 if self.mode == "maxmean" else 1


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = EMBED_DIM,
        conv_channels=(128, 128),
        kernel_sizes=(3,),
        pool: str = "max",
        fc_hidden=(),
        dropout: float = 0.3,
        batchnorm: bool = True,
        activation: str = "relu",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")
        conv_channels = tuple(conv_channels)
        kernel_sizes = tuple(kernel_sizes)
        fc_hidden = tuple(fc_hidden)
        self.config = dict(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            pool=pool,
            fc_hidden=fc_hidden,
            dropout=dropout,
            batchnorm=batchnorm,
            activation=activation,
            num_classes=num_classes,
        )

        blocks = []
        c_in = in_channels
        for c_out in conv_channels:
            blocks.append(MultiKernelConvBlock(c_in, c_out, kernel_sizes, batchnorm, activation))
            c_in = c_out
        self.features = nn.Sequential(*blocks)

        self.pool = GlobalPool(pool)
        feat_dim = c_in * self.pool.out_multiplier()

        head = []
        for h in fc_hidden:
            head += [nn.Linear(feat_dim, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            feat_dim = h
        head += [nn.Dropout(dropout), nn.Linear(feat_dim, num_classes)]
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    @classmethod
    def from_config(cls, config: dict) -> "FlexibleCNN":
        cfg = {k: v for k, v in config.items() if k != "per_residue"}
        return cls(**cfg)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for cfg in [
        dict(conv_channels=(128, 128), kernel_sizes=(3,)),
        dict(conv_channels=(64, 128, 256), kernel_sizes=(3, 5, 7), pool="maxmean", fc_hidden=(64,)),
        dict(conv_channels=(32,), kernel_sizes=(5,), batchnorm=False, activation="gelu"),
    ]:
        m = FlexibleCNN(**cfg)
        out = m(torch.randn(4, EMBED_DIM, 20))
        assert out.shape == (4, 2), out.shape
        print(f"{cfg} -> out {tuple(out.shape)}, params {m.num_parameters():,}")
