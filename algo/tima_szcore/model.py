"""TIMA-Net: Topology-Invariant Montage-Agnostic seizure detector.

Each channel is a token = [scale-invariant channel features] + [electrode-coord
positional encoding]. A Transformer attends OVER CHANNELS (a set, not a fixed
grid), then attention-pools to a global representation -> seizure classifier.

Because position comes from electrode coordinates (not channel index), the model
is montage-agnostic: a different montage is just a different set of (feature,
coordinate) tokens. The encoder also exposes the pooled representation `z` so a
domain-invariance loss (CORAL/IRM, Stage 3) can be applied to it.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TIMANet(nn.Module):
    def __init__(self, feat_dim: int = 9, coord_dim: int = 4, d_model: int = 128,
                 nhead: int = 4, layers: int = 3, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Sequential(nn.Linear(coord_dim, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        # attention pooling over channels (learned query)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))

    def forward(self, feats: torch.Tensor, coords: torch.Tensor,
                key_padding_mask: torch.Tensor = None, return_z: bool = False):
        # feats: (B, C, feat_dim); coords: (B, C, coord_dim)
        # key_padding_mask: (B, C) bool, True at PADDED channel slots (ignored by
        # attention). Enables mixing montages with different channel counts (e.g.
        # 18 bipolar + 19 referential) in one batch.
        tok = self.feat_proj(feats) + self.coord_proj(coords)     # (B, C, d)
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)                                          # (B, C, d)
        B = h.size(0)
        q = self.query.expand(B, -1, -1)                          # (B, 1, d)
        z, _ = self.attn_pool(q, h, h, key_padding_mask=key_padding_mask)
        z = z.squeeze(1)                                          # (B, d)
        logits = self.classifier(z)
        return (logits, z) if return_z else logits


def coral_loss(zs: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """CORAL: align 2nd-order feature statistics across domains (Stage 3).

    zs: (B, d) pooled features; domains: (B,) integer domain ids.
    Returns mean pairwise covariance distance across domains present in batch.
    """
    uniq = torch.unique(domains)
    if uniq.numel() < 2:
        return zs.new_zeros(())
    covs = []
    for d in uniq:
        zd = zs[domains == d]
        if zd.size(0) < 2:
            continue
        zd = zd - zd.mean(0, keepdim=True)
        covs.append((zd.t() @ zd) / (zd.size(0) - 1))
    if len(covs) < 2:
        return zs.new_zeros(())
    loss = zs.new_zeros(())
    n = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + ((covs[i] - covs[j]) ** 2).sum()
            n += 1
    return loss / max(1, n) / (zs.size(1) ** 2)
