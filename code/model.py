"""Encoders and multi-task heads.

Every sample is forward-passed independently: there is no attention across a batch and
no statistic computed over the test set, which removes the coupling that the reviewers
identified in the original design.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    """Residual dilated causal convolution block."""

    def __init__(self, ch, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(ch, ch, k, dilation=d))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(ch, ch, k, dilation=d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        z = F.gelu(self.conv1(F.pad(h, (self.pad, 0)))[:, :, :h.shape[-1]])
        z = self.drop(z)
        z = self.conv2(F.pad(z, (self.pad, 0)))[:, :, :h.shape[-1]]
        return F.gelu(h + self.drop(z))


class TCN(nn.Module):
    def __init__(self, cin, ch=64, dil=(1, 2, 4, 8, 16), k=3, dropout=0.15):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 1)
        self.blocks = nn.ModuleList([CausalBlock(ch, k, d, dropout) for d in dil])
        self.norm = nn.LayerNorm(ch)

    def forward(self, x, m):
        h = self.inp(torch.cat([x, m], -1).transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.norm(h[:, :, -1])


class GRUEnc(nn.Module):
    def __init__(self, cin, ch=64, dropout=0.15):
        super().__init__()
        self.rnn = nn.GRU(cin, ch, num_layers=2, batch_first=True,
                          dropout=dropout, bidirectional=False)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x, m):
        h, _ = self.rnn(torch.cat([x, m], -1))
        return self.norm(h[:, -1])


class TransformerEnc(nn.Module):
    def __init__(self, cin, ch=64, nhead=4, layers=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(cin, ch)
        self.pos = nn.Parameter(torch.randn(1, 256, ch) * 0.02)
        layer = nn.TransformerEncoderLayer(ch, nhead, ch * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x, m):
        h = self.proj(torch.cat([x, m], -1)) + self.pos[:, :x.shape[1]]
        h = self.enc(h)
        return self.norm(h[:, -1])


class Heads(nn.Module):
    """Three coupled heads over one context vector.

    occ   per-future-step occupancy logits, (B, H, K)
    onset K x (1 + B_ONSET) logits: no onset, or onset in 5-min bin b
    dur   K x B_DUR logits for the duration of the onset event
    """

    def __init__(self, ch, horizon, n_type, n_bin=6, n_dur=6, hid=128):
        super().__init__()
        self.h, self.k, self.nb, self.nd = horizon, n_type, n_bin, n_dur
        self.cproj = nn.Linear(ch, hid)
        self.temb = nn.Parameter(torch.randn(horizon, hid) * 0.02)
        self.occ = nn.Linear(hid, n_type)
        self.onset = nn.Linear(ch, n_type * (n_bin + 1))
        self.dur = nn.Linear(ch, n_type * n_dur)

    def forward(self, c):
        u = F.gelu(self.cproj(c)[:, None, :] + self.temb[None])
        return dict(occ=self.occ(u),
                    onset=self.onset(c).view(-1, self.k, self.nb + 1),
                    dur=self.dur(c).view(-1, self.k, self.nd))


class AUGUR(nn.Module):
    def __init__(self, cin, horizon, n_type=4, enc='tcn', ch=64, dropout=0.15,
                 n_bin=6, n_dur=6):
        super().__init__()
        e = dict(tcn=TCN, gru=GRUEnc, tf=TransformerEnc)[enc]
        self.encoder = e(cin, ch, dropout=dropout)
        self.heads = Heads(ch, horizon, n_type, n_bin=n_bin, n_dur=n_dur)

    def forward(self, x, m):
        return self.heads(self.encoder(x, m))


def masked_bce(logits, target, mask, pos_weight):
    """BCE over observed occupancy frames; pos_weight rebalances the rare positives."""
    l = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight,
                                           reduction='none')
    l = l * mask
    denom = (mask.sum() * logits.shape[-1]).clamp(min=1.0)
    return l.sum() / denom


def onset_loss(logits, target, valid):
    l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                        target.reshape(-1), reduction='none')
    l = l.view(target.shape) * valid.float()
    return l.sum() / valid.float().sum().clamp(min=1.0)


def duration_loss(logits, target, valid):
    l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                        target.reshape(-1), reduction='none')
    l = l.view(target.shape) * valid.float()
    return l.sum() / valid.float().sum().clamp(min=1.0)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
