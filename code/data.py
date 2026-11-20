"""Dataset assembly: splits, censored targets, and handcrafted summary features."""
import numpy as np
import pandas as pd

from config import DATA, read_parquet
DT, NH, NW = 10.0, 60, 90
TYPES = ['VEC', 'LVL', 'SPD']
SPLITS = dict(train=('2025-01-01', '2025-02-28'),
              val=('2025-03-01', '2025-03-15'),
              test=('2025-03-16', '2025-03-31'))
BIN_S = 300.0


def load():
    X = np.load(f'{DATA}/X.npy')
    M = np.load(f'{DATA}/M.npy')
    Y = np.load(f'{DATA}/Y.npy')
    D = np.load(f'{DATA}/DUR.npy')
    VL = np.load(f'{DATA}/vlen.npy').astype(np.int32)
    meta = read_parquet(f'{DATA}/ds_meta.parquet')
    return X, M, Y, D, VL, meta


def split_index(meta):
    day = meta.day.values
    return {k: np.flatnonzero((day >= a) & (day <= b)) for k, (a, b) in SPLITS.items()}


def targets(Y, D, VL):
    """Onset bin (0 = none, 1..nb), onset validity, and duration bin.

    Frames past `VL` are right-censored at touchdown and never contribute a target.
    """
    B, H, K = Y.shape
    nb = int(np.ceil(H * DT / BIN_S))
    onset = np.zeros((B, K), np.int64)
    on_ok = np.zeros((B, K), bool)
    for k in range(K):
        y = Y[:, :, k] > 0.5
        started = y[:, 0]                              # already running at t0 -> left censored
        first = np.argmax(y, axis=1)
        has = y.any(axis=1)
        onset[:, k] = np.where(has, np.clip(first * DT // BIN_S, 0, nb - 1) + 1, 0)
        on_ok[:, k] = has & ~started
    dur = np.argmax(D, axis=2).astype(np.int64)
    return onset, on_ok, dur, nb


def frame_mask(VL):
    """(B, H, 1) mask of observed target frames."""
    H = NW
    idx = np.arange(H)[None, :]
    return (idx < VL[:, None]).astype(np.float32)[:, :, None]


def summary_features(X, M):
    f = [X[:, -1, :], X.mean(axis=1), X.std(axis=1),
         X.max(axis=1) - X.min(axis=1), X[:, -1, :] - X[:, 0, :],
         np.diff(X[:, -30:, :], axis=1).mean(axis=1), M[:, 0, :]]
    return np.concatenate(f, axis=1).astype(np.float32)
