"""Does a manoeuvre forecast improve the arrival-time forecast?

Design
------
Every variant shares one encoder (the dilated causal stack of the paper, ch=56) and one
scalar head.  Only the extra input vector changes, so a difference in error is attributable
to that vector and nothing else:

  kine      no extra input                      -- the arrival-time baseline
  augur     AUGUR's occupancy forecast at t0     -- 12 numbers, out-of-sample
  oracle    the realised manoeuvre in the window -- upper bound, not attainable
  shuffled  augur, permuted across flights       -- control for extra input dimensions

The target is the time from t0 to the end of the contiguous track (touchdown proxy), in
minutes.  AUGUR's occupancy probabilities for a training flight are produced by a model that
never saw that flight (two-fold cross-fitting over days), and those for val/test come from
the released `pred_tcn.npz`, which was trained on training days only.  No flight is scored
with a forecaster that saw it.

usage
  python src/eta_experiment.py augur --fold 0
  python src/eta_experiment.py eta --variant augur --seed 0
  python src/eta_experiment.py agg
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from model import TCN, masked_bce, onset_loss, duration_loss, count_params

from config import DATA, read_parquet
DT = 10.0
THR = 0.70                      # AUGUR's reported operating threshold
N_FOLD = 2
EPOCHS_A = 22                   # AUGUR folds: same recipe as the released run
EPOCHS_E = 15
BATCH = 256
LR = 2e-3
GAP_TOL = 2.0                   # min; drop arrivals whose TTL disagrees with the window


# ---------------------------------------------------------------- data

def load_all():
    X = np.load(f'{DATA}/X.npy')
    M = np.load(f'{DATA}/M.npy')
    Y = np.load(f'{DATA}/Y.npy')
    DUR = np.load(f'{DATA}/DUR.npy')
    meta = read_parquet(f'{DATA}/ds_meta.parquet')
    ttl = read_parquet(f'{DATA}/ttl.parquet')
    meta = meta.merge(ttl[['flight', 'day', 'ttl_min']], on=['flight', 'day'], how='left')
    return X, M, Y, DUR, meta


def keep_mask(meta):
    """Drop arrivals whose target is inconsistent with the observed window.

    A handful of call signs are reused on consecutive days and the stored track then spans
    the gap; for those the window and the target disagree by more than GAP_TOL.  See
    build_ttl.py.
    """
    ok = np.isfinite(meta.ttl_min.values)
    ok &= meta.ttl_min.values >= meta.vlen_min.values - GAP_TOL
    ok &= meta.ttl_min.values > 1.0
    return ok


def splits(meta, ok):
    sp = D.split_index(meta)
    return {k: v[ok[v]] for k, v in sp.items()}


# ---------------------------------------------------------------- AUGUR features

def augur_features(P, thr=THR):
    """Twelve numbers per arrival from the occupancy probabilities P (n, 90, 3)."""
    n, H, K = P.shape
    f = np.zeros((n, 4 * K), np.float32)
    mins = DT / 60.0
    first = P >= thr
    for k in range(K):
        p = P[:, :, k]
        f[:, 4 * k + 0] = p.max(axis=1)
        f[:, 4 * k + 1] = p.sum(axis=1) * mins
        has = first[:, :, k].any(axis=1)
        idx = np.where(has, first[:, :, k].argmax(axis=1), H)
        f[:, 4 * k + 2] = idx * mins
        f[:, 4 * k + 3] = p[:, 0]
    return f


def oracle_features(Y, VL, DT_=DT):
    """The same twelve numbers computed from the realised events (attainable? no)."""
    n, H, K = Y.shape
    f = np.zeros((n, 4 * K), np.float32)
    mins = DT_ / 60.0
    for k in range(K):
        y = Y[:, :, k].copy()
        mask = np.arange(H)[None, :] < VL[:, None]
        y = y * mask
        f[:, 4 * k + 0] = y.max(axis=1)
        f[:, 4 * k + 1] = y.sum(axis=1) * mins
        has = y > 0.5
        idx = np.where(has.any(axis=1), has.argmax(axis=1), H)
        f[:, 4 * k + 2] = idx * mins
        f[:, 4 * k + 3] = y[:, 0]
    return f


# ---------------------------------------------------------------- cross-fitting AUGUR

def cmd_augur_fold(args):
    X, M, Y, DUR, meta = load_all()
    ok = keep_mask(meta)
    sp = splits(meta, ok)
    VL = meta.vlen.values
    mask = D.frame_mask(VL)
    onset, on_ok, dur, nb = D.targets(Y, DUR, VL)

    days = np.unique(meta.day.values[sp['train']])
    rng = np.random.default_rng(0)
    fold_of_day = {d: i % N_FOLD for i, d in enumerate(rng.permutation(days))}
    fd = np.array([fold_of_day.get(d, -1) for d in meta.day.values])
    tr = sp['train'][fd[sp['train']] != args.fold]
    te = sp['train'][fd[sp['train']] == args.fold]
    print(f'fold {args.fold}: train {len(tr)} holdout {len(te)} days '
          f'{len(np.unique(meta.day.values[tr]))}/{len(np.unique(meta.day.values[te]))}', flush=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mtr = mask[tr]
    pos = (Y[tr] * mtr).sum(axis=(0, 1))
    neg = mtr.sum() - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 30.0), dtype=torch.float32)

    from model import AUGUR
    model = AUGUR(X.shape[2] + M.shape[2], Y.shape[1], len(D.TYPES), enc='tcn', ch=56,
                  dropout=0.15, n_bin=nb)
    print('params', count_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = ((len(tr) + BATCH - 1) // BATCH) * EPOCHS_A
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    t0 = time.time()
    for ep in range(EPOCHS_A):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), BATCH):
            b = tr[perm[i:i + BATCH]]
            out = model(torch.from_numpy(X[b]), torch.from_numpy(M[b]))
            loss = masked_bce(out['occ'], torch.from_numpy(Y[b]),
                              torch.from_numpy(mask[b]), pw)
            vb = torch.from_numpy(on_ok[b])
            loss = loss + 0.5 * onset_loss(out['onset'], torch.from_numpy(onset[b]), vb)
            loss = loss + 0.3 * duration_loss(out['dur'], torch.from_numpy(dur[b]), vb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print(f'  ep {ep + 1:02d} loss {tot / len(tr):.4f} [{time.time() - t0:.0f}s]', flush=True)

    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            b = te[i:i + 512]
            out = model(torch.from_numpy(X[b]), torch.from_numpy(M[b]))
            P.append(torch.sigmoid(out['occ']).numpy())
    np.save(f'{DATA}/augur_feat_fold{args.fold}.npy',
            augur_features(np.concatenate(P)))
    np.save(f'{DATA}/augur_idx_fold{args.fold}.npy', te)
    print('saved fold', args.fold, flush=True)


def augur_feature_matrix(sp, meta, ok):
    """Features for every arrival: fold models for training days, released model elsewhere.

    The released `pred_tcn.npz` stores one row per arrival of the *unfiltered* split, so the
    rows are re-aligned by absolute index before the filtered subset is taken.
    """
    n = len(meta)
    F = np.zeros((n, 12), np.float32)
    filled = np.zeros(n, bool)
    for k in range(N_FOLD):
        idx = np.load(f'{DATA}/augur_idx_fold{k}.npy')
        F[idx] = np.load(f'{DATA}/augur_feat_fold{k}.npy')
        filled[idx] = True
    sp_all = D.split_index(meta)
    z = np.load(f'{DATA}/pred_tcn.npz')
    for key in ('val', 'test'):
        full = augur_features(z[key])
        where = {i: j for j, i in enumerate(sp_all[key])}
        sub = np.array([where[i] for i in sp[key]], dtype=np.int64)
        F[sp[key]] = full[sub]
        filled[sp[key]] = True
    used = np.concatenate([sp['train'], sp['val'], sp['test']])
    assert filled[used].all(), f'{(~filled[used]).sum()} arrivals without AUGUR features'
    return F


# ---------------------------------------------------------------- ETA model

class ETA(nn.Module):
    def __init__(self, cin, extra, ch=56, dropout=0.15):
        super().__init__()
        self.enc = TCN(cin, ch, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(ch + extra, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, m, e):
        c = self.enc(x, m)
        if e is not None and e.shape[1] > 0:
            c = torch.cat([c, e], dim=-1)
        return self.head(c).squeeze(-1)


def cmd_eta(args):
    X, M, Y, DUR, meta = load_all()
    ok = keep_mask(meta)
    sp = splits(meta, ok)
    VL = meta.vlen.values

    F = augur_feature_matrix(sp, meta, ok) if args.variant in ('augur', 'shuffled') else None
    if args.variant == 'oracle':
        F = oracle_features(Y, VL)

    tr, va, te = sp['train'], sp['val'], sp['test']
    y = meta.ttl_min.values.astype(np.float32)
    mu, sd = y[tr].mean(), y[tr].std()
    yz = (y - mu) / sd

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    extra = 0 if args.variant == 'kine' else 12
    model = ETA(X.shape[2] + M.shape[2], extra, ch=56, dropout=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = ((len(tr) + BATCH - 1) // BATCH) * EPOCHS_E
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    lossf = nn.SmoothL1Loss(beta=0.5)

    E = None
    if F is not None:
        E = F.copy()
        if args.variant == 'shuffled':
            perm = rng.permutation(len(tr))
            E[tr] = E[tr][perm]

    def fwd(ix):
        e = None if E is None else torch.from_numpy(E[ix])
        p = model(torch.from_numpy(X[ix]), torch.from_numpy(M[ix]), e)
        return p.detach().numpy() * sd + mu

    t0 = time.time()
    best = (1e9, None, -1)
    for ep in range(EPOCHS_E):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), BATCH):
            b = tr[perm[i:i + BATCH]]
            out = model(torch.from_numpy(X[b]), torch.from_numpy(M[b]), None if E is None
                        else torch.from_numpy(E[b]))
            loss = lossf(out, torch.from_numpy(yz[b]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        pv = fwd(va)
        mae = np.abs(pv - y[va]).mean()
        print(f'  ep {ep + 1:02d} loss {tot / len(tr):.4f} valMAE {mae:.3f} '
              f'[{time.time() - t0:.0f}s]', flush=True)
        if mae < best[0]:
            best = (mae, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    pt, pv = fwd(te), fwd(va)

    os.makedirs(f'{DATA}/eta', exist_ok=True)
    np.savez_compressed(f'{DATA}/eta/{args.variant}_s{args.seed}.npz',
                        test=pt, val=pv, test_idx=te, val_idx=va,
                        val_mae=best[0], best_ep=best[2], params=count_params(model))
    print(f'{args.variant} seed{args.seed}: val MAE {best[0]:.3f} min '
          f'(ep {best[2]}) test MAE {np.abs(pt - y[te]).mean():.3f} min', flush=True)


# ---------------------------------------------------------------- aggregation

def cmd_agg(args):
    X, M, Y, DUR, meta = load_all()
    ok = keep_mask(meta)
    sp = splits(meta, ok)
    y = meta.ttl_min.values.astype(np.float32)
    te = sp['test']
    days = meta.day.values[te]

    preds = {}
    for v in ('kine', 'augur', 'oracle', 'shuffled'):
        fs = sorted(f for f in os.listdir(f'{DATA}/eta') if f.startswith(v + '_s'))
        if not fs:
            continue
        P = [np.load(f'{DATA}/eta/{f}')['test'] for f in fs]
        preds[v] = np.mean(P, axis=0)
        preds[v + '_nseed'] = len(P)

    # dead reckoning: range over ground speed at t0 (channels 1 and 3 of Table 3); it
    # carries no deceleration profile and is kept only to show how far that gets you
    r_km = np.abs(X[te, -1, 0]) * 100.0
    gs = X[te, -1, 2] * 100.0
    preds['deadreckon'] = np.clip(r_km * 1000.0 / np.maximum(gs, 5.0) / 60.0, 1.0, 60.0)

    # no history at all: ridge on the last frame's 25 channels, the analogue of the
    # static logistic of Table 4
    feat = np.concatenate([X[:, -1, :], M[:, -1, :]], axis=1).astype(np.float64)
    tr = sp['train']
    mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-9
    A = np.c_[(feat[tr] - mu) / sd, np.ones(len(tr))]
    w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y[tr].astype(np.float64))
    preds['nohistory'] = np.c_[(feat[te] - mu) / sd, np.ones(len(te))] @ w

    # constant: median of the training target
    preds['constant'] = np.full(len(te), float(np.median(y[sp['train']])))

    def mae(p, ix=None):
        return float(np.abs(p - (y[te] if ix is None else y[te][ix])).mean() * 60.0)

    out = {}
    for k, p in preds.items():
        if k.endswith('_nseed'):
            out.setdefault('n_seed', {})[k.split('_')[0]] = int(p)
            continue
        e = (p - y[te]) * 60.0
        out[k] = dict(mae_s=float(np.abs(e).mean()), rmse_s=float(np.sqrt((e ** 2).mean())),
                      bias_s=float(e.mean()),
                      p90_s=float(np.percentile(np.abs(e), 90)))

    # paired day-block bootstrap on the difference against kine
    uday = np.unique(days)
    rf = np.random.default_rng(0)
    B = 1000
    boots = {}
    for k in preds:
        if k.endswith('_nseed') or k == 'kine':
            continue
        d = np.empty(B)
        for b in range(B):
            pick = rf.choice(uday, len(uday), replace=True)
            ix = np.concatenate([np.flatnonzero(days == x) for x in pick])
            base = np.abs(preds['kine'][ix] - y[te][ix]).mean()
            alt = np.abs(preds[k][ix] - y[te][ix]).mean()
            d[b] = (alt - base) * 60.0
        boots[k] = dict(delta_s=out[k]['mae_s'] - out['kine']['mae_s'],
                        lo_s=float(np.percentile(d, 2.5)), hi_s=float(np.percentile(d, 97.5)))
    out['paired_vs_kine'] = boots

    # stratified: by realised TTL quartile, by whether a reference event exists, and by
    # whether a manoeuvre is already running at t0 -- the last is where the forecast has to
    # add something the instantaneous state does not already say
    q = np.quantile(y[te], [.25, .5, .75])
    grp = np.digitize(y[te], q)
    any_ev = (Y[te] * (np.arange(Y.shape[1])[None, :] < meta.vlen.values[te, None])[:, :, None]
              ).sum(axis=1).max(axis=1) > 0
    ongoing = Y[te, 0, :].max(axis=1) > 0.5
    strat = {}
    for k in ('kine', 'augur', 'oracle'):
        if k not in preds:
            continue
        strat[k] = dict(
            by_ttl_q=[float(np.abs(preds[k][grp == g] - y[te][grp == g]).mean() * 60)
                      for g in range(4)],
            with_event=float(np.abs(preds[k][any_ev] - y[te][any_ev]).mean() * 60),
            without_event=float(np.abs(preds[k][~any_ev] - y[te][~any_ev]).mean() * 60),
            ongoing_at_t0=float(np.abs(preds[k][ongoing] - y[te][ongoing]).mean() * 60),
            quiet_at_t0=float(np.abs(preds[k][~ongoing] - y[te][~ongoing]).mean() * 60))
    out['stratified'] = strat
    out['n_ongoing'] = int(ongoing.sum())
    out['n'] = dict(test=int(len(te)), days=int(len(uday)),
                    with_event=int(any_ev.sum()), ttl_median_min=float(np.median(y[te])),
                    ttl_iqr=[float(np.percentile(y[te], 25)), float(np.percentile(y[te], 75))])
    keep = {k: v for k, v in preds.items() if not k.endswith('_nseed')}
    np.savez_compressed(f'{DATA}/eta/eta_test_preds.npz', y=y[te], days=days,
                        **keep)
    with open(f'{DATA}/eta_results.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('augur'); a.add_argument('--fold', type=int, required=True)
    b = sub.add_parser('eta'); b.add_argument('--variant', required=True)
    b.add_argument('--seed', type=int, default=0)
    sub.add_parser('agg')
    args = ap.parse_args()
    dict(augur=cmd_augur_fold, eta=cmd_eta, agg=cmd_agg)[args.cmd](args)
