"""Train one configuration and emit probabilities for validation and test."""
import argparse
import json
import time

import numpy as np
import torch

import data as D
from metrics import extract, score
from model import AUGUR, count_params, duration_loss, masked_bce, onset_loss

DATA = D.DATA


def batches(idx, bs, shuffle, rng=None):
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def predict(model, X, M, idx, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for b in batches(idx, bs, False):
            out.append(torch.sigmoid(model(torch.from_numpy(X[b]),
                                           torch.from_numpy(M[b]))['occ']).numpy())
    return np.concatenate(out)


def event_scores(P, Y, VL, idx, thr):
    """Event-level scores; both sides are truncated at the censoring point."""
    K = P.shape[2]
    pred, ref = [[] for _ in range(K)], [[] for _ in range(K)]
    for j, i in enumerate(idx):
        v = int(VL[i])
        for k in range(K):
            pred[k].append(extract(P[j, :v, k] >= thr))
            ref[k].append(extract(Y[i, :v, k] > 0.5))
    res = {name: score(pred[k], ref[k], n_flight=len(idx)) for k, name in enumerate(D.TYPES)}
    res['macro_f1'] = float(np.mean([res[n]['f1'] for n in D.TYPES]))
    res['macro_precision'] = float(np.mean([res[n]['precision'] for n in D.TYPES]))
    res['macro_recall'] = float(np.mean([res[n]['recall'] for n in D.TYPES]))
    return res, pred, ref


def tune_threshold(P, Y, VL, idx, grid=np.arange(0.10, 0.86, 0.05)):
    best, bt = -1.0, 0.5
    for t in grid:
        r, _, _ = event_scores(P, Y, VL, idx, t)
        if r['macro_f1'] > best:
            best, bt = r['macro_f1'], float(t)
    return bt, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', default='tcn', choices=['tcn', 'gru', 'tf'])
    ap.add_argument('--tag', default='tcn')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--ch', type=int, default=56)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--no_static', action='store_true')
    ap.add_argument('--single_task', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    X, M, Y, DUR, VL, meta = D.load()
    sp = D.split_index(meta)
    onset, on_ok, dur, nb = D.targets(Y, DUR, VL)
    mask = D.frame_mask(VL)
    if a.no_static:
        M = np.zeros_like(M)
    tr, va, te = sp['train'], sp['val'], sp['test']

    mtr = mask[tr]
    pos = (Y[tr] * mtr).sum(axis=(0, 1))
    neg = mtr.sum() - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 30.0), dtype=torch.float32)

    model = AUGUR(X.shape[2] + M.shape[2], Y.shape[1], len(D.TYPES), enc=a.enc, ch=a.ch,
                  dropout=a.dropout, n_bin=nb)
    print(f'{a.tag}: {count_params(model):,} parameters', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = ((len(tr) + a.bs - 1) // a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps,
                                                pct_start=0.15)
    best_f1, best_state, best_thr = -1.0, None, 0.5
    t_start = time.time()

    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for b in batches(tr, a.bs, True, rng):
            out = model(torch.from_numpy(X[b]), torch.from_numpy(M[b]))
            loss = masked_bce(out['occ'], torch.from_numpy(Y[b]),
                              torch.from_numpy(mask[b]), pw)
            if not a.single_task:
                vb = torch.from_numpy(on_ok[b])
                loss = loss + 0.5 * onset_loss(out['onset'], torch.from_numpy(onset[b]), vb)
                loss = loss + 0.3 * duration_loss(out['dur'], torch.from_numpy(dur[b]), vb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        sub = va if len(va) <= 8000 else va[rng.choice(len(va), 8000, replace=False)]
        P = predict(model, X, M, sub)
        thr, f1 = tune_threshold(P, Y, VL, sub)
        print(f'  ep {ep + 1:02d} loss {tot / len(tr):.4f} valF1 {f1:.4f} thr {thr:.2f} '
              f'[{time.time() - t_start:.0f}s]', flush=True)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    Pv, Pt = predict(model, X, M, va), predict(model, X, M, te)
    np.savez_compressed(f'{DATA}/pred_{a.tag}.npz', val=Pv, test=Pt,
                        best_thr=best_thr, val_f1=best_f1)
    rv, _, _ = event_scores(Pv, Y, VL, va, best_thr)
    rt, _, _ = event_scores(Pt, Y, VL, te, best_thr)
    out = dict(tag=a.tag, enc=a.enc, params=count_params(model), best_thr=best_thr,
               epochs=a.epochs, seed=a.seed, ch=a.ch, no_static=a.no_static,
               single_task=a.single_task, val=rv, test=rt)
    with open(f'{DATA}/res_{a.tag}.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rt.items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
