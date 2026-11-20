"""Aggregate per-method results with day-block bootstrap intervals.

Every number that appears in the paper is produced here, from the saved probability
tensors, by one code path, so that the tables and the figures cannot disagree.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import data as D
from metrics import extract, match

DATA = D.DATA
METHODS = ['persistence', 'static', 'gru', 'tf', 'tcn_nostatic', 'tcn_single', 'tcn', 'tcn_s1']
LABEL = dict(persistence='no event', static='static logistic', gru='GRU',
             tf='Transformer', tcn_nostatic='TCN, no static covariates',
             tcn_single='TCN, occupancy head only', tcn='AUGUR (full)', tcn_s1='AUGUR, seed 1')
PRIMARY = dict(tol=180.0, iou=0.1)


def tune(P, Y, VL, idx, tol, iou, grid=np.arange(0.10, 0.86, 0.05)):
    """Operating threshold on the validation split, under the reported admissibility.

    The threshold is reselected here rather than taken from training, so that the
    condition used to choose it is the condition used to report the score.  The grid is
    walked from the top and ties are broken towards the higher threshold, i.e. towards the
    more conservative operating point; without this, a method whose probabilities are
    nearly binary (the static reference) would report the bottom of a plateau.
    """
    best, bt = -1.0, 0.5
    K = P.shape[2]
    for t in np.sort(np.asarray(grid))[::-1]:
        f1 = []
        for k in range(K):
            pred = [extract(P[j, :int(VL[i]), k] >= t) for j, i in enumerate(idx)]
            refk = [extract(Y[i, :int(VL[i]), k] > 0.5) for i in idx]
            tp = fp = fn = 0
            for p, r in zip(pred, refk):
                a, b, c, _ = match(p, r, tol, iou)
                tp, fp, fn = tp + a, fp + b, fn + c
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            f1.append(2 * pr * rc / max(pr + rc, 1e-9))
        if float(np.mean(f1)) > best:
            best, bt = float(np.mean(f1)), float(t)
    return bt, best


def counts_for(P, Y, VL, idx, thr, tol, iou):
    """Per-flight TP/FP/FN and matched errors, per type."""
    K = P.shape[2]
    out = {k: dict(tp=[], fp=[], fn=[], on=[], du=[]) for k in range(K)}
    for j, i in enumerate(idx):
        v = int(VL[i])
        for k in range(K):
            p = extract(P[j, :v, k] >= thr)
            r = extract(Y[i, :v, k] > 0.5)
            tp, fp, fn, pairs = match(p, r, tol, iou)
            out[k]['tp'].append(tp)
            out[k]['fp'].append(fp)
            out[k]['fn'].append(fn)
            for a, b in pairs:
                out[k]['on'].append(abs(p[a, 0] - r[b, 0]) / 60.0)
                out[k]['du'].append(abs((p[a, 1] - p[a, 0]) - (r[b, 1] - r[b, 0])) / 60.0)
    return out


def summarise(c, k, hours=1.0):
    tp, fp, fn = np.sum(c[k]['tp']), np.sum(c[k]['fp']), np.sum(c[k]['fn'])
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=pr, recall=rc,
                f1=2 * pr * rc / max(pr + rc, 1e-9),
                fa_per_h=float(fp / max(hours, 1e-9)),
                onset_mae=float(np.mean(c[k]['on'])) if c[k]['on'] else np.nan,
                dur_mae=float(np.mean(c[k]['du'])) if c[k]['du'] else np.nan)


def boot_macro(c, days, n_boot=1000, seed=0):
    """Day-block bootstrap of macro F1, precision and recall."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(days)))
    members = {d: np.flatnonzero(np.asarray(days) == d) for d in uniq}
    f1, pr, rc = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([members[d] for d in pick])
        f, p, r = [], [], []
        for k in c:
            tp = np.sum(np.asarray(c[k]['tp'])[rows])
            fp = np.sum(np.asarray(c[k]['fp'])[rows])
            fn = np.sum(np.asarray(c[k]['fn'])[rows])
            a = tp / max(tp + fp, 1)
            b = tp / max(tp + fn, 1)
            p.append(a); r.append(b); f.append(2 * a * b / max(a + b, 1e-9))
        f1.append(np.mean(f)); pr.append(np.mean(p)); rc.append(np.mean(r))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(f1=q(f1), precision=q(pr), recall=q(rc))


def main():
    X, M, Y, DUR, VL, meta = D.load()
    sp = D.split_index(meta)
    te, va = sp['test'], sp['val']
    days_te = meta.day.values[te]

    rows, store = [], {}
    for tag in METHODS:
        f = f'{DATA}/pred_{tag}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f)
        P = z['test']
        if 'val' in z.files:
            thr, val_f1 = tune(z['val'], Y, VL, va, **PRIMARY)
        else:
            thr, val_f1 = (float(z['best_thr']) if 'best_thr' in z else 0.5), np.nan
        c = counts_for(P, Y, VL, te, thr, **PRIMARY)
        hours = float(np.sum(VL[te]) * 10.0 / 3600.0)
        res = {D.TYPES[k]: summarise(c, k, hours) for k in c}
        res['macro_f1'] = float(np.mean([res[t]['f1'] for t in D.TYPES]))
        res['macro_precision'] = float(np.mean([res[t]['precision'] for t in D.TYPES]))
        res['macro_recall'] = float(np.mean([res[t]['recall'] for t in D.TYPES]))
        res['hours'] = hours
        ci = boot_macro(c, days_te)
        res.update({f'macro_{k}_ci': v for k, v in ci.items()})
        res.update(dict(tag=tag, label=LABEL.get(tag, tag), thr=thr, val_f1=float(val_f1)))
        rows.append(res)
        store[tag] = dict(counts=c, thr=thr)

    df = pd.DataFrame(rows).set_index('tag')
    df.to_csv(f'{DATA}/results_test.csv')
    with open(f'{DATA}/results_test.json', 'w') as fh:
        json.dump({r['tag']: r for r in rows}, fh, indent=1, default=float)

    # paired differences against the full model on identical resamples
    if 'tcn' in store:
        rng = np.random.default_rng(0)
        uniq = np.array(sorted(set(days_te)))
        members = {d: np.flatnonzero(days_te == d) for d in uniq}
        picks = [np.concatenate([members[d] for d in rng.choice(uniq, len(uniq), True)])
                 for _ in range(400)]

        def macro_f1(c, rows):
            out = []
            for k in c:
                tp = np.sum(np.asarray(c[k]['tp'])[rows])
                fp = np.sum(np.asarray(c[k]['fp'])[rows])
                fn = np.sum(np.asarray(c[k]['fn'])[rows])
                a = tp / max(tp + fp, 1)
                b = tp / max(tp + fn, 1)
                out.append(2 * a * b / max(a + b, 1e-9))
            return float(np.mean(out))

        base = store['tcn']['counts']
        diffs = {}
        for tag, s in store.items():
            if tag == 'tcn':
                continue
            d = np.array([macro_f1(base, r) - macro_f1(s['counts'], r) for r in picks])
            diffs[tag] = dict(mean=float(d.mean()),
                              lo=float(np.percentile(d, 2.5)),
                              hi=float(np.percentile(d, 97.5)))
        with open(f'{DATA}/paired_diff.json', 'w') as fh:
            json.dump(diffs, fh, indent=1)

    cols = ['label', 'thr', 'macro_f1', 'macro_precision', 'macro_recall']
    print(df[cols].round(4).to_string())
    for k in D.TYPES:
        print(k, df[[f'{k}']].round(3).to_string() if k in df else '')


if __name__ == '__main__':
    main()
