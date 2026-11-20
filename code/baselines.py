"""Reference models: a no-event floor and a static-feature multinomial model.

Both emit the same per-step probability tensor as the learned models, so every method is
scored by one code path.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data as D
from metrics import extract
from train import event_scores

DATA = D.DATA
BIN_EDGES = np.array([1, 2, 4, 6, 8], float)      # duration class upper edges, minutes


def _dur_mid(cls):
    lo = np.r_[0.0, BIN_EDGES[:-1]]
    return ((lo + BIN_EDGES) / 2.0)[cls]


def static_predictions(X, M, Y, onset, on_ok, dur, VL, tr, target, nb):
    """One multinomial model per (type, quantity), fitted on training flights only."""
    Ftr = D.summary_features(X[tr], M[tr])
    sc = StandardScaler().fit(Ftr)
    Ztr = sc.transform(Ftr)
    Ft = sc.transform(D.summary_features(X[target], M[target]))

    H = Y.shape[1]
    P = np.zeros((len(target), H, len(D.TYPES)), np.float32)
    any_tr = np.stack([(Y[tr][:, :, k] > 0.5).any(axis=1) for k in range(len(D.TYPES))], 1)
    for k in range(len(D.TYPES)):
        occ = LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced')
        occ.fit(Ztr, any_tr[:, k].astype(int))
        m_on = on_ok[tr, k]
        cls_b = LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced')
        cls_b.fit(Ztr[m_on], onset[tr][m_on, k])
        cls_d = LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced')
        cls_d.fit(Ztr[m_on], dur[tr][m_on, k])

        p = occ.predict_proba(Ft)[:, 1]
        b = np.clip(cls_b.predict(Ft) - 1, 0, nb - 1)
        d = _dur_mid(np.clip(cls_d.predict(Ft), 0, len(BIN_EDGES) - 1))
        for j, i in enumerate(target):
            if p[j] < 0.5:
                continue
            a = int(b[j] * D.BIN_S / D.DT)
            e = min(int(VL[i]), a + int(d[j] * 60 / D.DT))
            P[j, a:e, k] = p[j]
    return P


def main():
    X, M, Y, DUR, VL, meta = D.load()
    sp = D.split_index(meta)
    onset, on_ok, dur, nb = D.targets(Y, DUR, VL)
    tr, va, te = sp['train'], sp['val'], sp['test']

    n_ref = sum(len(extract(Y[i, :int(VL[i]), k] > 0.5))
                for i in te for k in range(len(D.TYPES)))
    floor = dict(tp=0, fp=0, fn=n_ref, precision=float('nan'), recall=0.0, f1=0.0,
                 fa_per_hour=0.0, onset_mae=float('nan'), dur_mae=float('nan'), n_match=0)
    print(f'persistence: {n_ref} reference events, recall 0 by construction')
    with open(f'{DATA}/res_persistence.json', 'w') as fh:
        json.dump(dict(tag='persistence', enc='none', best_thr=0.5, val=floor, test=floor),
                  fh, indent=1)

    Pv = static_predictions(X, M, Y, onset, on_ok, dur, VL, tr, va, nb)
    Pt = static_predictions(X, M, Y, onset, on_ok, dur, VL, tr, te, nb)
    np.savez_compressed(f'{DATA}/pred_static.npz', val=Pv, test=Pt, best_thr=0.5)
    rv, _, _ = event_scores(Pv, Y, VL, va, 0.5)
    rt, _, _ = event_scores(Pt, Y, VL, te, 0.5)
    with open(f'{DATA}/res_static.json', 'w') as fh:
        json.dump(dict(tag='static', enc='logreg-static', best_thr=0.5, val=rv, test=rt),
                  fh, indent=1)
    print('static macro F1', round(rt['macro_f1'], 4),
          {k: round(rt[k]['f1'], 3) for k in D.TYPES})


if __name__ == '__main__':
    main()
