"""Appendix A, second half: how the reported score moves with the matching tolerance.

The admissibility conditions of Section 5.1 are a design choice, so the same predictions
are scored under a grid of them and the movement is reported rather than asserted.
"""
import json

import numpy as np

import data as D
from metrics import extract, match

DATA = D.DATA
TOL = [60.0, 120.0, 180.0, 300.0, 600.0]
IOU = [0.0, 0.1, 0.3, 0.5]


def run(tag='tcn'):
    X, M, Y, DUR, VL, meta = D.load()
    te = D.split_index(meta)['test']
    z = np.load(f'{DATA}/pred_{tag}.npz')
    P = z['test']
    thr = float(z['best_thr'])

    # events do not depend on the tolerance, so extract once
    pred = [[extract(P[j, :int(VL[i]), k] >= thr) for k in range(len(D.TYPES))]
            for j, i in enumerate(te)]
    ref = [[extract(Y[i, :int(VL[i]), k] > 0.5) for k in range(len(D.TYPES))]
           for i in te]

    out = {}
    for tol in TOL:
        for iou in IOU:
            tp = np.zeros(len(D.TYPES))
            fp = np.zeros(len(D.TYPES))
            fn = np.zeros(len(D.TYPES))
            for j in range(len(te)):
                for k in range(len(D.TYPES)):
                    a, b, c, _ = match(pred[j][k], ref[j][k], tol, iou)
                    tp[k] += a
                    fp[k] += b
                    fn[k] += c
            pr = tp / np.maximum(tp + fp, 1)
            rc = tp / np.maximum(tp + fn, 1)
            f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
            out[f'{tol:g}|{iou:g}'] = dict(
                macro=float(f1.mean()),
                **{t: float(f1[k]) for k, t in enumerate(D.TYPES)},
                precision=float(pr.mean()), recall=float(rc.mean()))
            print(f'tol {tol:5.0f}  IoU {iou:.1f}  macro {f1.mean():.3f}  '
                  + ' '.join(f'{t} {f1[k]:.3f}' for k, t in enumerate(D.TYPES)), flush=True)
    json.dump(out, open(f'{DATA}/tol_sensitivity.json', 'w'), indent=1)


if __name__ == '__main__':
    run()
