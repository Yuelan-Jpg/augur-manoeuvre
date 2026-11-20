"""Appendix A: how the definitions respond to their own parameters.

The label is recomputed on a random 4,000-arrival subsample of the analysis sample, with
one parameter moved and the others held at the values of Section 3.2.  Both quantities are
measured on the forecast window the model is scored on, and the precedence HOLD > VEC >
LVL > SPD is applied as in the paper, so the nominal row reproduces the prevalence of
Table 3 rather than the raw output of one rule in isolation.
"""
import json

import numpy as np

import labels

DT = 10.0
HOR = 90
from config import DATA
NMAX = 4000
TYPES = ['VEC', 'LVL', 'SPD']


def label_with(fl, prof, vref, over):
    """label_all with per-type parameter overrides."""
    ev = dict(VEC=labels.vec(fl, **over.get('VEC', {})),
              HOLD=labels.hold(fl, **over.get('HOLD', {})),
              LVL=labels.lvl(fl, **over.get('LVL', {})),
              SPD=labels.spd(fl, vref, prof, **over.get('SPD', {})))
    ev['SPD'] &= ~ev['LVL'] & ~ev['VEC'] & ~ev['HOLD']
    ev['LVL'] &= ~ev['VEC'] & ~ev['HOLD']
    ev['VEC'] &= ~ev['HOLD']
    return ev


def measure(fls, prof, i0s, vl, vrefs, typ, over):
    prev, cnt = [], []
    for fl, i0, v, vr in zip(fls, i0s, vl, vrefs):
        m = label_with(fl, prof, vr, over)[typ][i0:i0 + v]
        prev.append(bool(m.any()))
        cnt.append(len(labels.episodes(m, gap=60.0, min_dur=60.0)))
    return float(np.mean(prev)), float(np.mean(cnt))


def main():
    fls, i0s, vl, vrefs = [], [], [], []
    for m in ['202501', '202502', '202503']:
        for fl in np.load(f'{DATA}/fl_{m}.npy', allow_pickle=True):
            r = fl['r']
            n = len(r)
            hit = np.flatnonzero(r <= 60.0)
            if hit.size == 0:
                continue
            i0 = int(hit[0])
            if i0 < 60 or (n - i0) < 30:
                continue
            fls.append(fl)
            i0s.append(i0)
            vl.append(int(min(HOR, n - i0)))
            vrefs.append(labels.touchdown_speed(fl))
    print(f'{len(fls)} arrivals in the analysis sample', flush=True)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(fls), min(NMAX, len(fls)), replace=False)
    fls = [fls[i] for i in pick]
    i0s = [i0s[i] for i in pick]
    vl = [vl[i] for i in pick]
    vrefs = [vrefs[i] for i in pick]
    prof = labels.gs_profile([f for f in np.load(f'{DATA}/fl_202501.npy',
                                                allow_pickle=True)])

    rows = []
    for typ in TYPES:
        p, c = measure(fls, prof, i0s, vl, vrefs, typ, {})
        rows.append(dict(type=typ, parameter='nominal', value='-', prevalence=p,
                         episodes=c))
        print(f'{typ} nominal  prevalence {p:.3f}  mean episodes {c:.2f}', flush=True)

    grid = {
        'VEC': [('dev_min', [35.0, 45.0]), ('r_onset', [20.0, 30.0]),
                ('min_dur', [60.0, 120.0]), ('r_hi', [40.0, 50.0])],
        'LVL': [('drop', [900.0, 1500.0]), ('min_dur', [60.0, 120.0]),
                ('rate', [50.0, 150.0])],
        'SPD': [('alpha', [0.80, 0.90]), ('margin_kt', [3.0, 8.0]),
                ('r_lo', [5.0, 12.0])],
    }
    for typ, items in grid.items():
        for name, vals in items:
            for v in vals:
                p, c = measure(fls, prof, i0s, vl, vrefs, typ, {typ: {name: v}})
                rows.append(dict(type=typ, parameter=name, value=str(v), prevalence=p,
                                 episodes=c))
                print(f'{typ} {name}={v}  prevalence {p:.3f}  mean episodes {c:.2f}',
                      flush=True)

    hp = {}
    for turn in (240.0, 300.0, 360.0):
        for rad in (3.0, 5.0, 8.0):
            n = sum(1 for f in fls if labels.hold(f, turn_min=turn, radius=rad).any())
            hp[f'turn={turn:g},radius={rad:g}'] = n / len(fls)
            print(f'HOLD turn={turn:g} radius={rad:g}  share {n / len(fls):.4f}', flush=True)
    json.dump(dict(rows=rows, hold=hp), open(f'{DATA}/sensitivity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
