"""How much does the speed-control label owe to the arrival's future touchdown speed?

The published definition scales the nominal profile by the arrival's own touchdown speed,
which is a quantity observed only at the end of the track. This script recomputes the same
rule with the only scale a forecaster could have known at the anchor --- the ground speed
observed at t0 --- and measures how far the two label sets drift apart.

Both variants are evaluated on the frames after t0, which is the window the model is scored
on, and both inherit the precedence order used for the published labels.
"""
import json

import numpy as np
import pandas as pd

import labels as L

from config import DATA
MONTHS = ['202501', '202502', '202503']
R0, DT = 60.0, 10.0


def prof_table():
    p = pd.read_csv(f'{DATA}/speed_profile.csv')
    edges = np.arange(0.0, 90.0, 2.0)
    # the saved table sits on bin centres 1, 3, ... km; the rules index bin edges
    return np.interp(edges, p.s.values, p.gs.values)


def bin_of(r, prof):
    return np.clip(np.round(r / 2.0).astype(int), 0, len(prof) - 1)


def spd_from_scale(fl, scale, prof, alpha=0.85, margin_kt=5.0, min_dur=60.0,
                   r_lo=8.0, r_hi=62.0, floor=40.0):
    """Same rule as labels.spd, with the profile scaled by an externally supplied factor."""
    gs, r = fl['gs'], fl['r']
    b = bin_of(r, prof)
    nom = prof[b] * scale
    m = (gs < alpha * nom - margin_kt * L.KT) & (r > r_lo) & (r < r_hi) & (gs > floor)
    return L.runs(m, min_dur)


def onsets(mask, min_dur=60.0):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([])
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[brk + 1]])
    keep = []
    for s in starts:
        j = s
        while j < len(mask) and mask[j]:
            j += 1
        if (j - s) * DT >= min_dur:
            keep.append(s)
    return np.array(keep, int)


def match(a, b, tol=180.0):
    """Greedy one-to-one pairing of onset instants within tol seconds."""
    a, b = np.sort(a), np.sort(b)
    used = np.zeros(len(b), bool)
    hit = 0
    for x in a:
        d = np.abs(b - x) * DT
        d[used] = 1e9
        k = int(np.argmin(d)) if len(b) else -1
        if k >= 0 and d[k] <= tol:
            used[k] = True
            hit += 1
    return hit


def main():
    prof = prof_table()
    nfl = 0
    acc = {k: dict(num=0, den=0, ev=0, on=0, hit=0) for k in ('t0', 'fleet', 'fix')}
    ref = dict(on=0, ev=0)
    vref_all, vhat_all, vfix_all = [], [], []

    for m in MONTHS:
        fl = np.load(f'{DATA}/fl_{m}.npy', allow_pickle=True)
        for f in fl:
            r = f['r']
            hit = np.flatnonzero(r <= R0)
            if hit.size == 0:
                continue
            t0 = int(hit[0])
            if t0 < 60 or (len(r) - t0) < 30:
                continue
            nfl += 1

            # published label, with precedence already applied at build time, scored on the
            # same 15 min window the model is evaluated on
            vlen = int(min(90, len(r) - t0))
            a = f['ev_spd'][t0:t0 + vlen].astype(bool)
            oa = onsets(a)
            ref['on'] += len(oa)
            if a.any():
                ref['ev'] += 1

            gs = f['gs']
            b0 = bin_of(np.array([r[t0]]), prof)[0]
            vref_all.append(float(np.median(gs[-4:])))
            if prof[b0] <= 1e-6 or gs[t0] <= 1e-6:
                continue
            excl = (~f['ev_lvl'][t0:t0 + vlen].astype(bool)
                    & ~f['ev_vec'][t0:t0 + vlen].astype(bool))

            # a scale read where the low-altitude speed reports are still usable
            band = (f['alt'] >= 150.0) & (f['alt'] <= 600.0)
            vfix = float(np.median(gs[band][-10:])) if band.any() else prof[0]

            for name, scale in (('t0', gs[t0] / prof[b0]), ('fleet', 1.0),
                                ('fix', vfix / prof[0])):
                alt = spd_from_scale(f, scale, prof)[t0:t0 + vlen] & excl
                n = min(len(a), len(alt))
                aa, bb = a[:n], alt[:n]
                acc[name]['num'] += int((aa & bb).sum())
                acc[name]['den'] += int((aa | bb).sum())
                if bb.any():
                    acc[name]['ev'] += 1
                ob = onsets(bb)
                acc[name]['on'] += len(ob)
                acc[name]['hit'] += match(oa, ob)

            vfix_all.append(vfix)
            vhat_all.append(float(gs[t0] * prof[0] / prof[b0]))

        del fl

    vref_all = np.array(vref_all)
    vhat_all = np.array(vhat_all)
    out = dict(
        flights=int(nfl),
        published=dict(share_with_event=round(ref['ev'] / nfl, 4), onsets=int(ref['on'])),
        frame_iou={k: round(acc[k]['num'] / max(acc[k]['den'], 1), 4) for k in acc},
        share_with_event={k: round(acc[k]['ev'] / nfl, 4) for k in acc},
        onsets={k: int(acc[k]['on']) for k in acc},
        onset_precision={k: round(acc[k]['hit'] / max(acc[k]['on'], 1), 4) for k in acc},
        onset_recall={k: round(acc[k]['hit'] / max(ref['on'], 1), 4) for k in acc},
        corr_vref_vhat=round(float(np.corrcoef(vref_all, vhat_all)[0, 1]), 4),
        vref_median=round(float(np.median(vref_all)), 2),
        vhat_median=round(float(np.median(vhat_all)), 2),
        vref_iqr=[round(float(np.percentile(vref_all, 25)), 2),
                  round(float(np.percentile(vref_all, 75)), 2)],
        vfix_median=round(float(np.median(vfix_all)), 2),
    )
    json.dump(out, open(f'{DATA}/spd_alt.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
