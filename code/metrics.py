"""Event-level scoring.

A manoeuvre is a temporally extended object, so frame-wise scores do not answer the
question a controller asks ("is something going to happen, and when").  Predictions are
therefore reduced to events and matched one-to-one with the reference events.
"""
import numpy as np

DT = 10.0


def extract(mask, min_dur=60.0, gap=60.0, t_off=0.0):
    """Contiguous runs of `mask`, bridged over gaps <= `gap` s, shorter than `min_dur` dropped.

    Returns an array of [onset_s, offset_s] relative to t0.
    """
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        return np.zeros((0, 2))
    # bridge short gaps
    brk = np.flatnonzero(np.diff(idx) * DT > gap)
    segs = np.split(idx, brk + 1)
    out = []
    for s in segs:
        a, b = s[0], s[-1]
        if (b - a + 1) * DT >= min_dur:
            out.append((a * DT + t_off, (b + 1) * DT + t_off))
    return np.array(out, float).reshape(-1, 2)


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(pred, ref, tol=300.0, iou_min=0.1):
    """Maximum-cardinality matching on the onset-tolerance graph.

    Pairs are admitted only when the onsets agree within `tol` and the temporal IoU is at
    least `iou_min`; among the admissible pairs the assignment that maximises the number of
    matches is used (Hungarian on the bipartite graph, minimising onset deviation).
    """
    if len(pred) == 0 or len(ref) == 0:
        return 0, len(pred), len(ref), []
    cost = np.full((len(pred), len(ref)), 1e6)
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            d = abs(p[0] - r[0])
            if d <= tol and _iou(p, r) >= iou_min:
                cost[i, j] = d
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(ri, ci) if cost[i, j] < 1e5]
    except Exception:                                    # greedy fallback
        pairs, used_r = [], set()
        for i, j in sorted(np.ndindex(cost.shape), key=lambda t: cost[t]):
            if cost[i, j] >= 1e5 or i in {p[0] for p in pairs} or j in used_r:
                continue
            pairs.append((i, j))
            used_r.add(j)
    return len(pairs), len(pred) - len(pairs), len(ref) - len(pairs), pairs


def score(pred_list, ref_list, tol=300.0, iou_min=0.1, n_flight=1, horizon_h=0.5):
    """Aggregate event counts over a split."""
    tp = fp = fn = 0
    onset_err, dur_err = [], []
    for p, r in zip(pred_list, ref_list):
        a, b, c, pairs = match(p, r, tol, iou_min)
        tp, fp, fn = tp + a, fp + b, fn + c
        for i, j in pairs:
            onset_err.append(abs(p[i, 0] - r[j, 0]) / 60.0)
            dur_err.append(abs((p[i, 1] - p[i, 0]) - (r[j, 1] - r[j, 0])) / 60.0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                fa_per_hour=fp / max(n_flight * horizon_h, 1e-9),
                onset_mae=float(np.mean(onset_err)) if onset_err else np.nan,
                dur_mae=float(np.mean(dur_err)) if dur_err else np.nan,
                n_match=len(onset_err))


def frame_iou(pred, ref):
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union else np.nan


def day_bootstrap(per_flight, days, stat, n_boot=1000, seed=0):
    """Day-block bootstrap: whole test days are resampled, never individual flights."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(days)))
    buckets = {d: np.flatnonzero(np.asarray(days) == d) for d in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([buckets[d] for d in pick])
        vals.append(stat(idx))
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prec_rec_curve(pred_prob, ref_prob, thresholds):
    """Operating-characteristic style sweep at the frame level, for model comparison."""
    out = []
    for t in thresholds:
        out.append(frame_iou(pred_prob >= t, ref_prob >= 0.5))
    return np.array(out)
