"""Time-to-land (TTL) per arrival, measured from t0 to the end of the contiguous track.

`flight_times.parquet` stores the first and last report of a flight id, but a small number
of call signs are reused on consecutive days and the two tracks are stored under one id,
separated by a gap of roughly 24 h.  Taking the last report as touchdown then yields a TTL
of ~1480 min.  Here the track is cut at the first gap longer than GAP_MAX, so the target is
the end of the same contiguous surveillance segment that t0 belongs to.
"""
import numpy as np
import pandas as pd

from config import DATA, read_parquet
GAP_MAX = 90.0          # s, a break longer than this starts a new segment
MONTHS = ['202501', '202502', '202503']


def main():
    meta = read_parquet(f'{DATA}/ds_meta.parquet')
    frames = []
    for mth in MONTHS:
        a = read_parquet(f'{DATA}/arr_{mth}.parquet', columns=['flight', 'ts'])
        frames.append(a)
    a = pd.concat(frames, ignore_index=True)
    a['ts'] = a['ts'].astype('datetime64[ms]').astype('int64') // 1000        # s
    a = a.sort_values(['flight', 'ts'], kind='mergesort')
    print('reports', len(a), 'flights', a.flight.nunique(), flush=True)

    fl = a.flight.values
    ts = a.ts.values
    newf = np.r_[True, fl[1:] != fl[:-1]]
    gap = np.r_[np.inf, np.diff(ts).astype(float)]
    gap[newf] = np.inf
    # segment id increments at every flight change or long gap
    seg = np.cumsum(newf | (gap > GAP_MAX))

    # t0 lookup: index of the report at (or just after) t0 within the same flight
    t0 = meta.t0.values.astype('datetime64[ms]').astype('int64') // 1000
    key = pd.Series(np.arange(len(meta)), index=meta.flight.values)
    # per (flight, t0) find report index via searchsorted on the sorted array
    first = np.searchsorted(fl, meta.flight.values, side='left')
    last = np.searchsorted(fl, meta.flight.values, side='right')
    idx = np.empty(len(meta), dtype=np.int64)
    for i, (f0, f1, t) in enumerate(zip(first, last, t0)):
        j = np.searchsorted(ts[f0:f1], t, side='left')
        j = min(j, f1 - f0 - 1)
        idx[i] = f0 + j

    # end of the segment that contains idx
    seg_of = seg[idx]
    seg_end = np.empty(len(meta), dtype=np.int64)
    # last index of each segment
    order = np.argsort(seg, kind='mergesort')
    seg_sorted = seg[order]
    seg_last = np.r_[order[np.r_[seg_sorted[1:] != seg_sorted[:-1], True]]]
    lut = pd.Series(np.arange(len(seg_last)), index=seg_sorted[np.r_[True, seg_sorted[1:] != seg_sorted[:-1]]])
    seg_end = seg_last[lut.reindex(seg_of).values]

    ttl = (ts[seg_end] - ts[idx]) / 60.0
    out = pd.DataFrame(dict(flight=meta.flight.values, day=meta.day.values,
                            ttl_min=ttl, n_seg=seg_end - idx + 1))
    out.to_parquet(f'{DATA}/ttl.parquet', index=False)
    print(out.ttl_min.describe([.01, .05, .25, .5, .75, .95, .99]).round(2).to_string())
    print('negative', (out.ttl_min < 0).sum(), '>60min', (out.ttl_min > 60).sum())
    m = meta.merge(out, on=['flight', 'day'])
    d = m.ttl_min - m.vlen_min
    print('ttl - vlen_min: min %.2f  p1 %.2f  median %.2f' % (d.min(), d.quantile(.01), d.median()))
    print('violations (<-0.2):', (d < -0.2).sum())


if __name__ == '__main__':
    main()
