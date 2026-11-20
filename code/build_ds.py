"""Step 3: build the forecasting dataset.

Anchor.  t0 is the first surveillance report inside 60 km of the touchdown reference: an
event that is observable at t0 and, unlike "20 min before touchdown", independent of how
long the arrival subsequently spends in the terminal area.  Conditioning the sample on
total track length would condition it on vectoring, because a vectored arrival stays in
the area longer and therefore produces a longer track.

Inputs.  The 10 min that end at t0 (60 samples at 10 s).
Targets.  Manoeuvre occupancy on every 10 s frame in the 15 min after t0, right-censored
at touchdown: `vlen` records how many frames are observed, and frames beyond it are masked
in the loss and in scoring.

Splits are by calendar date: train 1 Jan - 28 Feb, validation 1-15 Mar, test 16-31 Mar.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import DATA, RAW, read_parquet
DT = 10.0
R0 = 60.0                 # km, anchor range to touchdown
NH = 60                   # 10 min of history
NW = 90                   # 15 min of target frames
MIN_FUT = 30              # at least 5 min of observed future
TYPES = ['VEC', 'LVL', 'SPD']   # holding never occurs inside 60 km in this sample
MONTHS = ['202501', '202502', '202503']
DUR_EDGES = np.array([1, 2, 4, 6, 8], float)


def flight_times():
    out = f'{DATA}/flight_times.parquet'
    try:
        return read_parquet(out)
    except Exception:
        parts = []
        for m in MONTHS:
            d = read_parquet(f'{DATA}/arr_{m}.parquet', columns=['flight', 'ts'])
            parts.append(d.groupby('flight').ts.agg(['min', 'max']).reset_index())
        t = pd.concat(parts).groupby('flight').agg({'min': 'min', 'max': 'max'}).reset_index()
        t.to_parquet(out, index=False)
        return t


def metar_table():
    cols = ['cccc', 'observationTime', 'meanWindDirection', 'meanWindSpeed',
            'windGustSpeed', 'minVisibility', 'qnh', 'temperature']
    m = pq.read_table(f'{RAW}/metar_2025.parquet', columns=cols).to_pandas()
    m = m[m.cccc.isin(['ZUUU', 'ZUTF'])].copy()
    m['ts'] = pd.to_datetime(m['observationTime'])
    for c in cols[2:]:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    return m.dropna(subset=['ts']).sort_values(['cccc', 'ts'])


def rolling(a, w):
    return pd.Series(a).rolling(w, min_periods=1).mean().values


def build():
    ft = flight_times().set_index('flight')
    met = metar_table()
    met_by = {k: g.reset_index(drop=True) for k, g in met.groupby('cccc')}
    ft2 = ft.copy()
    ft2['bin'] = ft2['max'].dt.floor('10min')
    demand = ft2.groupby('bin').size().rename('demand')

    X, M, Y, DUR, VL, rows = [], [], [], [], [], []
    for m in MONTHS:
        fl = np.load(f'{DATA}/fl_{m}.npy', allow_pickle=True)
        for f in fl:
            fid = str(f['flight'])
            r = f['r']
            n = len(r)
            hit = np.flatnonzero(r <= R0)
            if hit.size == 0:
                continue
            t0 = int(hit[0])
            if t0 < NH or (n - t0) < MIN_FUT:
                continue
            vlen = int(min(NW, n - t0))
            h = slice(t0 - NH, t0)
            s, alt, gs, vs, x, rel, turn = (f['s'], f['alt'], f['gs'], f['vs'],
                                            f['x'], f['rel'], f['turn'])
            X.append(np.column_stack([
                s[h] / 100.0, alt[h] / 1000.0, gs[h] / 100.0, vs[h] / 1000.0,
                np.abs(x[h]) / 5.0, rel[h] / 30.0, turn[h] / 5.0,
                np.gradient(alt, DT)[h] / 1000.0, np.gradient(s, DT)[h] / 100.0,
                np.gradient(gs, DT)[h] / 5.0, np.abs(np.gradient(x, DT))[h] / 20.0,
                rolling(np.abs(x), 30)[h] / 5.0, rolling(np.abs(turn), 12)[h] / 5.0,
                alt[h] / 3000.0 - 1.0,
            ]).astype(np.float32))

            t0_abs = ft.loc[fid, 'max'] - pd.Timedelta(seconds=(n - 1 - t0) * DT)
            ap = str(f['airport'])
            mt = met_by[ap]
            j = mt.ts.searchsorted(t0_abs, side='right') - 1
            mt = mt.iloc[max(0, j)]
            wd = np.radians(mt.meanWindDirection if pd.notna(mt.meanWindDirection) else 0.0)
            ws = mt.meanWindSpeed if pd.notna(mt.meanWindSpeed) else 0.0
            rh = np.radians(float(f['rwy_hdg']))
            tod = t0_abs.hour + t0_abs.minute / 60.0
            stat = np.array([np.sin(wd - rh) * ws / 10.0, np.cos(wd - rh) * ws / 10.0,
                             (mt.windGustSpeed if pd.notna(mt.windGustSpeed) else 0.0) / 10.0,
                             (mt.minVisibility if pd.notna(mt.minVisibility) else 8000.0) / 8000.0,
                             ((mt.qnh if pd.notna(mt.qnh) else 1013.0) - 1013.0) / 10.0,
                             (mt.temperature if pd.notna(mt.temperature) else 20.0) / 20.0,
                             float(ap == 'ZUTF'), float(f['rwy_hdg'] > 90),
                             np.sin(2 * np.pi * tod / 24), np.cos(2 * np.pi * tod / 24),
                             float(demand.get(t0_abs.floor('10min'), 0)) / 10.0], np.float32)
            M.append(np.tile(stat, (NH, 1)))

            lab = np.zeros((NW, len(TYPES)), np.float32)
            dur = np.zeros((len(TYPES), 6), np.float32)
            for k, name in enumerate(TYPES):
                mk = f['ev_' + name.lower()][t0:t0 + vlen]
                lab[:vlen, k] = mk
                if mk.any():
                    idx = np.flatnonzero(mk)
                    d = (idx[-1] - idx[0] + 1) * DT / 60.0
                    dur[k, np.digitize(d, DUR_EDGES)] = 1.0
            Y.append(lab)
            DUR.append(dur)
            VL.append(vlen)
            rows.append(dict(flight=fid, day=t0_abs.strftime('%Y-%m-%d'), t0=t0_abs,
                             airport=ap, vlen=vlen, vlen_min=vlen * DT / 60.0))
        del fl

    X = np.stack(X); M = np.stack(M); Y = np.stack(Y); DUR = np.stack(DUR)
    VL = np.array(VL, np.int16)
    np.save(f'{DATA}/X.npy', X); np.save(f'{DATA}/M.npy', M)
    np.save(f'{DATA}/Y.npy', Y); np.save(f'{DATA}/DUR.npy', DUR)
    np.save(f'{DATA}/vlen.npy', VL)
    pd.DataFrame(rows).to_parquet(f'{DATA}/ds_meta.parquet', index=False)
    print('X', X.shape, 'M', M.shape, 'Y', Y.shape, 'vlen median', np.median(VL))
    occ = np.array([[(Y[i, :VL[i], k] > 0.5).any() for k in range(len(TYPES))]
                    for i in range(len(VL))])
    print('share of flights with an onset', dict(zip(TYPES, occ.mean(0).round(3))))
    print('frame occupancy', dict(zip(TYPES, [
        float(np.sum([Y[i, :VL[i], k].sum() for i in range(len(VL))]) / VL.sum())
        for k in range(len(TYPES))])))


if __name__ == '__main__':
    build()
