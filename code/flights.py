"""Step 2: per-flight resampling and the runway-anchored frame used by the labeller.

Each arrival is resampled on a 10 s grid and described by
    r     great-circle range to the touchdown reference (km)
    s     signed along-track distance in the landing direction (km)
    x     signed cross-track offset from the extended centreline (km, +right)
    dev   angle between ground track and the bearing to the touchdown reference (deg)
All kinematics are differentiated from the positions; the reported ground-speed and
vertical-rate fields are quantised and used for quality assurance only.

Outputs are written one month at a time (fl_YYYYMM.npy + fl_meta_YYYYMM.parquet).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import labels

from config import DATA, read_parquet
MONTHS = ['202501', '202502', '202503']
DT = 10.0
R_EARTH = 6371.0088
KEEP = ('flight', 'airport', 't', 'lat', 'lon', 'alt', 'gs', 'vs', 's', 'x', 'rel',
        'turn', 'r', 'dev', 'rwy_hdg', 'ev_vec', 'ev_hold', 'ev_lvl', 'ev_spd')


def iter_month(m):
    d = read_parquet(f'{DATA}/arr_{m}.parquet')
    d['flight'] = d['flight'].astype('category')
    return d.sort_values(['flight', 'ts'], kind='mergesort')


def smooth(a, w):
    if w <= 1 or len(a) < w:
        return a
    pad = w // 2
    ap = np.pad(a, (pad, pad), mode='edge')
    return np.convolve(ap, np.ones(w) / w, mode='valid')[:len(a)]


def bearing_to(lat, lon, lat1, lon1):
    """Initial bearing from each point to the reference, degrees clockwise from north."""
    p1, p2 = np.radians(lat), np.radians(lat1)
    dl = np.radians(lon1 - lon)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def medf(a, k=3):
    """Median filter to remove single-report position jumps before differentiation."""
    if len(a) < k:
        return a
    return median_filter(a, size=k, mode='nearest')


def process(g):
    g = g.drop_duplicates(subset='ts').sort_values('ts')
    if len(g) < 40:
        return None
    t0, t1 = g.ts.iloc[0], g.ts.iloc[-1]
    dur = (t1 - t0).total_seconds()
    if dur < 600:
        return None
    tg = (g.ts - t0).dt.total_seconds().values
    tt = np.arange(0.0, dur + 1e-9, DT)
    lat = smooth(medf(np.interp(tt, tg, g.lat.values), 3), 3)
    lon = smooth(medf(np.interp(tt, tg, g.lon.values), 3), 3)
    alt = smooth(medf(np.interp(tt, tg, g.alt.values) * 10.0, 3), 5)

    lat0, lon0 = lat[-1], lon[-1]
    e = np.radians(lon - lon0) * R_EARTH * 1000.0 * np.cos(np.radians(lat0))
    n = np.radians(lat - lat0) * R_EARTH * 1000.0
    r = np.hypot(e, n) / 1000.0
    de, dn = np.gradient(e, DT), np.gradient(n, DT)
    gs = smooth(medf(np.hypot(de, dn), 3), 3)
    hdg = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    vs = smooth(medf(np.gradient(alt, DT) * 60.0, 3), 3)

    tail = max(0, len(hdg) - 12)
    rwy_hdg = float(np.median(hdg[tail:])) if len(hdg) > 12 else float(hdg[-1])
    rh = np.radians(rwy_hdg)
    # the aircraft sits behind the touchdown point, i.e. opposite to the landing course
    along = -(n * np.cos(rh) + e * np.sin(rh))
    cross = e * np.cos(rh) - n * np.sin(rh)
    s = along / 1000.0
    s = s - s[-1]
    x = cross / 1000.0

    rel = (hdg - rwy_hdg + 180.0) % 360.0 - 180.0
    turn = np.degrees(np.gradient(np.unwrap(np.radians(rel)), DT))
    dev = np.abs((hdg - bearing_to(lat, lon, lat0, lon0) + 180.0) % 360.0 - 180.0)

    f32 = np.float32
    return dict(flight=str(g.flight.iloc[0]), airport=str(g.airport.iloc[0]),
                t=np.arange(len(tt)) * DT - DT * (len(tt) - 1),
                lat=lat.astype(f32), lon=lon.astype(f32), alt=alt.astype(f32),
                gs=gs.astype(f32), vs=vs.astype(f32), s=s.astype(f32), x=x.astype(f32),
                rel=rel.astype(f32), turn=turn.astype(f32), r=r.astype(f32),
                dev=dev.astype(f32), rwy_hdg=np.float32(rwy_hdg))


def process_month(m):
    df = iter_month(m)
    out = []
    for fid, g in df.groupby('flight', sort=False, observed=True):
        fl = process(g)
        if fl is not None:
            out.append(fl)
    del df
    return out


def slim(fl):
    d = {}
    for k in KEEP:
        if k not in fl:
            continue
        v = fl[k]
        d[k] = str(v) if k in ('flight', 'airport') else np.asarray(v)
    return d


def main():
    # population speed profile from the training months only (quarter of the flights)
    sub = process_month('202501')[::4] + process_month('202502')[::4]
    prof = labels.gs_profile(sub)
    np.save(f'{DATA}/gs_profile.npy', prof)
    print('speed profile (m/s) at 0/20/40/60/80 km:',
          prof[[0, 10, 20, 30, 40]].round(1).tolist())
    del sub

    for m in MONTHS:
        fls = process_month(m)
        rows = []
        for fl in fls:
            ev = labels.label_all(fl, prof)
            for k, v in ev.items():
                fl['ev_' + k.lower()] = v
            rows.append(dict(flight=fl['flight'], airport=fl['airport'],
                             rwy_hdg=float(fl['rwy_hdg']), n=len(fl['t']),
                             **{f'n_{k.lower()}': int(v.sum()) for k, v in ev.items()}))
        np.save(f'{DATA}/fl_{m}.npy', np.array([slim(f) for f in fls], dtype=object),
                allow_pickle=True)
        meta = pd.DataFrame(rows)
        meta.to_parquet(f'{DATA}/fl_meta_{m}.parquet', index=False)
        with_ev = {k: round(float((meta[f'n_{k.lower()}'] > 0).mean()), 3)
                   for k in labels.TYPES}
        print(f'{m}: {len(fls):,} flights; share with an event {with_ev}', flush=True)
        del fls, meta


if __name__ == '__main__':
    main()
