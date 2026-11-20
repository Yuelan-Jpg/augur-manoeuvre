"""Step 1: extract terminal-area arrival flights from the Chengdu surveillance feed.

Source: track_message_YYYYMM_clean.parquet (ATCAS feed, ZUUU ACC), 15 s surveillance
reports.  Coordinates are DDMM.MM (value/1e4 -> deg + decimal minutes); altitude is
reported in 10-ft increments.  The scan is streaming and columnar (numpy, int keys) so
that a month of ~35 M reports fits in memory; 70 min of track preceding touchdown is
written out.

Fixed with respect to the first pass: the day index is permuted together with the sorted
arrays (it was applied in the original order, which merged different days into one
"flight"), the calendar date is derived with numpy datetime64 arithmetic rather than an
integer-to-D cast, and a flight is cut whenever the report stream pauses for more than
15 min so that two rotations of the same flight number on one day stay separate.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from config import TRACKS as RAW, DATA as OUT
MONTHS = ['202501', '202502', '202503']

ZUUU = (30.5785, 103.9471)
ZUTF = (30.3125, 104.4413)
LL_LAT = (288000, 320000)
LL_LON = (1023000, 1058000)
WINDOW_MIN = 70
SEG_GAP = 900          # s; a longer pause starts a new flight segment
CALLSIGN = r'^[A-Z]{3}[0-9]{1,4}[A-Z]?$'


def ddmm_to_deg(v):
    v = np.asarray(v, dtype='float64') / 10000.0
    deg = np.floor(v)
    return deg + (v - deg) * 100.0 / 60.0


def haversine(lat, lon, ref):
    R = 6371.0088
    p1 = np.radians(lat)
    p2 = np.radians(ref[0])
    dp = p2 - p1
    dl = np.radians(lon - ref[1])
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def scan_month(m):
    p = f'{RAW}/track_message_{m}_clean.parquet'
    d = ds.dataset(p, format='parquet')
    f = ((ds.field('latitude') >= LL_LAT[0]) & (ds.field('latitude') <= LL_LAT[1]) &
         (ds.field('longitude') >= LL_LON[0]) & (ds.field('longitude') <= LL_LON[1]))
    cols = ['call_sign', 'log_timestamp', 'altitude', 'longitude', 'latitude',
            'ground_speed', 'vector', 'vertical_rate']
    sc = d.scanner(columns=cols, filter=f, batch_size=2_000_000)
    buckets = {k: [] for k in ['cs_id', 'ts', 'alt', 'lat', 'lon', 'gs', 'hdg', 'vr']}
    codes, n_raw = {}, 0
    for b in sc.to_batches():
        n_raw += b.num_rows
        b = b.filter(pc.match_substring_regex(pc.utf8_upper(b.column('call_sign')), CALLSIGN))
        if b.num_rows == 0:
            continue
        cs = b.column('call_sign').to_pylist()
        ids = np.empty(len(cs), np.int32)
        for i, c in enumerate(cs):
            c = c.strip()
            v = codes.get(c)
            if v is None:
                v = codes[c] = len(codes)
            ids[i] = v
        buckets['cs_id'].append(ids)
        buckets['ts'].append(b.column('log_timestamp').to_numpy(zero_copy_only=False)
                             .astype('datetime64[s]').astype(np.int64))
        buckets['alt'].append(b.column('altitude').to_numpy().astype(np.float32))
        buckets['lat'].append(ddmm_to_deg(b.column('latitude').to_numpy()).astype(np.float32))
        buckets['lon'].append(ddmm_to_deg(b.column('longitude').to_numpy()).astype(np.float32))
        buckets['gs'].append(b.column('ground_speed').to_numpy().astype(np.float32))
        buckets['hdg'].append(b.column('vector').to_numpy().astype(np.float32))
        buckets['vr'].append(b.column('vertical_rate').to_numpy().astype(np.float32))
    arrs = {k: np.concatenate(v) for k, v in buckets.items()}
    del buckets
    print(f'  {m}: {n_raw:,} reports in box -> {len(arrs["ts"]):,} airline reports '
          f'({len(codes):,} callsigns)', flush=True)
    return arrs, list(codes)


def build_table(arrs, names, m):
    d_uuu = haversine(arrs['lat'], arrs['lon'], ZUUU)
    d_utf = haversine(arrs['lat'], arrs['lon'], ZUTF)
    n = len(arrs['ts'])
    days0 = int(np.datetime64(f'{m[:4]}-{m[4:]}', 's').astype(np.int64))
    ts = arrs['ts']
    # reports logged a little before midnight of the 1st would give -1; clamp to day 0
    day_idx = np.clip((ts - days0) // 86400, 0, 39)
    key = arrs['cs_id'].astype(np.int64) * 100000 + day_idx

    order = np.lexsort((ts, key))
    for k in ['cs_id', 'alt', 'lat', 'lon', 'gs', 'hdg', 'vr']:
        arrs[k] = arrs[k][order]
    d_uuu, d_utf = d_uuu[order], d_utf[order]
    ts = ts[order]
    key = key[order]
    day_idx = day_idx[order]

    # segments: new callsign-day, or a surveillance pause longer than SEG_GAP
    new = np.empty(n, bool)
    new[0] = True
    new[1:] = (key[1:] != key[:-1]) | (ts[1:] - ts[:-1] > SEG_GAP)
    seg = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n] - 1
    cnt = ends - starts + 1

    tail_idx = np.stack([np.maximum(ends - 3, starts), np.maximum(ends - 2, starts),
                         np.maximum(ends - 1, starts), ends], axis=1)
    tail_alt = arrs['alt'][tail_idx].min(axis=1)
    tail_d = np.minimum(d_uuu, d_utf)[tail_idx].min(axis=1)
    dur = ts[ends] - ts[starts]
    good = ((tail_alt <= 120) & (tail_d <= 6.0) & (arrs['alt'][starts] >= 150) &
            (cnt >= 80) & (dur >= 900))

    t_end_row = np.repeat(ts[ends], cnt)
    in_win = ts >= (t_end_row - WINDOW_MIN * 60)
    keep_rows = np.flatnonzero(np.repeat(good, cnt) & in_win)
    print(f'  segments {len(starts):,} -> arrivals {int(good.sum()):,}, '
          f'{len(keep_rows):,} reports', flush=True)

    # flight identifiers, built once over the (callsign, day) pairs actually kept
    seg_rows = seg[keep_rows]
    seg_cs = arrs['cs_id'][starts][seg_rows]
    seg_day = day_idx[starts][seg_rows]
    ukey = seg_cs.astype(np.int64) * 1000 + seg_day
    uniq, inv = np.unique(ukey, return_inverse=True)
    day_str = np.array([str(np.datetime64(days0, 's') + np.timedelta64(int(d), 'D'))
                        for d in range(40)], dtype='<U10')
    nm = np.asarray(names, dtype=object)[(uniq // 1000).astype(int)].astype(str)
    uid = np.char.add(np.char.add(nm, '_'), day_str[(uniq % 1000).astype(int)])
    flight_id = uid[inv]

    return pa.table({
        'flight': pa.array(flight_id),
        'ts': pa.array(ts[keep_rows]).cast(pa.timestamp('s')),
        'alt': pa.array(arrs['alt'][keep_rows]),
        'lat': pa.array(arrs['lat'][keep_rows]),
        'lon': pa.array(arrs['lon'][keep_rows]),
        'gs': pa.array(arrs['gs'][keep_rows]),
        'hdg': pa.array(arrs['hdg'][keep_rows]),
        'vr': pa.array(arrs['vr'][keep_rows]),
        'airport': pa.array(np.where(d_uuu[keep_rows] <= d_utf[keep_rows], 'ZUUU', 'ZUTF')),
    })


def main():
    os.makedirs(OUT, exist_ok=True)
    for m in MONTHS:
        arrs, names = scan_month(m)
        tb = build_table(arrs, names, m)
        pq.write_table(tb, f'{OUT}/arr_{m}.parquet')
        print(f'  {m}: {len(tb):,} rows written', flush=True)
        del arrs, tb


if __name__ == '__main__':
    main()
