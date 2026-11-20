"""Trajectory-observable manoeuvre definitions.

Each event is a *departure and recovery* around the aircraft's own path, not a departure
from a population average.  A population reference puts half the fleet permanently on one
side of the line, so a level rule applied to it fires continuously instead of at an onset;
departure-recovery rules fire once per manoeuvre, which is what a forecasting target needs.

Coordinates: r is the great-circle range to the touchdown reference (km), x the signed
cross-track offset (km), dev the angle between the ground track and the bearing to the
touchdown reference (deg), vs the vertical speed (ft/min), gs the ground speed (m/s).
"""
import numpy as np

DT = 10.0
KT = 0.5144
TYPES = ['VEC', 'HOLD', 'LVL', 'SPD']
PROF_EDGES = np.arange(0.0, 90.0, 2.0)      # range grid for the population speed profile


def runs(mask, min_dur):
    """Keep only True runs at least `min_dur` seconds long."""
    n = len(mask)
    out = np.zeros(n, bool)
    need = int(round(min_dur / DT))
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= need:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def episodes(mask, gap=120.0, min_dur=60.0):
    """Index ranges of separated episodes, after bridging dropouts shorter than `gap`."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) * DT > gap)
    out = []
    for s in np.split(idx, brk + 1):
        if (s[-1] - s[0] + 1) * DT >= min_dur:
            out.append((int(s[0]), int(s[-1])))
    return out


def vec(fl, dev_min=40.0, dev_rel=15.0, r_onset=25.0, r_hi=45.0, min_dur=90.0):
    """Close-in vectoring: a fresh departure from the runway axis, later recaptured.

    The onset must fall inside `r_onset`, which selects the sequencing manoeuvres flown
    within about 25 km of touchdown rather than the routine downwind leg flown by almost
    every arrival.
    """
    dev, r = fl['dev'], fl['r']
    cand = (dev > dev_min) & (r < r_hi) & (r > 2.0)
    out = np.zeros(len(dev), bool)
    for a, b in episodes(cand, gap=120.0, min_dur=min_dur):
        if r[a] > r_onset:
            continue
        k = b
        limit = min(len(dev) - 1, b + int(600 / DT))
        while k < limit and dev[k] > dev_rel:
            k += 1
        if dev[k] <= dev_rel:                  # recaptured, so the episode is closed
            out[a:k + 1] = True
    return out


def hold(fl, turn_min=300.0, win=360.0, radius=5.0, r_hi=70.0):
    """One closed circuit of at least 300 deg inside a bounded horizontal extent."""
    turn, r = fl['turn'], fl['r']
    n = len(turn)
    out = np.zeros(n, bool)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(turn[1:]) * DT)])
    w = int(win / DT)
    step = max(1, w // 4)
    lat, lon = fl['lat'], fl['lon']
    for i in range(0, max(1, n - w), step):
        j = min(i + w, n - 1)
        if cum[j] - cum[i] < turn_min or r[i] > r_hi:
            continue
        dy = np.ptp(lat[i:j + 1]) * 111.0
        dx = np.ptp(lon[i:j + 1]) * 111.0 * np.cos(np.radians(lat[i]))
        if 0.5 * (dy + dx) <= radius:
            out[i:j + 1] = True
    return out


def lvl(fl, rate=100.0, min_dur=90.0, drop=1200.0, span=480.0,
        alt_lo=2000.0, alt_hi=14000.0):
    """Level-off: a vertical plateau that is bracketed by at least `drop` ft of descent."""
    alt, vs = fl['alt'], fl['vs']
    n = len(alt)
    out = np.zeros(n, bool)
    seg = runs((np.abs(vs) < rate) & (alt > alt_lo) & (alt < alt_hi), min_dur)
    s = int(span / DT)
    for a, b in episodes(seg, gap=DT, min_dur=min_dur):
        pre, post = max(0, a - s), min(n - 1, b + s)
        if alt[pre] - alt[a] >= drop and alt[b] - alt[post] >= drop:
            out[a:b + 1] = True
    return out


def spd(fl, vref, prof, alpha=0.85, margin_kt=5.0, min_dur=60.0,
        r_lo=8.0, r_hi=62.0, floor=40.0):
    """Speed control: sustained flight below the type-normalised nominal speed profile.

    The nominal profile is scaled by the arrival's own touchdown speed, so a heavy jet is
    not flagged for flying faster than a regional turboprop, and a slow turboprop is not
    flagged for flying below the fleet median.
    """
    gs, r = fl['gs'], fl['r']
    b = np.clip(np.round(r / 2.0).astype(int), 0, len(prof) - 1)
    nom = prof[b] * (vref / max(prof[0], 1e-3))
    m = (gs < alpha * nom - margin_kt * KT) & (r > r_lo) & (r < r_hi) & (gs > floor)
    return runs(m, min_dur)


def gs_profile(fls):
    """Population ground-speed profile on PROF_EDGES, from training months only."""
    bins = {}
    for fl in fls:
        b = np.digitize(fl['r'], PROF_EDGES) - 1
        ok = (b >= 0) & (b < len(PROF_EDGES) - 1)
        for bi in np.unique(b[ok]):
            m = ok & (b == bi)
            bins.setdefault(PROF_EDGES[bi], []).append(float(np.median(fl['gs'][m])))
    keys = sorted(bins)
    prof = np.array([np.median(bins[k]) for k in keys])
    full = np.interp(PROF_EDGES, np.array(keys), prof, left=prof[0], right=prof[-1])
    return full


def touchdown_speed(fl):
    return float(np.median(fl['gs'][-4:]))


RULES = dict(vec=dict(), hold=dict(), lvl=dict(), spd=dict())


def label_all(fl, prof, rules=RULES):
    ev = dict(VEC=vec(fl, **rules['vec']), HOLD=hold(fl, **rules['hold']),
              LVL=lvl(fl, **rules['lvl']),
              SPD=spd(fl, touchdown_speed(fl), prof, **rules['spd']))
    # precedence: HOLD > VEC > LVL > SPD, one label per frame
    ev['SPD'] &= ~ev['LVL'] & ~ev['VEC'] & ~ev['HOLD']
    ev['LVL'] &= ~ev['VEC'] & ~ev['HOLD']
    ev['VEC'] &= ~ev['HOLD']
    return ev
