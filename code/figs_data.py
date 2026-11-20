"""Data-side figures: study area, surveillance quality, manoeuvre examples, prevalence.

Style conventions are in style.py: no in-figure titles, panel letters, thin strokes, muted
palette, direct labels.  Units follow aviation practice (NM or km, kt, ft).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import style as S
from style import C, TYPE_C, TYPE_LABEL

from config import DATA, FIGS, RAW, read_parquet
ZUUU = (30.5785, 103.9471)
ZUTF = (30.3125, 104.4413)
KT = 0.5144


def load_month(m):
    return np.load(f'{DATA}/fl_{m}.npy', allow_pickle=True)


def fig_area():
    fl = np.load(f'{DATA}/fl_202501.npy', allow_pickle=True)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(fl), 700, replace=False)

    fig = plt.figure(figsize=(178 / 25.4, 72 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.26,
                          left=0.08, right=0.965, bottom=0.19, top=0.90)
    a = fig.add_subplot(gs[0, 0])
    for i in pick:
        f = fl[i]
        a.plot(f['lon'], f['lat'], color=C['grey'], lw=0.22, alpha=0.32,
               solid_capstyle='round', zorder=1)
    for (lat, lon), nm, dx, dy in [(ZUUU, 'ZUUU', -40, 7), (ZUTF, 'ZUTF', 9, -12)]:
        th = np.linspace(0, 2 * np.pi, 240)
        a.plot(lon + 60 / (111.32 * np.cos(np.radians(lat))) * np.cos(th),
               lat + 60 / 110.57 * np.sin(th), color=C['red'], lw=0.55, ls=(0, (3, 2)),
               zorder=3)
        a.plot(lon, lat, marker='s', color=C['ink'], ms=2.4, zorder=4)
        a.annotate(nm, (lon, lat), xytext=(dx, dy), textcoords='offset points',
                   fontsize=6.5, color=C['ink'])
    a.set_xlabel('longitude (°E)')
    a.set_ylabel('latitude (°N)')
    a.set_aspect(1 / np.cos(np.radians(30.6)))
    a.set_xlim(102.55, 105.65)
    a.set_ylim(29.15, 31.95)
    a.annotate('60 km anchor', (103.9471, 30.5785 + 60 / 110.57),
               xytext=(0, 2), textcoords='offset points', ha='center', va='bottom',
               fontsize=5.8, color=C['red'])
    S.panel(a, 'a', x=-0.11, y=1.015)

    # traffic volume per day, with the split boundaries marked
    b = fig.add_subplot(gs[0, 1])
    days = []
    for m in ['202501', '202502', '202503']:
        d = read_parquet(f'{DATA}/arr_{m}.parquet', columns=['flight', 'ts'])
        days.append(d.groupby('flight').ts.min().dt.floor('D').value_counts())
    v = pd.concat(days).sort_index()
    x = np.arange(len(v))
    b.bar(x, v.values, width=0.78, color=C['grey'], edgecolor='none')
    b.set_xlim(-0.5, len(v) - 0.5)
    b.set_ylim(0, 780)
    b.set_ylabel('arrivals per day')
    b.set_xticks([0, 30, 59, 89])
    b.set_xticklabels(['1 Jan', '31 Jan', '1 Mar', '31 Mar'])
    S.light_grid(b)
    for xb in (58.5, 73.5):
        b.plot([xb, xb], [0, 730], color=C['ink'], lw=0.5, ls=(0, (4, 3)), zorder=3)
    for xx, txt in [(29, 'training'), (66, 'validation'), (81, 'test')]:
        b.annotate(txt, (xx, 745), ha='center', fontsize=6.0, color=C['ink'])
    S.panel(b, 'b', x=-0.11, y=1.015)
    S.save(fig, f'{FIGS}/fig1_study_area.png')


def fig_quality():
    d = read_parquet(f'{DATA}/arr_202501.parquet', columns=['flight', 'ts', 'gs', 'alt', 'vr'])
    g = d.sort_values(['flight', 'ts']).groupby('flight', observed=True)
    dt = g.ts.diff().dt.total_seconds().dropna()
    dt = dt[(dt > 0) & (dt < 120)]

    fig, ax = S.use(178, 62, 1, 3)
    fig.subplots_adjust(wspace=0.42, left=0.075, right=0.985, bottom=0.22, top=0.90)
    a = ax[0, 0]
    a.hist(dt, bins=np.arange(0, 61, 1.5), color=C['dr'], edgecolor='none', alpha=0.9,
           density=True)
    a.set_xlabel('interval between reports (s)')
    a.set_ylabel('density')
    a.set_xlim(0, 40)
    a.annotate(f'median {dt.median():.0f} s', (0.60, 0.82), xycoords='axes fraction',
               fontsize=6.5)
    S.panel(a, 'a')

    b = ax[0, 1]
    vr = pd.to_numeric(d['vr'], errors='coerce').dropna()
    vals, counts = np.unique(vr.round(0), return_counts=True)
    top = np.argsort(counts)[-12:]
    order = top[np.argsort(vals[top])]
    share = counts[order] / len(vr)
    b.barh(np.arange(len(order)), share, height=0.72, color=C['hold'], edgecolor='none')
    b.set_yticks(np.arange(len(order)))
    b.set_yticklabels([f'{vals[i]:.0f}' for i in order], fontsize=6.6)
    b.set_ylim(-0.7, len(order) - 0.3)
    b.set_xlim(0, 0.42)
    b.set_xlabel('share of reports')
    b.set_ylabel('reported vertical rate\n(transmitted value, ft/min)')
    b.annotate(f'{len(top)} values cover {100 * counts[top].sum() / len(vr):.0f}%\nof all reports',
               (0.30, 0.30), xycoords='axes fraction', fontsize=7.4, linespacing=1.4)
    S.panel(b, 'b')

    c = ax[0, 2]
    dur = (g.ts.max() - g.ts.min()).dt.total_seconds() / 60.0
    c.hist(dur, bins=np.arange(0, 55, 2), color=C['grey'], edgecolor='none', alpha=0.75)
    c.axvline(18, color=C['red'], lw=0.7, ls=(0, (3, 2)))
    c.annotate('shortest admissible\ntrack, 18 min',
               (0.04, 0.97), xycoords='axes fraction', va='top', fontsize=5.8,
               color=C['ink'])
    c.set_xlabel('track duration per arrival (min)')
    c.set_ylabel('arrivals')
    c.set_ylabel('arrivals')
    c.set_xlim(0, 52)
    S.panel(c, 'c')
    S.save(fig, f'{FIGS}/fig2_surveillance_quality.png')


def _pick_example(fl, kind):
    """Choose a legible illustration of one manoeuvre type."""
    best, score = None, -1e9
    for f in fl:
        if len(f['t']) < 220:
            continue
        m = f['ev_' + kind]
        if m.sum() < 30:
            continue
        r, x = f['r'], f['x']
        # skip arrivals whose early frames carry a differencing artefact, so that the
        # illustrated example is representative rather than pathological
        if f['gs'][r < 70].max() > 200.0:
            continue
        if kind == 'vec':
            sel = m & (r < 30)
            if not sel.any():
                continue
            xa = np.abs(x[(r < 40) & (r > 2)])
            if xa.size == 0 or xa.max() > 14:
                continue
            s = m.sum() * 0.05 - np.abs(x[sel]).mean()
        elif kind == 'lvl':
            alt = f['alt']
            i = np.flatnonzero(m)
            step = alt[max(0, i[0] - 60)] - alt[min(len(alt) - 1, i[-1] + 60)]
            if step < 1500:
                continue
            s = step / 1000.0
        else:
            gs = f['gs']
            i = np.flatnonzero(m)
            dip = gs[max(0, i[0] - 30):i[0]].max() - gs[i].min() if i[0] > 30 else 0
            if dip < 25:
                continue
            s = dip
        if s > score:
            best, score = f, s
    return best


def fig_examples():
    fl = np.load(f'{DATA}/fl_202501.npy', allow_pickle=True)
    fig, ax = S.use(178, 72, 1, 4, **{'font.size': 8.0, 'axes.labelsize': 8.0,
                                      'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2})
    fig.subplots_adjust(wspace=0.42, bottom=0.46, top=0.88, left=0.055, right=0.99)

    def one(ax_, name, series, ylab, ylim=None, dyn=None):
        """Context stroke in grey, the flagged episode in colour, and its span shaded."""
        f = _pick_example(fl, name.lower())
        m = f['ev_' + name.lower()]
        col = TYPE_C[name]
        ax_.plot(f['r'], series(f), color=C['light'], lw=1.3, solid_capstyle='round',
                 zorder=1)
        ax_.plot(np.where(m, f['r'], np.nan), np.where(m, series(f), np.nan),
                 color=col, lw=2.2, zorder=3)
        if m.any():
            r0, r1 = f['r'][m].max(), f['r'][m].min()
            ax_.axvspan(r1, r0, color=col, alpha=0.09, lw=0, zorder=0)
        ax_.set_xlim(64, 0)
        if ylim:
            ax_.set_ylim(*ylim)
        ax_.set_xlabel('range to touchdown (km)')
        ax_.set_ylabel(ylab)
        ax_.annotate(TYPE_LABEL[name], (0.04, 0.94), xycoords='axes fraction',
                     fontsize=7.6, color=col, va='top')
        return f, m

    a = ax[0, 0]
    one(a, 'VEC', lambda f: f['x'], 'cross-track offset (km)', (-11, 11))
    a.axhline(0, color=C['grey'], lw=0.6, ls=(0, (1, 1.6)), zorder=0)
    S.panel(a, 'a', x=-0.34, size=9)

    b = ax[0, 1]
    one(b, 'LVL', lambda f: f['alt'] / 1000.0, 'altitude (1000 ft)', (-0.6, 8))
    S.panel(b, 'b', x=-0.34, size=9)

    c = ax[0, 2]
    one(c, 'SPD', lambda f: f['gs'] / KT, 'ground speed (kt)', (80, 520))
    S.panel(c, 'c', x=-0.34, size=9)

    # (d) event raster: 40 arrivals, one row each, range domain
    d = ax[0, 3]
    rng = np.random.default_rng(7)
    pick = rng.choice(len(fl), 40, replace=False)
    for row, i in enumerate(pick):
        f = fl[i]
        for name in ['VEC', 'LVL', 'SPD']:
            m = f['ev_' + name.lower()]
            idx = np.flatnonzero(m)
            if idx.size == 0:
                continue
            brk = np.flatnonzero(np.diff(idx) > 6)
            for seg in np.split(idx, brk + 1):
                d.plot([f['r'][seg[0]], f['r'][seg[-1]]], [row, row],
                       color=TYPE_C[name], lw=1.8, solid_capstyle='butt')
    d.set_xlim(64, 0)
    d.set_ylim(-1, 41)
    d.set_yticks([])
    d.set_xlabel('range to touchdown (km)')
    d.set_ylabel('arrivals')
    handles = [Line2D([], [], color=TYPE_C[k], lw=1.8, label=TYPE_LABEL[k])
               for k in ['VEC', 'LVL', 'SPD']]
    d.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.44, -0.30), ncol=1,
             fontsize=7.2, handlelength=1.2, frameon=False, borderpad=0.0,
             labelspacing=0.28)
    S.panel(d, 'd', x=-0.12, size=9)
    S.save(fig, f'{FIGS}/fig3_manoeuvre_examples.png')


def fig_prevalence():
    fl = np.load(f'{DATA}/fl_202501.npy', allow_pickle=True)
    edges = np.arange(0, 65, 5.0)
    counts = np.zeros((len(edges) - 1, 3))
    total = np.zeros(len(edges) - 1)
    for f in fl:
        b = np.digitize(f['r'], edges) - 1
        ok = (b >= 0) & (b < len(edges) - 1)
        for bi in np.unique(b[ok]):
            total[bi] += 1
            for k, name in enumerate(['vec', 'lvl', 'spd']):
                if f['ev_' + name][ok & (b == bi)].any():
                    counts[bi, k] += 1
    share = counts / np.maximum(total[:, None], 1)

    # episode counts over January, used for the legend labels so that panel (b) needs
    # no separate key
    durs_by = {}
    for name in ['VEC', 'LVL', 'SPD']:
        durs = []
        for f in fl:
            m = f['ev_' + name.lower()]
            idx = np.flatnonzero(m)
            if idx.size == 0:
                continue
            brk = np.flatnonzero(np.diff(idx) * 10 > 120)
            for s_ in np.split(idx, brk + 1):
                L = (s_[-1] - s_[0] + 1) * 10 / 60.0
                if L >= 1:
                    durs.append(L)
        durs_by[name] = np.sort(durs)

    fig, ax = S.use(165, 70, 1, 2)
    fig.subplots_adjust(wspace=0.32, bottom=0.17, top=0.88, left=0.155, right=0.985)

    # (a) how often each manoeuvre is present, against range to touchdown
    a = ax[0, 0]
    mid = (edges[:-1] + edges[1:]) / 2
    for k, name in enumerate(['VEC', 'LVL', 'SPD']):
        a.plot(mid, share[:, k], color=TYPE_C[name], lw=2.0, marker='o', ms=5.0,
               mfc='white', mew=1.4,
               label=f'{TYPE_LABEL[name]}  ({len(durs_by[name]):,} episodes)')
    k = int(np.argmax(share[:, 0]))
    a.annotate(f'peak {share[k, 0]:.2f}', xy=(mid[k], share[k, 0]), xytext=(0, -16),
               textcoords='offset points', ha='center', fontsize=8.0, color=TYPE_C['VEC'])
    a.set_xlabel('range to touchdown (km)')
    a.set_ylabel('share of arrivals with the manoeuvre')
    a.set_xlim(0, 62)
    a.set_ylim(0, 0.74)
    a.set_yticks(np.arange(0, 0.61, 0.2))
    a.legend(loc='upper right', fontsize=7.4, frameon=False, handlelength=1.6,
             labelspacing=0.45)
    S.panel(a, 'a', x=-0.12)

    # (b) how long they last
    b = ax[0, 1]
    tag = {'VEC': dict(dx=9, dy=0, ha='left'), 'LVL': dict(dx=8, dy=10, ha='left'),
           'SPD': dict(dx=-8, dy=-10, ha='right')}
    for name in ['VEC', 'LVL', 'SPD']:
        durs = durs_by[name]
        cdf = np.arange(1, len(durs) + 1) / len(durs)
        b.plot(durs, cdf, color=TYPE_C[name], lw=2.0)
        med = np.interp(0.5, cdf, durs)
        b.plot([med, med], [0, 0.5], color=TYPE_C[name], lw=0.9, ls=(0, (2, 2)),
               zorder=1)
        b.plot(med, 0.5, 'o', ms=5, color=TYPE_C[name], mfc='white', mew=1.4, zorder=4)
        t = tag[name]
        b.annotate(f'{med:.1f}', xy=(med, 0.5), xytext=(t['dx'], t['dy']),
                   textcoords='offset points', ha=t['ha'], va='center',
                   fontsize=8.0, color=TYPE_C[name])
    b.axhline(0.5, color=C['grey'], lw=0.7, ls=(0, (1, 2)), zorder=0)
    b.set_xscale('log')
    b.set_xlabel('manoeuvre duration (min)')
    b.set_ylabel('cumulative share of episodes')
    b.set_xlim(1, 30)
    b.set_ylim(0, 1.02)
    b.set_yticks(np.arange(0, 1.01, 0.25))
    b.set_xticks([1, 2, 5, 10, 20])
    b.set_xticklabels(['1', '2', '5', '10', '20'])
    b.minorticks_off()
    S.panel(b, 'b', x=-0.12)
    S.save(fig, f'{FIGS}/fig4_prevalence_duration.png')


if __name__ == '__main__':
    fig_area()
    fig_quality()
    fig_examples()
    fig_prevalence()
