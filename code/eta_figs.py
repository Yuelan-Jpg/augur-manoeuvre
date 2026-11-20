"""Figure for the downstream experiment: does the manoeuvre forecast reduce arrival-time error?

(a) cumulative distribution of the absolute error in seconds, with the median of each curve
marked, and (b) error against the realised time-to-land quartile with a 95% interval on
each mean, where the manoeuvre forecast should matter most: a long arrival is usually a
vectored one.
"""
import json

import numpy as np

import style as S
from style import C

from config import DATA
LABEL = {'kine': 'kinematic history', 'augur': '+ manoeuvre forecast',
         'oracle': '+ realised manoeuvres (oracle)'}
COL = {'kine': C['grey'], 'augur': C['dr'], 'oracle': C['lvl']}
STY = {'kine': dict(lw=2.2, ls='-'), 'augur': dict(lw=1.6, ls=(0, (4, 2))),
       'oracle': dict(lw=2.0, ls='-')}


def main():
    z = np.load(f'{DATA}/eta/eta_test_preds.npz')
    y = z['y']
    res = json.load(open(f'{DATA}/eta_results.json'))

    fig, ax = S.use(170, 70, 1, 2)
    fig.subplots_adjust(wspace=0.30, left=0.085, right=0.985, bottom=0.17, top=0.88)

    # (a) CDF of |error|
    a = ax[0, 0]
    for k in ('kine', 'oracle', 'augur'):
        e = np.sort(np.abs(z[k] - y) * 60.0)
        cdf = np.arange(1, len(e) + 1) / len(e)
        a.plot(e, cdf, color=COL[k], label=LABEL[k], zorder=3 if k == 'augur' else 2,
               **STY[k])
        med = np.interp(0.5, cdf, e)
        a.plot([med, med], [0, 0.5], color=COL[k], lw=0.8, ls=(0, (1, 2)), zorder=1)
        a.plot(med, 0.5, 'o', ms=5.5, color=COL[k], mfc='white', mew=1.5, zorder=4)
    a.axhline(0.5, color=C['grey'], lw=0.7, ls=(0, (1, 2)), zorder=0)
    a.set_xlim(0, 430)
    a.set_ylim(0, 1.02)
    a.set_yticks(np.arange(0, 1.01, 0.25))
    a.set_xlabel('absolute error in arrival time (s)')
    a.set_ylabel('share of test arrivals')
    a.legend(loc='lower right', fontsize=7.2, handlelength=1.8, labelspacing=0.35)
    a.annotate(f'n = {len(y):,} arrivals, {res["n"]["days"]} days', (0.03, 0.95),
               xycoords='axes fraction', fontsize=7.2, color=C['grey'])
    a.annotate('median  %.0f s  (oracle)\nmedian  %.0f s  (upper pair)'
               % (np.median(np.abs(z['oracle'] - y)) * 60,
                  np.median(np.abs(z['kine'] - y)) * 60),
               xy=(0.975, 0.47), xycoords='axes fraction', fontsize=7.2,
               color=C['ink'], linespacing=1.5, ha='right', va='top')
    S.panel(a, 'a', x=-0.17)

    # (b) error by realised time-to-land quartile, with a 95% interval on each mean
    b = ax[0, 1]
    q = np.quantile(y, [.25, .5, .75])
    g = np.digitize(y, q)
    edges = [y.min(), *q, y.max()]
    labels = [f'{edges[i]:.0f}–{edges[i + 1]:.0f}' for i in range(4)]
    x = np.arange(4)
    for k in ('kine', 'augur', 'oracle'):
        e = np.abs(z[k] - y) * 60.0
        m = np.array([e[g == i].mean() for i in range(4)])
        ci = np.array([1.96 * e[g == i].std(ddof=1) / np.sqrt((g == i).sum())
                       for i in range(4)])
        b.errorbar(x, m, yerr=ci, color=COL[k], marker='o', ms=5.0, mfc='white',
                   mew=1.5, capsize=3, elinewidth=1.2, label=LABEL[k], zorder=3,
                   **{kk: vv for kk, vv in STY[k].items() if kk == 'ls'})
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_xlabel('realised time to land (min)')
    b.set_ylabel('mean absolute error (s)')
    b.set_ylim(0, 175)
    b.set_xlim(-0.35, 3.35)
    S.panel(b, 'b', x=-0.17)

    S.save(fig, 'figs/fig9_downstream.png')
    print('wrote figs/fig9_downstream.png')


if __name__ == '__main__':
    main()
