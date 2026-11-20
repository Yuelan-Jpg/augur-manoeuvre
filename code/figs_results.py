"""Result-side figures: architecture, event-level scores, calibration, error, case studies."""
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D

import data as D
import style as S
from style import C, TYPE_C, TYPE_LABEL
from metrics import extract, match

DATA = D.DATA
from config import FIGS


def reported_threshold(tag):
    """Operating point exactly as reported in the table, so figure and table agree."""
    with open(f'{DATA}/results_test.json') as fh:
        return float(json.load(fh)[tag]['thr'])
ORDER = ['persistence', 'static', 'gru', 'tf', 'tcn']
TYPE_SHORT = {'VEC': 'vectoring', 'LVL': 'level-off', 'SPD': 'speed control'}
NAME = {'persistence': 'no event', 'static': 'static logistic', 'gru': 'GRU',
        'tf': 'Transformer', 'tcn_nostatic': 'TCN, no static inputs',
        'tcn_single': 'TCN, occupancy head only', 'tcn': 'AUGUR (full)'}


def fig_arch():
    """The forecaster, annotated with the shapes and the losses actually used."""
    fig, ax_ = S.use(178, 66, 1, 1, **{'font.size': 9.0})
    ax = ax_[0, 0]
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 34)
    ax.axis('off')

    def band(x0, x1, label):
        ax.add_patch(mp.FancyBboxPatch((x0, 1.2), x1 - x0, 31.2,
                                       boxstyle='round,pad=0,rounding_size=1.5',
                                       fc='#f6f6f4', ec='none', zorder=0))
        ax.text(x0 + 1.0, 31.0, label, fontsize=7.2, color=C['grey'], va='center',
                ha='left', zorder=1)

    def box(x, y, w, h, txt, fs=7.2, fc='white', ec=None, lw=0.9, bold=False):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                                       boxstyle='round,pad=0,rounding_size=1.0',
                                       fc=fc, ec=ec or C['ink'], lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs,
                zorder=3, linespacing=1.6,
                fontweight='bold' if bold else 'normal')

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=0.9, color=C['ink'],
                                    shrinkA=0, shrinkB=0, mutation_scale=7))

    def note(x, y, txt, ha='center', col=None, fs=6.9):
        ax.text(x, y, txt, ha=ha, va='top', fontsize=fs, color=col or C['grey'],
                linespacing=1.45, zorder=3)

    band(0.5, 20.0, 'inputs')
    band(21.0, 66.5, 'encoder')
    band(67.5, 119.0, 'prediction heads')

    # ---- two input streams and their concatenation
    box(2.0, 19.0, 13.5, 8.0, 'dynamic\n60 frames × 14')
    box(2.0, 4.5, 13.5, 8.0, 'static\n11, tiled to 60')
    ax.add_patch(mp.FancyBboxPatch((18.0, 6.5), 4.0, 18.5,
                                   boxstyle='round,pad=0,rounding_size=0.8',
                                   fc=C['light'], ec=C['ink'], lw=0.7, zorder=2))
    for i in (1, 2):
        ax.plot([18.0 + i * 4.0 / 3] * 2, [6.5, 25.0], color='white', lw=0.8, zorder=3)
    note(20.0, 5.6, '60 × 25', fs=6.9)
    arrow(15.8, 23.0, 17.7, 21.0)
    arrow(15.8, 8.5, 17.7, 10.5)

    # ---- pointwise projection
    box(24.5, 12.0, 7.5, 7.0, '1×1\nconv', fs=7.0)
    note(28.25, 10.8, '25 → 56', col=C['ink'])
    arrow(22.3, 15.5, 24.3, 15.5)

    # ---- residual dilated stack
    box(35.0, 6.8, 17.5, 17.4,
        '5 × residual\ndilated causal\nconv\n$k$=3, $d$=1…16\n56 ch, GELU', fs=7.0)
    note(43.75, 28.4, 'receptive field 125 frames\n> the 60 frames available')
    arrow(32.3, 15.5, 34.8, 15.5)

    # ---- encoding of the last frame
    box(55.5, 12.0, 10.5, 7.0, 'last frame\nLayerNorm', fs=7.0)
    note(60.75, 10.8, '$c \\in \\mathrm{R}^{56}$', col=C['ink'])
    arrow(52.8, 15.5, 55.3, 15.5)

    # ---- heads: one scored, two auxiliary
    heads = [(22.0, 'occupancy', 'Linear 128 → 3\nover 90 frames', '90 × 3',
              'masked BCE', 'scored', C['ink'], True),
             (12.0, 'onset', 'Linear 56 → 12', '3 × 4', 'CE', 'auxiliary',
              C['grey'], False),
             (2.0, 'duration', 'Linear 56 → 18', '3 × 6', 'CE', 'auxiliary',
              C['grey'], False)]
    for y, name, body, dim, loss, tag, col, strong in heads:
        ec = C['ink'] if strong else C['grey']
        box(69.0, y, 17.0, 7.0, f'{name} head\n{body}', fs=7.0, ec=ec,
            lw=1.3 if strong else 0.7, bold=strong)
        box(89.5, y + 0.8, 11.0, 5.4, f'{dim}\nlogits', fs=6.9, ec=ec,
            lw=1.3 if strong else 0.7)
        arrow(66.4, 15.5, 68.7, y + 3.5)
        arrow(86.3, y + 3.5, 89.2, y + 3.5)
        ax.text(102.5, y + 4.4, loss, va='center', ha='left', fontsize=7.0, color=col)
        ax.text(102.5, y + 2.2, tag, va='center', ha='left', fontsize=6.7, color=col,
                style='italic')
    S.save(fig, f'{FIGS}/fig5_architecture.png')


def fig_scores():
    """Event-level scores: a dot-and-interval plot, and the same scores split by type."""
    with open(f'{DATA}/results_test.json') as fh:
        res = json.load(fh)
    tags = [t for t in ORDER if t in res]
    n = len(tags)
    ref = res['static']['macro_f1']

    fig, ax = S.use(178, 84, 1, 2)
    fig.subplots_adjust(wspace=0.26, left=0.225, right=0.975, bottom=0.15, top=0.88)

    # ---- (a) macro F1 per method, with its day-block interval
    a = ax[0, 0]
    a.axvline(ref, color=C['grey'], lw=0.7, ls=(0, (4, 3)), zorder=1)
    for i, t in enumerate(tags):
        r = res[t]
        y = n - 1 - i
        m, (lo, hi) = r['macro_f1'], r['macro_f1_ci']
        full = t == 'tcn'
        if hi > lo:
            a.plot([lo, hi], [y, y], color=C['ink'], lw=1.2 if full else 0.7,
                   solid_capstyle='butt', zorder=3)
            for x in (lo, hi):
                a.plot([x, x], [y - 0.085, y + 0.085], color=C['ink'],
                       lw=1.2 if full else 0.7, zorder=3)
        a.plot(m, y, 'o', ms=4.9 if full else 4.4, color=C['ink'],
               mfc=C['ink'] if full else 'white', mew=0.8, zorder=4, clip_on=False)
        a.annotate('%.3f' % m, xy=(max(hi, m), y), xytext=(9, 0),
                   textcoords='offset points', fontsize=7.4, va='center', ha='left',
                   color=C['ink'], fontweight='bold' if full else 'normal')

    a.set_yticks(range(n))
    a.set_yticklabels([NAME[t] for t in tags][::-1])
    a.tick_params(axis='y', length=0, pad=6)
    a.set_xlabel('macro event-level $F_1$')
    a.set_xlim(-0.045, 0.80)
    a.set_ylim(-0.62, n - 0.45)
    a.set_xticks(np.arange(0, 0.81, 0.2))
    a.annotate('static reference', xy=(ref, n - 0.62), xytext=(4, 0),
               textcoords='offset points', fontsize=6.4, color=C['grey'],
               va='top', ha='left')
    S.panel(a, 'a', x=-0.34, y=1.06, size=9.5)

    # ---- (b) the same scores broken down by manoeuvre type
    b = ax[0, 1]
    ramp = {'static': 0.94, 'gru': 0.64, 'tf': 0.40, 'tcn': 0.13}
    short = {'static': 'static', 'gru': 'GRU', 'tf': 'Transformer', 'tcn': 'AUGUR'}
    # the no-event baseline scores zero on every type, so it is left out of the bars
    bar_tags = [t for t in tags if t in ramp]
    bw, gap = 0.195, 0.215
    for j, ty in enumerate(D.TYPES):
        x0 = j - (len(bar_tags) - 1) * gap / 2
        for i, t in enumerate(bar_tags):
            v = res[t][ty]['f1']
            x = x0 + i * gap
            b.bar(x, v, width=bw, color=str(ramp[t]), edgecolor=C['ink'],
                  linewidth=0.45, zorder=2)
    b.set_xlim(-0.62, len(D.TYPES) - 0.38)
    b.set_ylim(0, 1.16)
    b.set_yticks(np.arange(0, 1.01, 0.2))
    b.set_xticks(range(len(D.TYPES)))
    b.set_xticklabels([TYPE_LABEL[ty].replace('close-in ', 'close-in\n')
                       for ty in D.TYPES], fontsize=7.2)
    b.tick_params(axis='x', length=0, pad=4)
    b.set_ylabel('event-level $F_1$')
    handles = [mp.Patch(facecolor=str(ramp[t]), edgecolor=C['ink'], linewidth=0.45,
                        label=short[t]) for t in bar_tags]
    b.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.02, 0.995), ncol=2,
             fontsize=6.6, frameon=False, handlelength=1.25, handleheight=0.9,
             columnspacing=1.1, labelspacing=0.45, borderpad=0.0)
    S.panel(b, 'b', x=-0.11, y=1.06, size=9.5)

    S.save(fig, f'{FIGS}/fig6_event_scores.png')


def _lighten(hexcol, f):
    """Blend a colour towards white; f = 0 keeps it, f = 1 is white."""
    h = hexcol.lstrip('#')
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)
    rgb = rgb + (255 - rgb) * f
    return tuple(rgb / 255.0)


def fig_calibration():
    X, M, Y, DUR, VL, meta = D.load()
    sp = D.split_index(meta)
    te = sp['test']
    z = np.load(f'{DATA}/pred_tcn.npz')
    P = z['test']
    thr = reported_threshold('tcn')
    days = meta.day.values[te]

    fig, ax = S.use(178, 50, 1, 3)
    fig.subplots_adjust(wspace=0.36, left=0.07, right=0.98, bottom=0.19, top=0.90)

    # (a) reliability
    a = ax[0, 0]
    edges = np.linspace(0, 1, 11)
    for k, t in enumerate(D.TYPES):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (P[:, :, k] >= lo) & (P[:, :, k] < hi)
            ml = m & (np.arange(P.shape[1])[None, :] < VL[te][:, None])
            if ml.sum() < 50:
                continue
            xs.append(P[:, :, k][ml].mean())
            ys.append((Y[te][:, :, k][ml] > 0.5).mean())
        a.plot(xs, ys, marker='o', ms=2.2, lw=0.9, color=TYPE_C[t], label=TYPE_LABEL[t])
    a.plot([0, 1], [0, 1], color=C['grey'], lw=0.5, ls=(0, (3, 2)))
    a.set_xlabel('predicted probability')
    a.set_ylabel('observed frequency')
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.annotate('bins with $n \\geq 50$', (0.02, 0.58), xycoords='axes fraction', fontsize=5.6,
               color=C['grey'])
    a.legend(loc='upper left', fontsize=5.6, frameon=False, handlelength=1.6,
             borderaxespad=0.2, labelspacing=0.25, handletextpad=0.5)
    S.panel(a, 'a')

    # (b) onset error among matched events
    b = ax[0, 1]
    for k, t in enumerate(D.TYPES):
        errs = []
        for j, i in enumerate(te):
            v = int(VL[i])
            p = extract(P[j, :v, k] >= thr)
            r = extract(Y[i, :v, k] > 0.5)
            _, _, _, pairs = match(p, r, 180.0, 0.1)
            errs += [abs(p[a2, 0] - r[b2, 0]) / 60.0 for a2, b2 in pairs]
        if not errs:
            continue
        e = np.sort(errs)
        b.plot(e, np.arange(1, len(e) + 1) / len(e), color=TYPE_C[t], lw=1.0)
        # the three step curves share the same colour key as panel (a), so the annotation
        # carries the name and the median rather than a separate legend
        b.annotate(f'{TYPE_SHORT[t]} {np.median(e):.1f}', (0.55, 0.10 + 0.085 * k),
                   xycoords='axes fraction', fontsize=5.6, color=TYPE_C[t])
    b.axvline(3.0, color=C['grey'], lw=0.5, ls=(0, (3, 2)))
    b.set_xlabel('|onset error| (min)')
    b.set_ylabel('cumulative share')
    b.set_xlim(0, 3.2); b.set_ylim(0, 1)
    S.panel(b, 'b')

    # (c) event-level operating curve
    c = ax[0, 2]
    for k, t in enumerate(D.TYPES):
        xs, ys = [], []
        for thr in np.arange(0.05, 0.96, 0.05):
            tp = fp = fn = 0
            for j, i in enumerate(te):
                v = int(VL[i])
                p = extract(P[j, :v, k] >= thr)
                r = extract(Y[i, :v, k] > 0.5)
                a2, b2, c2, _ = match(p, r, 180.0, 0.1)
                tp, fp, fn = tp + a2, fp + b2, fn + c2
            xs.append(tp / max(tp + fp, 1)); ys.append(tp / max(tp + fn, 1))
        c.plot(xs, ys, color=TYPE_C[t], lw=1.0, marker='o', ms=1.6)
    c.set_xlabel('event precision')
    c.set_ylabel('event recall')
    c.set_xlim(0, 1); c.set_ylim(0, 1)
    S.panel(c, 'c')
    S.save(fig, f'{FIGS}/fig7_calibration_operating.png')


def fig_cases():
    """Three error shapes: a clean match, a split episode, an invented episode."""
    X, M, Y, DUR, VL, meta = D.load()
    sp = D.split_index(meta)
    te = sp['test']
    z = np.load(f'{DATA}/pred_tcn.npz')
    P = z['test']
    thr = reported_threshold('tcn')

    good = split = false = None
    used = set()
    for j, i in enumerate(te):
        v = int(VL[i])
        if v < 80:
            continue
        cs = meta.flight.values[i]
        if cs in used:
            continue
        # clean vectoring match
        if good is None:
            r = extract(Y[i, :v, 0] > 0.5)
            p = extract(P[j, :v, 0] >= thr)
            if len(r) == 1 and len(p) == 1 and abs(p[0, 0] - r[0, 0]) < 120:
                good = (j, i, 0)
                used.add(cs)
                continue
        # vectoring split into more than one prediction
        if split is None:
            r = extract(Y[i, :v, 0] > 0.5)
            p = extract(P[j, :v, 0] >= thr)
            if len(r) == 1 and len(p) >= 2 and r[0, 1] - r[0, 0] > 240:
                split = (j, i, 0)
                used.add(cs)
                continue
        # level-off predicted where none occurs
        if false is None:
            r = extract(Y[i, :v, 1] > 0.5)
            p = extract(P[j, :v, 1] >= thr)
            if len(r) == 0 and len(p) == 1 and p[0, 1] - p[0, 0] > 120:
                false = (j, i, 1)
                used.add(cs)
        if good and split and false:
            break

    cases = [('a', good, 'matched'), ('b', split, 'split'), ('c', false, 'invented')]
    fig, ax = S.use(178, 76, 3, 1)
    fig.subplots_adjust(hspace=0.62, left=0.14, right=0.86, bottom=0.13, top=0.94)
    for row, (letter, case, kind) in enumerate(cases):
        a = ax[row, 0]
        if case is None:
            continue
        j, i, k = case
        v = int(VL[i])
        t = np.arange(v) * 10 / 60.0
        col = TYPE_C[D.TYPES[k]]
        a.fill_between(t, 0, P[j, :v, k] >= thr, color=col, alpha=0.28, lw=0,
                       step='mid')
        a.plot(t, P[j, :v, k], color=col, lw=0.9)
        refs = extract(Y[i, :v, k] > 0.5)
        if len(refs) == 0:
            # the invented episode is the one row with nothing to match; say so where the
            # reference bar would sit, rather than leaving a silent gap
            a.annotate('no reference episode', (0.42, 1.02), xycoords='axes fraction',
                       va='bottom', fontsize=5.6, color=C['grey'])
        for ev in refs:
            a.plot([ev[0] / 60, ev[1] / 60], [1.06, 1.06], color=C['ink'], lw=2.0,
                   solid_capstyle='butt', clip_on=False)
        a.set_xlim(0, max(6, v * 10 / 60))
        a.set_ylim(0, 1.0)
        a.set_yticks([0, 0.5, 1.0])
        if row == 2:
            a.set_xlabel('time after $t_0$ (min)')
        a.text(-0.085, 0.5, 'probability', rotation=90, va='center', ha='center',
               transform=a.transAxes, fontsize=6.2)
        a.annotate(f'{TYPE_LABEL[D.TYPES[k]]}: {kind}', (0.012, 0.94),
                   xycoords='axes fraction', va='top', fontsize=6.2, color=col)
        a.annotate(f"{meta.flight.values[i]}", (1.01, 0.5),
                   xycoords='axes fraction', va='center', fontsize=5.8, color=C['grey'])
        S.panel(a, letter, x=-0.20, y=1.0)
    handles = [Line2D([], [], color=C['ink'], lw=2.0, label='reference episode'),
               mp.Patch(facecolor=C['grey'], alpha=0.35, label='predicted occupancy'),
               Line2D([], [], color=C['ink'], lw=0.9, label='probability')]
    ax[0, 0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.16),
                    ncol=3, fontsize=5.8, frameon=False, handlelength=1.6)
    S.save(fig, f'{FIGS}/fig8_case_studies.png')


if __name__ == '__main__':
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'arch'):
        fig_arch()
    if what in ('all', 'scores'):
        fig_scores()
    if what in ('all', 'cal'):
        fig_calibration()
    if what in ('all', 'cases'):
        fig_cases()
