"""Shared figure style: boxed axes, generous type, no grid, muted palette, one accent.

The reference is panel (a) of Fig. 1 -- a full frame round every panel, large axis type,
thin grey context strokes, a single dark-red accent for whatever is being pointed at, and
no gridlines.  Every figure in the set follows that, so the nine read as one hand.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

PT = 1.0 / 72.27

# muted, print-safe palette (no saturated primaries)
C = dict(
    ink='#1b1b1b', grey='#8a8a8a', light='#d8d8d8', paper='#ffffff',
    dr='#2f5d7c', hold='#8c5a3c', lvl='#4c7a52', spd='#7a5a86',
    alt='#b08a4a', red='#8f3a3a',
)
TYPE_C = {'VEC': C['dr'], 'HOLD': C['hold'], 'LVL': C['lvl'], 'SPD': C['spd']}
TYPE_LABEL = {'VEC': 'close-in vectoring', 'HOLD': 'holding',
              'LVL': 'level-off', 'SPD': 'speed control'}


def use(width_mm=170.0, height_mm=60.0, nrows=1, ncols=1, **kw):
    """Create a figure sized in millimetres, the way journals ask for it."""
    base = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9.0,
        'axes.titlesize': 9.0,
        'axes.labelsize': 9.0,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.0,
        'axes.linewidth': 0.8,
        'axes.edgecolor': C['ink'],
        'axes.labelcolor': C['ink'],
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': False,
        'xtick.color': C['ink'],
        'ytick.color': C['ink'],
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.2,
        'ytick.major.size': 3.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.5,
        'lines.markersize': 4.0,
        'lines.solid_capstyle': 'round',
        'patch.linewidth': 0.6,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.5,
        'legend.labelspacing': 0.35,
        'legend.borderaxespad': 0.2,
        'grid.color': C['light'],
        'grid.linewidth': 0.4,
        'figure.facecolor': C['paper'],
        'axes.facecolor': C['paper'],
        'savefig.facecolor': C['paper'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
    base.update(kw)
    mpl.rcParams.update(base)
    fig, ax = plt.subplots(nrows, ncols, figsize=(width_mm / 25.4, height_mm / 25.4),
                           squeeze=False)
    fig.subplots_adjust(wspace=0.32, hspace=0.36)
    return fig, ax


def panel(ax, letter, x=-0.20, y=1.03, size=10.0):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=size, fontweight='bold', color=C['ink'])


def label_end(ax, x, y, text, color=None, dx=0.02, dy=0.0, **kw):
    """Direct label at a curve end instead of a legend entry."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords='offset fraction',
                color=color or C['ink'], va='center', ha='left',
                fontsize=kw.pop('fontsize', 8.5), **kw)


def light_grid(ax, axis='y'):
    """Kept for call sites: the house style carries no gridlines."""
    ax.grid(False)


def despine(ax, left=True, bottom=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def save(fig, path, dpi=600):
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.replace('.png', '.pdf'))
    plt.close(fig)
    print('wrote', path)
