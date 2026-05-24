"""Notebook setup: imports, theme, and drawing helpers."""
import sys, pathlib, warnings
warnings.filterwarnings('ignore')

# Locate project root via VS Code's __vsc_ipynb_file__ or CWD walk
def _find_root(nb_path=None):
    candidates = []
    if nb_path:
        candidates = list(pathlib.Path(nb_path).parents)
    candidates += [pathlib.Path.cwd()] + list(pathlib.Path.cwd().parents)
    for p in candidates:
        if (p / 'pyproject.toml').exists():
            return p
    return None

import builtins
_nb  = builtins.__dict__.get('__vsc_ipynb_file__')
_root = _find_root(_nb)
if _root is None:
    raise RuntimeError(f'Cannot find project root. CWD={pathlib.Path.cwd()}')
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
print(f'Project root : {_root}')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
print(f'matplotlib   : {matplotlib.__version__}  backend={matplotlib.get_backend()}')

BG, BG2  = '#0b1120', '#131f30'
FG, GRID = '#cdd6f4', '#1e2d42'
GREEN, RED, GOLD = '#26a69a', '#ef5350', '#f9a825'
BLUE, PURPLE, CYAN = '#5c9cf5', '#ce93d8', '#80cbc4'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': BG2,
    'axes.edgecolor':   GRID, 'text.color':     FG,
    'axes.labelcolor':  FG,   'xtick.color':    FG,
    'ytick.color':      FG,   'grid.color':     GRID,
    'grid.linewidth':   0.5,  'grid.alpha':     0.5,
    'grid.linestyle':   '--', 'font.size':      8,
})

def draw_candles(ax, df, x0=0):
    df = df.reset_index(drop=True)
    for i, r in df.iterrows():
        x  = i + x0
        up = r['close'] >= r['open']
        c  = GREEN if up else RED
        lo = min(r['open'], r['close'])
        hi = max(r['open'], r['close'])
        ax.add_patch(mpatches.Rectangle(
            (x - 0.38, lo), 0.76, max(hi - lo, 1e-8),
            fc=c, ec=c, alpha=0.85, zorder=2))
        ax.plot([x, x], [r['low'], lo],  c, lw=0.7, zorder=1)
        ax.plot([x, x], [hi, r['high']], c, lw=0.7, zorder=1)
    pad = (df['high'].max() - df['low'].min()) * 0.025
    ax.set_xlim(x0 - 1, x0 + len(df))
    ax.set_ylim(df['low'].min() - pad, df['high'].max() + pad)

def style_ax(ax, title=''):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.grid(color=GRID, lw=0.5, ls='--', alpha=0.5)
    if title:
        ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=5)

def time_ticks(ax, df, step=10, x0=0, fmt='%m/%d %H:%M'):
    idx = list(range(0, len(df), step))
    ax.set_xticks([i + x0 for i in idx])
    ax.set_xticklabels(
        [pd.to_datetime(df.iloc[i]['time_key']).strftime(fmt) for i in idx],
        rotation=30, ha='right', fontsize=6)

print('Setup complete')
