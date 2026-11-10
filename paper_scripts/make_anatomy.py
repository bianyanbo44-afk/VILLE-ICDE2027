"""Anatomy-of-a-decision figure.  Run: python -I make_anatomy.py <anatomy.json> <out_dir>"""
import sys, os, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent / 'figure_tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment

A = json.load(open(sys.argv[1])); OUT = sys.argv[2]
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'pdf.fonttype': 42,
                     'font.size': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
                     'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.6, 'legend.frameon': False})
t = np.array(A['t']); xmax = 1.12 * A['fire_cs']
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8))
ax = axes[0]
# each test stops sampling when it fires, so its wealth is drawn up to that point
for key, fire, col, lab in (('logwealth_ville', A['fire_ville'], '#C2410C', 'VILLE'),
                            ('logwealth_ville_noproxy', A['fire_ville_noproxy'], '#F59E0B', 'VILLE$-$P')):
    m = t <= fire
    ax.plot(t[m], np.clip(np.array(A[key])[m], -1.4, None), color=col, lw=1.1, label=f'{lab} ({fire:,})')
    ax.plot([fire], [np.array(A[key])[m][-1]], marker='o', ms=3, color=col)
ax.axhline(A['threshold'], color='#6B7280', lw=0.7, ls='--')
ax.text(xmax * 0.98, A['threshold'] + 0.25, 'reject at log(1/a)', ha='right', va='bottom', fontsize=6.5, color='#6B7280')
ax.set_xlim(0, xmax); ax.set_ylim(-2.2, A['threshold'] * 1.45)
ax.set_xlabel('LLM calls'); ax.set_ylabel('log-wealth')
ax.set_title('(a) Boundary e-processes', fontsize=7.5, pad=3)
ax.legend(loc='lower right', handlelength=1.4)
ax = axes[1]
ylo, yhi = A['mu'] - 0.06, A['c'] + 0.08
m = t <= A['fire_cs']
lo, up = np.clip(np.array(A['cs_lower'])[m], ylo, yhi), np.clip(np.array(A['cs_upper'])[m], ylo, yhi)
ax.fill_between(t[m], lo, up, color='#93C5FD', alpha=0.6, lw=0)
ax.plot(t[m], up, color='#1D4ED8', lw=0.9)
ax.plot(t[m], lo, color='#1D4ED8', lw=0.9)
ax.plot([A['fire_cs']], [up[-1]], marker='o', ms=3, color='#1D4ED8')
ax.axhline(A['c'], color='#6B7280', lw=0.7, ls='--')
ax.axhline(A['mu'], color='#111827', lw=0.7, ls='-')
ax.text(xmax * 0.55, A['c'] - 0.003, 'threshold $c$', ha='center', va='top', fontsize=6.5, color='#6B7280')
ax.text(xmax * 0.03, A['mu'] - 0.003, 'true group mean', ha='left', va='top', fontsize=6.5, color='#111827')
ax.text(A['fire_cs'], A['c'] + 0.016, f"{A['fire_cs']:,} calls", ha='right', va='bottom', fontsize=6.5, color='#1D4ED8')
ax.set_xlim(0, xmax); ax.set_ylim(ylo, yhi)
ax.set_xlabel('LLM calls'); ax.set_ylabel('share of negative reviews')
ax.set_title('(b) Evergreen-CS interval', fontsize=7.5, pad=3)
fig.tight_layout(w_pad=1.0)
require_matplotlib_panel_alignment(fig, json_out=os.path.join(OUT, 'fig_anatomy.alignment.json'), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(os.path.join(OUT, 'fig_anatomy.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig_anatomy.png'), dpi=300, bbox_inches='tight')
print('ok', A['fire_ville'], A['fire_ville_noproxy'], A['fire_cs'])
