"""Figures for the VILLE paper (matplotlib).  Run: python -I make_figs.py <results_dir> <out_dir>"""
import sys, os, json, math
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / 'figure_tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2, 'legend.fontsize': 6.5, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.0,
})
# method families: hero (VILLE) in a saturated accent, baselines in muted tones
C = {'BET-P': '#C2410C', 'BET': '#F59E0B', 'WSR': '#1D4ED8', 'WSR-P': '#60A5FA', 'BS-P': '#6B7280',
     'HS': '#111827', 'CLT-P': '#9CA3AF'}
LABEL = {'BET-P': 'VILLE', 'BET': 'VILLE$-$P', 'WSR': 'Evergreen-CS', 'WSR-P': 'Evergreen-CS+P',
         'BS-P': 'BlazeIt-EBS', 'HS': 'IFOCUS-HS', 'CLT-P': 'CLT+P (uncertified)'}
KINDS = [('having', 'HAVING', 'o'), ('order', 'ORDER BY', 's'), ('topk', 'Top-$k$', '^')]
KCOL = {'having': '#0072B2', 'order': '#D55E00', 'topk': '#009E73'}   # Okabe-Ito


def load(name):
    p = os.path.join(RES, name)
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith('_')} if os.path.exists(p) else {}


def save(fig, stem):
    require_matplotlib_panel_alignment(fig, json_out=os.path.join(OUT, stem + '.alignment.json'), tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(os.path.join(OUT, stem + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, stem + '.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, stem + '.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def gmean(x):
    x = [v for v in x if v > 0]
    return float(np.exp(np.mean(np.log(x)))) if x else float('nan')


from matplotlib.lines import Line2D
D = {k: v for k, v in (('AMZ-Stars', load('d1_main.json')), ('AMZ-LLM', load('d2_wor.json')),
                       ('ArXiv-LLM', load('d3_arxiv.json'))) if v}
DCOL = {'AMZ-Stars': '#0072B2', 'AMZ-LLM': '#D55E00', 'ArXiv-LLM': '#009E73'}   # Okabe-Ito

# ---------------- Fig. main: per-query cost, VILLE vs certified baselines ----------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True)
lo = 2e-4
for ax, base in zip(axes, ['WSR', 'BS-P', 'HS']):
    for dname, res in D.items():
        for kind, kl, mk in KINDS:
            xs = [max(v['summary']['BET-P']['cost'], lo) for k, v in res.items() if k.endswith('|' + kind)]
            ys = [max(v['summary'][base]['cost'], lo) for k, v in res.items() if k.endswith('|' + kind)]
            ax.scatter(xs, ys, s=13, marker=mk, facecolors=DCOL[dname], edgecolors='white', linewidths=0.35,
                       alpha=0.9, zorder=3)
    g = np.array([lo, 1.3])
    ax.plot(g, g, color='#6B7280', lw=0.7, zorder=1)
    ax.plot(g, np.minimum(10 * g, 1.3), color='#6B7280', lw=0.7, ls='--', zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, 1.3); ax.set_ylim(lo, 1.3)
    ax.set_xlabel('VILLE: fraction of tuples judged')
    ax.set_title(LABEL[base], fontsize=7.5, pad=3)
    ax.text(0.33, 0.95, '10$\\times$', transform=ax.transAxes, fontsize=6, color='#6B7280', va='top')
axes[0].set_ylabel('Baseline: fraction judged')
h = [Line2D([], [], ls='none', marker=mk, ms=4, mfc='#6B7280', mec='white', mew=0.35, label=kl) for _, kl, mk in KINDS]
h += [Line2D([], [], ls='none', marker='o', ms=4.5, mfc=DCOL[d], mec='white', mew=0.35, label=d) for d in D]
fig.legend(handles=h, loc='lower center', ncol=len(h), bbox_to_anchor=(0.5, -0.12), handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(w_pad=1.0)
save(fig, 'fig_main_scatter')

# ---------------- Fig. decomposition: geometric-mean cost per dataset ----------------
meths = ['WSR', 'WSR-P', 'BET', 'BET-P', 'CLT-P']
fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.9))
w = 0.16; base = 4e-3                     # log axis: bars start at the axis floor, not at zero
names = list(D)
for mi, m in enumerate(meths):
    vals = [gmean([v['summary'][m]['cost'] for v in D[d].values()]) for d in names]
    ax.bar(np.arange(len(names)) + (mi - 2) * w, np.array(vals) - base, w * 0.92, bottom=base,
           color='#D1D5DB' if m == 'CLT-P' else C[m], label=LABEL[m], zorder=2,
           edgecolor='#6B7280' if m == 'CLT-P' else 'none', linewidth=0.5)
ax.set_ylim(base, 1.0)
ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names)
ax.set_yscale('log'); ax.set_ylabel('Geo.-mean fraction judged')
ax.grid(axis='y', color='#E5E7EB', lw=0.5, zorder=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout()
save(fig, 'fig_decomposition')

# ---------------- Fig. sensitivity ----------------
P, M, G = load('abl_proxy.json'), load('abl_margin.json'), load('abl_groups.json')
if P and M and G:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    ax = axes[0]
    sizes = [100, 300, 1000, 5000]
    for m in ['BET-P', 'BET', 'WSR-P', 'WSR']:
        ys = [gmean([v['summary'][m]['cost'] for k, v in P.items() if k.startswith(f'proxy{n}|')]) for n in sizes]
        ax.plot(sizes, ys, marker='o', ms=3, color=C[m], label=LABEL[m])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Labeled reviews for the proxy')
    ax.set_ylabel('Geo.-mean fraction judged'); ax.set_title('(a) Proxy quality', fontsize=7.5, pad=3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.6, columnspacing=0.8)
    ax = axes[1]
    rs = [0.025, 0.05, 0.10]
    for m in ['BET-P', 'WSR', 'BS-P', 'HS']:
        ys = [gmean([v['summary'][m]['cost'] for k, v in M.items() if k.startswith(f'r{r}|')]) for r in rs]
        ax.plot([100 * r for r in rs], ys, marker='o', ms=3, color=C[m], label=LABEL[m])
    ax.set_yscale('log'); ax.set_xlabel('Margin $r$ (% of overall mean)'); ax.set_title('(b) Indifference margin', fontsize=7.5, pad=3)
    ax.set_xticks([2.5, 5, 10])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.6, columnspacing=0.8)
    ax = axes[2]
    Ks = [10, 20, 30]
    for m in ['BET-P', 'WSR']:
        for kind, ls in (('order', '-'), ('topk', '--')):
            ys = [gmean([v['summary'][m]['cost'] for k, v in G.items() if k.startswith(f'K{K}|') and k.endswith(kind)]) for K in Ks]
            ax.plot(Ks, ys, marker='o', ms=3, ls=ls, color=C[m], label=f'{LABEL[m]}, {"ORDER BY" if kind == "order" else "Top-5"}')
    ax.set_yscale('log'); ax.set_xlabel('Number of products $K$'); ax.set_title('(c) Number of groups', fontsize=7.5, pad=3)
    ax.set_xticks(Ks)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig_sensitivity')
print('figures written to', OUT)
