"""Draw Figure 1: VILLE executing one HAVING group (full-width schematic).

Run:  python -I make_overview_mechanism.py [--out DIR]

Every mark is schematic.  Panel (b) draws an exact toy example: eight 0/1 LLM
judgments y, proxy scores p and beta = 1, so z = y - beta (p - mean(p)) has the
same mean as y.  The wealth paths in panel (c) are illustrative.
Outputs: overview_mechanism.pdf / .svg (editable text) / .png (600 dpi) and an
alignment report for the three lower panels in overview_mechanism_qa/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'figure_tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

W, H = 516, 216                      # final size in points (IEEE text width)
FLOW_SHIFT = 8.0                     # panel (a) is drawn in design units 8 pt above its place
INK, GRAPHITE, RULE, GROUP = '#262626', '#5E5E5E', '#CACACA', '#8A8A8A'
BLUE, BLUE_PALE = '#3A6C9C', '#DCE6F1'
OCHRE, OCHRE_PALE, OCHRE_INK = '#B8862F', '#F3E8D3', '#8A6420'
MAROON, MAROON_PALE = '#8B2E45', '#F4E4E8'
PURPLE, LAVENDER = '#6B4E8E', '#ECE6F3'
GRAY_FILL, GRAY_EDGE = '#EDEDED', '#A8A8A8'
CAP, LAB, SMALL = 8.0, 7.5, 7.2      # three type sizes; mathtext sub/superscripts stay >= 5 pt

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': LAB,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})


# ----------------------------------------------------------------------------- primitives
def text(ax, x, y, s, size=LAB, color=INK, ha='center', va='center', bold=False, italic=False, rot=0):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, rotation=rot, rotation_mode='anchor', zorder=8,
                   fontweight='bold' if bold else 'normal', fontstyle='italic' if italic else 'normal')


def polyline(ax, pts, color=GRAPHITE, lw=0.75, ls='solid', z=4):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle='butt', dash_capstyle='butt', zorder=z)


def head(ax, p0, p1, color=GRAPHITE, lw=0.75, z=5):
    """Arrowhead on the segment p0 -> p1 (solid, 3.4 pt long)."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>,head_length=3.4,head_width=1.7', mutation_scale=1,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))


def arrow(ax, pts, color=GRAPHITE, lw=0.75, ls='solid'):
    if len(pts) > 2 or ls != 'solid':
        polyline(ax, pts[:-1] + [pts[-1]], color=color, lw=lw, ls=ls)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        d = np.hypot(x1 - x0, y1 - y0)
        k = min(4.0, d) / d
        head(ax, (x1 - (x1 - x0) * k, y1 - (y1 - y0) * k), (x1, y1), color=color, lw=lw)
    else:
        head(ax, pts[0], pts[1], color=color, lw=lw)


def rect(ax, x, y, w, h, fill='white', edge=GRAPHITE, lw=0.7, ls='solid', z=3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=lw, ls=ls, zorder=z))


def cells(ax, x, y, n, w, h, gap, fills, edges, lw=0.5):
    for i in range(n):
        rect(ax, x + i * (w + gap), y, w, h, fill=fills[i], edge=edges[i], lw=lw)
    return x + n * w + (n - 1) * gap


def trapezoid(ax, x0, x1, yc, h0, h1, fill, edge, label):
    ax.add_patch(Polygon([(x0, yc - h0 / 2), (x0, yc + h0 / 2), (x1, yc + h1 / 2), (x1, yc - h1 / 2)],
                         closed=True, facecolor=fill, edgecolor=edge, lw=0.75, zorder=3))
    text(ax, (x0 + x1) / 2 - 1, yc, label)


def panel_axes(fig, x, y, w, h, name):
    ax = fig.add_axes([x / W, y / H, w / W, h / H], label=name)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    return ax


# ----------------------------------------------------------------------------- (a) execution flow
def draw_flow(ax, dx=12.0):
    """Top band.  dx shifts the whole flow horizontally so that it sits centred."""
    X = lambda v: v + dx
    yc_llm, yc_proxy = 190, 148
    # group g: one shuffled order, judged prefix (4), next batch (2), unread (6)
    cw, ch, cg, x0, y0 = 7.0, 9.0, 1.5, X(10.0), 165.0
    fills = [BLUE_PALE] * 4 + ['white'] * 2 + [GRAY_FILL] * 6
    edges = [BLUE] * 6 + [GRAY_EDGE] * 6
    x_end = cells(ax, x0, y0, 12, cw, ch, cg, fills, edges)
    text(ax, x0 + 2 * (cw + cg) - cg / 2, y0 + ch + 5.0, 'judged', size=SMALL)
    nb = x0 + 4 * (cw + cg) + cw + cg / 2                       # centre of the next batch
    arrow(ax, [(nb, y0 + ch), (nb, yc_llm), (X(132), yc_llm)])
    text(ax, nb + 3.5, 181.5, 'next $b$', size=SMALL, ha='left')
    # every tuple goes to the proxy once
    yb = y0 - 4.0
    polyline(ax, [(x0, y0 - 2.0), (x0, yb), (x_end, yb), (x_end, y0 - 2.0)], color=GRAPHITE, lw=0.6)
    xp = X(100.0)
    arrow(ax, [(xp, yb), (xp, yc_proxy), (X(132), yc_proxy)])
    text(ax, X(114.5), yc_proxy + 5.0, 'all tuples', size=SMALL)
    text(ax, x0, 153.5, 'group $g$, in random order', ha='left')

    trapezoid(ax, X(132), X(162), yc_llm, 22, 14, BLUE_PALE, BLUE, 'LLM')
    trapezoid(ax, X(132), X(162), yc_proxy, 22, 14, OCHRE_PALE, OCHRE, 'proxy')

    # judgments of the prefix and next batch, and proxy scores of all tuples
    arrow(ax, [(X(162), yc_llm), (X(169.5), yc_llm)])
    yv = [0, 1, 0, 0, 1, 0]
    y_end = cells(ax, X(170), yc_llm - 4.5, 6, 7.0, 9.0, 1.5,
                  [BLUE if v else BLUE_PALE for v in yv], ['white'] * 6, lw=0.4)
    text(ax, (X(170) + y_end) / 2, yc_llm + 10.0, '$y_t$', color=BLUE)
    arrow(ax, [(X(162), yc_proxy), (X(169.5), yc_proxy)])
    shades = ['#F6EEDD', '#EAD5AE', '#F1E3C7', '#D9B97E', '#F6EEDD', '#C99E50',
              '#EAD5AE', '#F1E3C7', '#D9B97E', '#F6EEDD', '#C99E50', '#EAD5AE']
    p_end = cells(ax, X(170), yc_proxy - 3.5, 12, 4.4, 7.0, 0.9, shades, ['white'] * 12, lw=0.4)
    text(ax, (X(170) + p_end) / 2, yc_proxy + 9.0, '$p(t)$', color=OCHRE_INK)

    # VILLE decision layer
    bx0, bx1, by0, by1 = X(234), X(446), 156, 208
    rect(ax, bx0, by0, bx1 - bx0, by1 - by0, fill='none', edge=GROUP, lw=0.8, ls=(0, (4, 2.5)), z=2)
    text(ax, bx0 + 5, by1 - 6, 'VILLE', bold=True, ha='left')
    cx, cy, cr = X(250), yc_llm, 6.0
    arrow(ax, [(y_end + 2, yc_llm), (cx - cr, yc_llm)])
    arrow(ax, [(p_end + 2, yc_proxy), (cx, yc_proxy), (cx, cy - cr)], color=OCHRE)
    text(ax, cx + 4, 168, r'$p_t,\ \bar{p}_g$', color=OCHRE_INK, ha='left')
    ax.add_patch(Circle((cx, cy), cr, facecolor='white', edgecolor=MAROON, lw=0.85, zorder=5))
    text(ax, cx, cy + 0.3, '$-$', color=MAROON)
    text(ax, cx + 9, cy - 12.5, 'rectify', size=SMALL, color=MAROON, ha='left')
    zx = X(264)
    arrow(ax, [(cx + cr, yc_llm), (zx - 0.5, yc_llm)], color=MAROON)
    zshade = ['#C98C9B', '#B5687C', '#C07A8B', '#BD7587', '#B9707F', '#C4808F']
    z_end = cells(ax, zx, yc_llm - 4.5, 6, 7.0, 9.0, 1.5, zshade, ['white'] * 6, lw=0.4)
    text(ax, (zx + z_end) / 2, yc_llm + 10.0, '$z_t$', color=MAROON)
    tx = X(328)
    arrow(ax, [(z_end + 1.5, yc_llm), (tx, yc_llm)], color=MAROON)
    rect(ax, tx, yc_llm - 14, 58, 28, fill=MAROON_PALE, edge=MAROON, lw=0.8)
    text(ax, tx + 29, yc_llm + 4.3, 'four boundary')
    text(ax, tx + 29, yc_llm - 4.7, 'e-processes')
    dx0 = X(400)
    arrow(ax, [(tx + 58, yc_llm), (dx0, yc_llm)])
    rect(ax, dx0, yc_llm - 10, 36, 20, fill='white', edge=GRAPHITE, lw=0.7)
    text(ax, dx0 + 18, yc_llm, 'decide')
    arrow(ax, [(dx0 + 36, yc_llm), (bx1 + 8, yc_llm)])
    text(ax, bx1 + 10, yc_llm + 4.3, 'answer', ha='left')
    text(ax, bx1 + 10, yc_llm - 4.7, 'for $g$', ha='left')
    # undecided groups return for the next batch (dotted = feedback)
    xl = X(147)
    arrow(ax, [(dx0 + 18, yc_llm + 10), (dx0 + 18, 214), (xl, 214), (xl, yc_llm + 11.5)], ls=(0, (1.0, 1.6)), lw=0.8)
    text(ax, (xl + dx0 + 18) / 2, 218.5, 'undecided: next batch', size=SMALL)
    text(ax, W - 4, 218.5, 'Schematic', size=SMALL, ha='right', italic=True)


# ----------------------------------------------------------------------------- (b) rectification
def draw_rectification(ax):
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1], float)
    p = np.array([0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.70, 0.80])
    z = y - 1.0 * (p - p.mean())                                # beta = 1
    assert abs(z.mean() - y.mean()) < 1e-12
    vx = lambda v: 22 + 110 * v
    text(ax, 78, 84, r'$z_t = y_t - \beta\,(p_t - \bar{p}_g)$')
    mu = vx(y.mean())
    polyline(ax, [(mu, 27), (mu, 72)], color=GRAPHITE, lw=0.6, ls=(0, (2.2, 1.8)))
    text(ax, mu, 76, r'$\mu_g$', size=SMALL)
    stack = {0: 0, 1: 0}
    for v in y:                                                  # 0/1 judgments, stacked
        ax.add_patch(Circle((vx(v), 48 + 4.4 * stack[v]), 1.9, facecolor=BLUE, edgecolor='white', lw=0.3, zorder=5))
        stack[v] += 1
    for v in z:
        ax.add_patch(Circle((vx(v), 34), 1.9, facecolor=MAROON, edgecolor='white', lw=0.3, zorder=5))
    text(ax, 8, 54, '$y_t$', color=BLUE)
    text(ax, 8, 34, '$z_t$', color=MAROON)
    polyline(ax, [(14, 24), (142, 24)], color=GRAPHITE, lw=0.6)
    for v in (0, 1):
        polyline(ax, [(vx(v), 24), (vx(v), 21.5)], color=GRAPHITE, lw=0.6)
        text(ax, vx(v), 16.5, str(v), size=SMALL)


# ----------------------------------------------------------------------------- (c) boundary e-process
def draw_test(ax):
    text(ax, 78, 84, r'$W_t = W_{t-1}\,[\,1+\lambda_t\,(z_t-m_t)\,]$')
    x0, y0, x1, y1 = 16, 22, 146, 74
    arrow(ax, [(x0, y0), (x1, y0)], lw=0.6)
    arrow(ax, [(x0, y0), (x0, y1)], lw=0.6)
    text(ax, 7, 48, r'$\log W_t$', size=SMALL, rot=90)
    text(ax, 81, 14, 'judged tuples $t$', size=SMALL)
    thr = 62
    polyline(ax, [(x0, thr), (x1 - 2, thr)], color=GRAPHITE, lw=0.6, ls=(0, (2.2, 1.8)))
    text(ax, x0 + 3, thr + 4.5, r'$\log(1/a)$', size=SMALL, ha='left')
    zt = [(16, 26), (26, 28), (34, 26.5), (44, 33), (54, 31), (64, 39), (74, 44), (82, 42),
          (92, 51), (102, 57), (110, 55.5), (118, 63.6)]
    yt = [(16, 26), (30, 27.5), (44, 29.5), (58, 28.4), (72, 33), (86, 35.5), (100, 38.5),
          (114, 42.5), (128, 44.5), (140, 48.5)]
    polyline(ax, yt, color=BLUE, lw=0.95, ls=(0, (3.0, 1.6)), z=5)
    polyline(ax, zt, color=MAROON, lw=1.1, z=6)
    ax.add_patch(Circle(zt[-1], 1.9, facecolor=MAROON, edgecolor='white', lw=0.3, zorder=7))
    text(ax, zt[-1][0], zt[-1][1] + 5.5, 'reject', size=SMALL, color=MAROON)
    text(ax, 76, 55, 'on $z_t$', size=SMALL, color=MAROON)
    text(ax, 136, 54, 'on $y_t$', size=SMALL, color=BLUE)


# ----------------------------------------------------------------------------- (d) decision
def draw_decision(ax):
    lo, hi, c, r = 64, 124, 94, 16
    text(ax, 78, 84, r'all $N_g$ judged: exact side', size=SMALL)
    for xv in (c - r, c, c + r):
        polyline(ax, [(xv, 24), (xv, 76)], color=RULE, lw=0.45, ls=(0, (1, 1.5)), z=1)
    rows = [(70, [r'reject $H^{\leq}_g(c)$'], (c, hi), 'above'),
            (55, [r'reject $H^{\geq}_g(c)$'], (lo, c), 'below'),
            (38, [r'reject $H^{\geq}_g(c{+}r)$', r'and $H^{\leq}_g(c{-}r)$'], (c - r, c + r), 'within $r$')]
    for yr, labels, (a, b), out in rows:
        for k, s in enumerate(labels):
            text(ax, 60, yr + (4.6 if len(labels) == 2 else 0) - 9.2 * k, s, size=SMALL, ha='right')
        polyline(ax, [(lo, yr), (hi, yr)], color=RULE, lw=0.6)
        if out == 'within $r$':
            rect(ax, a, yr - 4, b - a, 8, fill=LAVENDER, edge='none', lw=0, z=2)
        polyline(ax, [(a, yr), (b, yr)], color=MAROON if out != 'within $r$' else PURPLE, lw=1.6, z=5)
        text(ax, 128, yr, out, size=SMALL, ha='left', color=INK)
    polyline(ax, [(lo, 24), (hi, 24)], color=GRAPHITE, lw=0.6)
    for xv, s in ((c - r, '$c{-}r$'), (c, '$c$'), (c + r, '$c{+}r$')):
        polyline(ax, [(xv, 24), (xv, 21.5)], color=GRAPHITE, lw=0.6)
        text(ax, xv, 16.5, s, size=SMALL)
    text(ax, hi + 5, 24, r'$\mu_g$', size=SMALL, ha='left')


# ----------------------------------------------------------------------------- assembly
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'figures')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / 'overview_mechanism_qa'
    qa.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(W / 72, H / 72), facecolor='white')
    base = panel_axes(fig, 0, 0, W, H, 'canvas')            # rule, separators, lower captions
    top = fig.add_axes([0, 0, 1, 1], label='flow')          # panel (a), drawn in design units
    top.set_xlim(0, W)
    top.set_ylim(FLOW_SHIFT, H + FLOW_SHIFT)                 # design y maps to figure y - FLOW_SHIFT
    top.set_axis_off()
    draw_flow(top)
    polyline(base, [(6, 108), (W - 6, 108)], color=RULE, lw=0.6)
    for xs in (171, 345):
        polyline(base, [(xs, 20), (xs, 98)], color=RULE, lw=0.6)
    pw, py, ph, lo = 156, 18, 82, 6.0                       # panel content uses local y 6..88
    axes = []
    for x, n in ((6, 'rectification'), (180, 'test'), (354, 'decision')):
        ax = panel_axes(fig, x, py, pw, ph, n)
        ax.set_ylim(lo, lo + ph)
        axes.append(ax)
    draw_rectification(axes[0])
    draw_test(axes[1])
    draw_decision(axes[2])

    text(top, W / 2, 125.5, '(a) Executing one HAVING group', size=CAP, bold=True)
    caps = [text(base, x + pw / 2, 8, s, size=CAP, bold=True)
            for x, s in ((6, '(b) Rectification'), (180, '(c) Boundary e-process'), (354, '(d) Decision'))]
    fig.canvas.draw()
    inv = base.transData.inverted()
    ext = [inv.transform(t.get_window_extent(fig.canvas.get_renderer())) for t in caps]
    for left, right in zip(ext[:-1], ext[1:]):
        arrow(base, [(left[1][0] + 8, 8), (right[0][0] - 8, 8)], lw=0.7)

    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=['rectification', 'test', 'decision'],
        row_groups=[['rectification', 'test', 'decision']],
        json_out=qa / 'alignment.json', overlay_svg=qa / 'alignment_overlay.svg', strict=True)
    fig.savefig(args.out / 'overview_mechanism.pdf')
    svg = args.out / 'overview_mechanism.svg'
    fig.savefig(svg)
    # drop the external DTD reference so the SVG has no external dependency
    lines = svg.read_text(encoding='utf-8').splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith('<!DOCTYPE'))
    del lines[k:k + 2]
    svg.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    fig.savefig(args.out / 'overview_mechanism.png', dpi=600)
    plt.close(fig)
    print(f'Rendered {W} x {H} pt to {args.out}')


if __name__ == '__main__':
    main()
