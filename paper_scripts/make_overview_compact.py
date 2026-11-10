"""Draw VILLE's original flow and visual explanations of its mechanisms.

Run with python -I paper/make_overview_compact.py. All artwork is original.
Linked marks and the wealth path are schematic, not experimental measurements.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as VectorPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'figure_tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment
WIDTH_PT, HEIGHT_PT = 252, 364
INK, MUTED = '#243547', '#617184'
BLUE, ORANGE, TEAL = '#346EAA', '#B75B32', '#247C70'
LIGHT = '#D9E2EA'
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.0, 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})


def text(ax, x, y, value, size=8.0, color=INK, bold=False, ha='center'):
    return ax.text(x, y, value, fontsize=size, color=color,
                   fontweight='bold' if bold else 'normal',
                   ha=ha, va='center', zorder=6)


def box(ax, x, y, width, height, fill='white', edge=LIGHT, lw=0.6, radius=2.4):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(patch)
    return patch


def line(ax, xs, ys, color=LIGHT, lw=0.65, dashed=False):
    ax.plot(xs, ys, color=color, linewidth=lw,
            linestyle=(0, (2, 2)) if dashed else 'solid',
            solid_capstyle='round', zorder=3)


def arrow(ax, points, color=INK, dashed=False, lw=0.8, scale=5):
    path = VectorPath(points, [VectorPath.MOVETO] +
                      [VectorPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>',
                               mutation_scale=scale, linewidth=lw, color=color,
                               linestyle=(0, (3, 2)) if dashed else 'solid',
                               capstyle='round', joinstyle='round', zorder=4))


def dot(ax, x, y, r=1.8, fill=BLUE, edge='white', lw=0.45):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge,
                       linewidth=lw, zorder=5))


def panel(fig, name, y, height, letter, title, color):
    ax = fig.add_axes([3 / WIDTH_PT, y / HEIGHT_PT,
                       246 / WIDTH_PT, height / HEIGHT_PT], label=name)
    ax.set(xlim=(0, 246), ylim=(0, height))
    ax.set_axis_off()
    box(ax, 0.5, 0.5, 245, height - 1, edge=LIGHT, lw=0.55)
    box(ax, 7, height - 18, 13, 12, fill=color, edge=color, lw=0, radius=2)
    text(ax, 13.5, height - 12, letter, size=7.8, color='white', bold=True)
    text(ax, 26, height - 12, title, size=8.6, color=color, bold=True, ha='left')
    return ax


def draw_workflow(fig):
    ax = fig.add_axes([0, 279 / HEIGHT_PT, 1, 85 / HEIGHT_PT], label='workflow')
    ax.set(xlim=(0, 252), ylim=(0, 85))
    ax.set_axis_off()
    box(ax, 91, 24, 160, 49, fill='#F5F8FB', edge='#D7E2EC', lw=0.5)
    box(ax, 3, 62, 13, 12, fill=MUTED, edge=MUTED, lw=0, radius=2)
    text(ax, 9.5, 68, 'A', size=7.8, color='white', bold=True)
    text(ax, 21, 68, 'Query flow', size=7.8, color=MUTED, bold=True, ha='left')
    text(ax, 98, 67, 'VILLE', size=7.8, color='#49647E', bold=True, ha='left')
    modules = [
        (3, 38, 'Group $g$', 'uniform', 'WoR', MUTED, '#F0F3F6'),
        (47, 38, 'LLM', 'batch', '$y_s$', MUTED, '#F0F3F6'),
        (97, 46, 'Rectify', 'B', '$z_s$', BLUE, '#E9F1FA'),
        (149, 51, 'Boundary', 'C', '$W_{1:4}$', ORANGE, '#FBEEE5'),
        (206, 42, 'Decide', 'D', 'stop?', TEAL, '#EAF5EF'),
    ]
    for x, w, title, row1, row2, color, tint in modules:
        card = box(ax, x, 29, w, 32, edge='#B8C8D6')
        band = Rectangle((x, 48), w, 13, facecolor=tint, edgecolor='none', zorder=2.2)
        band.set_clip_path(card)
        ax.add_patch(band)
        text(ax, x + w / 2, 54.2, title, size=8.1, color=color, bold=True)
        if row1 in ('B', 'C', 'D'):
            text(ax, x + 9, 38.5, row1, size=7.2, color=color, bold=True)
            text(ax, x + w / 2 + 6, 38.5, row2, size=8.8, color=color)
        else:
            text(ax, x + w / 2, 41.6, row1, size=7.8)
            text(ax, x + w / 2, 33.5, row2, size=8.0)
    for start, end in [(41, 47), (85, 97), (143, 149), (200, 206)]:
        arrow(ax, [(start, 40), (end - 0.6, 40)])
    arrow(ax, [(227, 61), (227, 76), (66, 76), (66, 62)], color=MUTED, dashed=True)
    text(ax, 146.5, 80, 'Unresolved group → next batch', size=7.6, color=MUTED)
    box(ax, 3, 1, 245, 18, fill='#F2F6FC', edge='#B8CDE1', lw=0.55)
    text(ax, 11, 10, r'Precompute $p(t)$ for all tuples and $\overline{p}_g$',
         size=8.0, color=BLUE, ha='left')
    arrow(ax, [(120, 19), (120, 28)], color=BLUE)
    return ax


def draw_rectification(fig):
    ax = panel(fig, 'rectification', 191, 82, 'B', 'Mean-preserving rectification', BLUE)
    # Symbolic paired geometry. Equal means encode the population identity.
    # No numeric scale, measured data, variance estimate or benchmark is shown.
    raw = [-36, -25, -16, -6, 6, 16, 25, 36]
    corrected = [-21, -15, -9, -3, 3, 9, 15, 21]
    assert sum(raw) == sum(corrected) == 0
    center = 67
    for yy, zz in zip(raw, corrected):
        line(ax, [center + yy, center + zz], [52, 33], color='#C6D6E8', lw=0.7)
        dot(ax, center + yy, 52, r=1.9, fill='#BAC8D5', edge='#7D94A8')
        dot(ax, center + zz, 33, r=1.9, fill=BLUE)
    line(ax, [center, center], [27, 59], color=BLUE, lw=0.7, dashed=True)
    text(ax, 13, 52, '$y$', size=10, color=MUTED)
    text(ax, 13, 33, '$z$', size=10, color=BLUE)
    text(ax, center + 6, 59, r'$\mu_g$', size=9, color=BLUE)
    text(ax, 66, 21, 'Informative proxy illustrated', size=7.1, color=MUTED)
    formula = text(ax, 180, 51.5, r'$z_s=y_s-\beta(p_s-\overline{p}_g)$', size=9.2, color=BLUE)
    text(ax, 180, 36.8, r'$\overline{p}_g=\mathrm{mean}_{t\in g}\,p(t)$', size=8.7)
    text(ax, 180, 24, r'$\beta$ fixed before query', size=7.8, color=MUTED)
    text(ax, 123, 9.6, r'Same population mean $\quad\mathrm{mean}_{t\in g}\,z(t)=\mu_g$',
         size=8.8, color=BLUE)
    return ax, formula


def draw_betting(fig):
    ax = panel(fig, 'betting', 91, 94, 'C', 'Finite-population betting', ORANGE)
    text(ax, 204, 82, r'$a=\alpha/(2K)$', size=8.8, color=ORANGE)
    text(ax, 69, 65.5, r'$\overline{x}_g\leq\theta,\quad x_s=\pm z_s$', size=8.8)
    text(ax, 69, 46, r'$m_s=\frac{N_g\theta-S_{s-1}}{N_g-s+1}$', size=10.5)
    text(ax, 69, 28.7, r'Past-based, capped $\lambda_s$', size=7.6, color=MUTED)
    # A qualitative wealth glyph illustrates a rejection event, not its speed.
    x0, x1, bottom, threshold = 152, 233, 31, 61
    arrow(ax, [(x0, bottom), (x0, 72)], color='#9CABB8', lw=0.55, scale=3)
    arrow(ax, [(x0, bottom), (x1 + 5, bottom)], color='#9CABB8', lw=0.55, scale=3)
    line(ax, [x0, x1 + 2], [threshold, threshold], color=ORANGE, lw=0.65, dashed=True)
    text(ax, x0 - 4, 35, '1', size=7.0, color=MUTED, ha='right')
    text(ax, x0 - 4, threshold, '$1/a$', size=8.3, color=ORANGE, ha='right')
    text(ax, 167, 71, '$W_h$', size=8.8, color=ORANGE)
    text(ax, 201, 72, 'schematic', size=7.0, color=MUTED)
    text(ax, 231, 72, 'reject', size=7.0, color=ORANGE)
    xs = [152, 160, 167, 175, 183, 190, 198, 205, 213, 221]
    ys = [35, 38, 36, 42, 46, 44, 51, 56, 54, 65]
    line(ax, xs, ys, color=ORANGE, lw=1.25)
    dot(ax, 221, 65, r=2.3, fill=ORANGE)
    text(ax, 196, 23.7, 'judged prefix', size=7.1, color=MUTED)
    text(ax, 123, 10.8, r'$W_s=W_{s-1}\,[1+\lambda_s(x_s-m_s)]$', size=10, color=ORANGE)
    return ax


def draw_decisions(fig):
    ax = panel(fig, 'decisions', 14, 71, 'D', 'On rejection → group decision', TEAL)
    text(ax, 17, 32.5, '$z_s$', size=11, color=BLUE)
    arrow(ax, [(24, 32.5), (33, 32.5)], color=BLUE, lw=0.7, scale=4)
    line(ax, [34, 34], [8.5, 47.5], color=BLUE, lw=0.65)
    rows = [(47.5, 1, r'$\mu_g\leq c$'), (34.5, 2, r'$\mu_g\geq c$'),
            (21.5, 3, r'$\mu_g\geq c+r$'), (8.5, 4, r'$\mu_g\leq c-r$')]
    for y, n, null in rows:
        arrow(ax, [(34, y), (43, y)], color=BLUE, lw=0.6, scale=3.7)
        text(ax, 53, y, rf'$H_{n}$', size=9.1, color=TEAL)
        text(ax, 98, y, null, size=9.2)
    for y, label in [(47.5, 'Above'), (34.5, 'Below')]:
        arrow(ax, [(132, y), (177, y)], color=TEAL, lw=0.7, scale=4)
        box(ax, 181, y - 5.2, 57, 10.4, fill='#EDF7F2', edge='#A8CDBE', lw=0.45)
        text(ax, 209.5, y, label, size=7.8, color=TEAL, bold=True)
    line(ax, [132, 145, 145, 132], [21.5, 21.5, 8.5, 8.5], color=TEAL, lw=0.75)
    arrow(ax, [(145, 15), (177, 15)], color=TEAL, lw=0.7, scale=4)
    text(ax, 159, 23.8, 'both', size=7.1, color=TEAL, bold=True)
    box(ax, 181, 9.8, 57, 10.4, fill='#EDF7F2', edge='#A8CDBE', lw=0.45)
    text(ax, 209.5, 15, r'Within $r$', size=7.8, color=TEAL, bold=True)
    text(ax, 123, -7.8, r'Certified → stop $g$     Exhausted → exact side', size=7.8, color=TEAL)
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / 'overview_compact_qa'
    qa.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), facecolor='white')
    draw_workflow(fig)
    rectification, formula = draw_rectification(fig)
    betting = draw_betting(fig)
    decisions = draw_decisions(fig)
    fig.canvas.draw()
    width = formula.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 72
    if width > 113:
        raise ValueError(f'Rectification formula too wide: {width:.2f} pt')
    require_matplotlib_panel_alignment(
        fig, axes=[rectification, betting, decisions],
        panel_ids=['rectification', 'betting', 'decisions'],
        column_groups=[['rectification', 'betting', 'decisions']],
        json_out=qa / 'alignment.json', overlay_svg=qa / 'alignment_overlay.svg', strict=True)
    fig.savefig(args.out / 'overview_compact.pdf')
    fig.savefig(args.out / 'overview_compact.svg')
    fig.savefig(args.out / 'overview_compact.png', dpi=600)
    plt.close(fig)
    print(f'Rendered visual mechanism at {WIDTH_PT} by {HEIGHT_PT} pt')
    print(f'Rectification formula width {width:.2f} pt')


if __name__ == '__main__':
    main()
