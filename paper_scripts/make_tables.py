"""LaTeX tables for the VILLE paper.  Run: python -I make_tables.py <results_dir> <out_dir>"""
import sys, os, json
import numpy as np

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(RES, name)
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith('_')} if os.path.exists(p) else {}


def gm(x):
    x = [v for v in x if v > 0]
    return float(np.exp(np.mean(np.log(x)))) if x else float('nan')


D = [('AMZ-Stars', load('d1_main.json')), ('AMZ-LLM', load('d2_wor.json')), ('ArXiv-LLM', load('d3_arxiv.json'))]
KIND = [('having', 'HAVING'), ('order', 'ORDER BY'), ('topk', 'Top-$k$')]
COLS = [('WSR', 'Evergreen-CS'), ('WSR-P', 'Evergreen-CS+P'), ('BS-P', 'BlazeIt-EBS'), ('HS', 'IFOCUS-HS'), ('BET', '\\sys{}$-$P')]
L = ['\\begin{table*}[t]', '\\centering', '\\caption{LLM calls relative to \\sys{}: geometric mean over queries of each method\'s calls divided by \\sys{}\'s calls ($>1$: more calls than \\sys{}). The \\sys{} column gives the median over queries of the fraction of tuples it judged; CLT+P is uncertified.}',
     '\\label{tab:main}', '\\setlength{\\tabcolsep}{3pt}', '\\footnotesize',
     '\\begin{tabular}{llrr' + 'r' * len(COLS) + 'r}', '\\toprule',
     'Data & Query & $n$ & \\sys{} & ' + ' & '.join(c[1].replace('Evergreen-CS+P', 'Everg.+P').replace('Evergreen-CS', 'Everg.').replace('BlazeIt-EBS', 'BlazeIt').replace('IFOCUS-HS', 'IFOCUS') for c in COLS) + ' & CLT+P \\\\',
     '\\midrule']
tot = {c: [] for c, _ in COLS}; totc = []
for dn, res in D:
    if not res:
        continue
    for ki, (kind, kl) in enumerate(KIND):
        rows = [v['summary'] for k, v in res.items() if k.endswith('|' + kind)]
        if not rows:
            continue
        cells = []
        for c, _ in COLS:
            rat = [r[c]['cost'] / r['BET-P']['cost'] for r in rows if r['BET-P']['cost'] > 0]
            tot[c] += rat; cells.append(f'{gm(rat):.1f}$\\times$')
        cr = [r['CLT-P']['cost'] / r['BET-P']['cost'] for r in rows if r['BET-P']['cost'] > 0]
        totc += cr
        med = np.median([r['BET-P']['cost'] for r in rows])
        L.append(f'{dn if ki == 0 else ""} & {kl} & {len(rows)} & {100 * med:.1f}\\% & ' + ' & '.join(cells) + f' & {gm(cr):.2f}$\\times$ \\\\')
    L.append('\\midrule')
L.append(f'All & & {len(totc)} & & ' + ' & '.join(f'{gm(tot[c]):.1f}$\\times$' for c, _ in COLS) + f' & {gm(totc):.2f}$\\times$ \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
open(os.path.join(OUT, 'tab_main.tex'), 'w', encoding='utf-8').write('\n'.join(L))

# validity table
M = [('BET-P', '\\sys{}'), ('BET', '\\sys{}$-$P'), ('WSR', 'Evergreen-CS'), ('WSR-P', 'Evergreen-CS+P'), ('BS-P', 'BlazeIt-EBS'), ('HS', 'IFOCUS-HS'), ('CLT', 'CLT (uncert.)'), ('CLT-P', 'CLT+P (uncert.)')]
NQ = sum(len(res) for _, res in D)
V = ['\\begin{table}[t]', '\\centering', '\\caption{Errors over all ' + str(NQ) + ' queries, each executed 30 times. Answer errors violate the returned-result definition of $r$-correctness. Decision errors concern a group or pair used by the answer, at least $r$ from its boundary, certified on the wrong side or as indifferent. The last column is the highest per-query answer-error rate.}',
     '\\label{tab:validity}', '\\footnotesize', '\\begin{tabular}{lrrr}', '\\toprule',
     'Method & Answers & Decisions & Worst rate \\\\', '\\midrule']
allres = {}
for dn, res in D:
    allres.update({dn + ':' + k: v for k, v in res.items()})
R = 30
for m, ml in M:
    errs = [v['summary'][m]['answer_err'] for v in allres.values()]
    decision_errors = sum(v['summary'][m]['decision_err'] * R for v in allres.values())
    n_exec = int(round(sum(e * R for e in errs)))
    V.append(f'{ml} & {n_exec} / {R * len(errs)} & {int(round(decision_errors))} / {R * len(errs)} & {max(errs):.2f} \\\\')
V += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
open(os.path.join(OUT, 'tab_validity.tex'), 'w', encoding='utf-8').write('\n'.join(V))
print('\n'.join(L)); print('\n'.join(V))
