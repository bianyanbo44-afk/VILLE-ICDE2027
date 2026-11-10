"""All numbers quoted in the paper text, from the final result files.  Run: python -I numbers.py <results_dir>"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
RES = sys.argv[1]


def load(name):
    p = os.path.join(RES, name)
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith('_')} if os.path.exists(p) else {}


def gm(x):
    x = [v for v in x if v > 0]
    return float(np.exp(np.mean(np.log(x)))) if x else float('nan')


D1, D2, D3 = load('d1_main.json'), load('d2_wor.json'), load('d3_arxiv.json')
ALL = {**{'S:' + k: v for k, v in D1.items()}, **{'L:' + k: v for k, v in D2.items()}, **{'A:' + k: v for k, v in D3.items()}}
B = ['WSR', 'WSR-P', 'BS-P', 'HS', 'BET', 'CLT-P', 'CLT']
print(f'queries: AMZ-Stars {len(D1)}, AMZ-LLM {len(D2)}, ArXiv-LLM {len(D3)}, total {len(ALL)}')
for name, S in (('ALL', ALL), ('AMZ-Stars', D1), ('AMZ-LLM', D2), ('ArXiv-LLM', D3), ('LLM-judged', {**{'L:' + k: v for k, v in D2.items()}, **{'A:' + k: v for k, v in D3.items()}})):
    if not S:
        continue
    print(f'[{name}] geo-mean ratio method/VILLE:', {b: round(gm([v['summary'][b]['cost'] / v['summary']['BET-P']['cost'] for v in S.values() if v['summary']['BET-P']['cost'] > 0]), 2) for b in B})
    print(f'[{name}] VILLE <= method on:', {b: sum(v['summary']['BET-P']['cost'] <= v['summary'][b]['cost'] + 1e-12 for v in S.values()) for b in B[:5]}, 'of', len(S))
    print(f'[{name}] max over queries of method/VILLE:', {b: round(max(v['summary'][b]['cost'] / max(v['summary']['BET-P']['cost'], 1e-9) for v in S.values()), 1) for b in B[:4]})
for g in ('cat', 'year', 'prod-Amazon_Fashion', 'prod-Appliances'):
    S = {k: v for k, v in D1.items() if k.startswith(g + '|')}
    if S:
        print(f'[AMZ-Stars {g}] n={len(S)}', {b: round(gm([v['summary'][b]['cost'] / v['summary']['BET-P']['cost'] for v in S.values() if v['summary']['BET-P']['cost'] > 0]), 2) for b in B})
for g in ('year', 'area'):
    S = {k: v for k, v in D3.items() if k.startswith(g + '|')}
    if S:
        print(f'[ArXiv-LLM {g}] n={len(S)}', {b: round(gm([v['summary'][b]['cost'] / v['summary']['BET-P']['cost'] for v in S.values() if v['summary']['BET-P']['cost'] > 0]), 2) for b in B})
for k, v in D3.items():
    N = v['N']
    print('ArXiv', k, 'K', v['K'], 'N', N, {m: int(round(v['summary'][m]['cost'] * N)) for m in ('BET-P', 'WSR', 'BS-P', 'HS', 'CLT-P')})
for name, S in (('AMZ-Stars', D1), ('AMZ-LLM', D2), ('ArXiv-LLM', D3)):
    cv = [v['summary']['BET-P']['cost_sd'] / v['summary']['BET-P']['cost'] for v in S.values() if 'cost_sd' in v['summary']['BET-P'] and v['summary']['BET-P']['cost'] > 0]
    if cv:
        print(f'[{name}] VILLE cost CV across runs: median {np.median(cv):.3f} max {max(cv):.3f} (n={len(cv)})')
for k in ('year|NEG|having', 'year|POS|order', 'cat|NEG|order', 'cat|NEG|topk'):
    if k in D1:
        v = D1[k]; N = v['N']
        print(k, 'N', N, {m: int(round(v['summary'][m]['cost'] * N)) for m in ('BET-P', 'WSR', 'BS-P', 'HS', 'CLT-P')})
# validity
R = 30
print('incorrect runs by method:', {m: int(round(sum(v['summary'][m]['answer_err'] * R for v in ALL.values()))) for m in ['BET-P', 'BET', 'WSR', 'WSR-P', 'BS-P', 'HS', 'CLT', 'CLT-P']}, 'of', R * len(ALL))
print('queries with CLT error:', sorted(((round(v['summary']['CLT']['answer_err'], 2), k) for k, v in ALL.items() if v['summary']['CLT']['answer_err'] > 0), reverse=True))
print('queries with CLT-P error:', sorted(((round(v['summary']['CLT-P']['answer_err'], 2), k) for k, v in ALL.items() if v['summary']['CLT-P']['answer_err'] > 0), reverse=True))
# ablations
for f in ('abl_proxy.json', 'abl_margin.json', 'abl_groups.json', 'abl_alpha.json', 'abl_bad.json'):
    A = load(f)
    if not A:
        continue
    print(f'--- {f}')
    prefixes = sorted({k.split('|')[0] for k in A})
    for p in prefixes:
        S = {k: v for k, v in A.items() if k.split('|')[0] == p}
        cols = ['BET-P', 'BET', 'WSR', 'WSR-P', 'BS-P', 'HS', 'CLT-P']
        print(p, f'n={len(S)}', 'geo-mean cost', {m: round(gm([v['summary'][m]['cost'] for v in S.values()]), 4) for m in cols},
              'max err', {m: max(v['summary'][m]['answer_err'] for v in S.values()) for m in ('BET-P', 'WSR-P', 'CLT-P')})
