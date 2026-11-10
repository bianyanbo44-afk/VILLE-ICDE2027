"""Gate evaluation for SEM2 (and descriptive summaries for any results JSON from exp_d1 / sem2_run).
Run: python -I gates.py <results.json> [--sem2]
"""
import sys, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
res = json.load(open(sys.argv[1]))
SEM2 = '--sem2' in sys.argv
rows = []
for name, v in res.items():
    if name.startswith('_'):
        continue
    s = v['summary']
    rows.append(dict(name=name, kind=name.split('|')[-1], **{m: s[m]['cost'] for m in s}, **{m + '_err': s[m]['err'] for m in s}))
eps = 1e-9
med = lambda x: float(np.median(x)) if len(x) else float('nan')
g1 = [r['BET-P'] / r['BET'] for r in rows if 0.01 < r['BET'] < 0.99]
g2a = float(np.mean([r['BET-P'] <= min(r['HS'], r['BS-P']) + eps for r in rows]))
g2b = [r['BET-P'] / min(r['HS'], r['BS-P']) for r in rows if 0.01 < min(r['HS'], r['BS-P']) < 0.99]
g3 = max(max(r['BET-P_err'], r['WSR-P_err']) for r in rows)
g4 = [r['BET-P'] / r['CLT-P'] for r in rows if r['CLT-P_err'] <= 0.10 and r['CLT-P'] > 0]
g5 = [r['BET-P'] / r['WSR'] for r in rows if 0.01 < r['WSR'] < 0.99]
ok = dict(G1=len(g1) > 0 and med(g1) <= 0.6, G2=g2a >= 0.9 and len(g2b) > 0 and med(g2b) <= 0.5, G3=g3 <= 0.094,
          G4=len(g4) > 0 and med(g4) <= 3, G5=len(g5) > 0 and med(g5) <= 0.6)
print(f'queries {len(rows)}')
print(f'G1 median BET-P/BET = {med(g1):.3f} (n={len(g1)}, need <=0.6) -> {ok["G1"]}')
print(f'G2 BET-P<=min(HS,BS-P): {g2a:.3f} (need >=0.9); median BET-P/min(HS,BS-P) = {med(g2b):.3f} (n={len(g2b)}, need <=0.5) -> {ok["G2"]}')
print(f'G3 max err BET-P/WSR-P = {g3:.3f} (need <=0.094) -> {ok["G3"]}')
print(f'G4 median BET-P/CLT-P = {med(g4):.3f} (n={len(g4)}, need <=3) -> {ok["G4"]}')
print(f'G5 median BET-P/WSR = {med(g5):.3f} (n={len(g5)}, need <=0.6) -> {ok["G5"]}')
if SEM2:
    print('ALL PASS' if all(ok.values()) else 'NOT PASSED: ' + ','.join(k for k, v in ok.items() if not v))
# descriptive by query kind
for kind in ('having', 'order', 'topk'):
    rk = [r for r in rows if r['kind'] == kind]
    if not rk:
        continue
    def gm(a, b):
        x = [r[a] / r[b] for r in rk if r[b] > 0 and r[a] > 0]
        return float(np.exp(np.mean(np.log(x)))) if x else float('nan')
    print(f'[{kind}] n={len(rk)} geo-mean ratio BET-P/WSR={gm("BET-P","WSR"):.3f} BET-P/WSR-P={gm("BET-P","WSR-P"):.3f} '
          f'BET-P/BS-P={gm("BET-P","BS-P"):.3f} BET-P/HS={gm("BET-P","HS"):.3f} BET-P/CLT-P={gm("BET-P","CLT-P"):.3f} '
          f'| CLT err>0.1: {sum(r["CLT_err"] > 0.1 for r in rk)} CLT-P err>0.1: {sum(r["CLT-P_err"] > 0.1 for r in rk)} '
          f'max CLT err {max(r["CLT_err"] for r in rk):.2f}/{max(r["CLT-P_err"] for r in rk):.2f}')
