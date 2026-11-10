"""Main experiment on D1 (Amazon-Stars, 6.8M reviews; star-derived oracles) with WoR sampling.
Run: python -I exp_d1.py <sem1_amazon.npz> <out.json> <R> grouping1 [grouping2 ...]
groupings: cat, year, prod-Amazon_Fashion, prod-Appliances
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import Query, run_query

sys.stdout.reconfigure(encoding='utf-8')
NPZ, OUT, R, WANT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
CAP = 10 ** 12          # sample until certified or exhausted
if '--nocap' in WANT:                 # sample until certified or exhausted (no per-group fallback cap)
    WANT = [w for w in WANT if w != '--nocap']; CAP = 10 ** 12
d = np.load(NPZ)
keep = ~d['train']
cat, year, asin = d['cat'][keep], d['year'][keep], d['asin'][keep]
rating = d['rating'][keep].astype(np.float64)
cats = list(d['cats']); beta = json.loads(str(d['beta']))
ORACLES = {'NEG': ((rating <= 2).astype(np.float64), d['p_neg'][keep].astype(np.float64), 0.0, 1.0, 'neg'),
           'POS': ((rating == 5).astype(np.float64), d['p_pos'][keep].astype(np.float64), 0.0, 1.0, 'pos'),
           'RAT': (rating, d['p_rat'][keep].astype(np.float64), 1.0, 5.0, 'rat')}
groupings = {}
groupings['cat'] = [np.flatnonzero(cat == i) for i in range(len(cats))]
yrs = [y for y in np.unique(year) if (year == y).sum() >= 2000]
groupings['year'] = [np.flatnonzero(year == y) for y in yrs]
sizes = np.bincount(cat, minlength=len(cats))
for ci in np.argsort(-sizes)[:3]:
    idx = np.flatnonzero(cat == ci)
    a, cnt = np.unique(asin[idx], return_counts=True)
    top = [x for x in a[np.argsort(-cnt)][:30] if (asin[idx] == x).sum() >= 2000]
    if len(top) >= 5:
        groupings[f'prod-{cats[ci]}'] = [idx[asin[idx] == x] for x in top]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
qi = 0
for gname in WANT:
    idxs = groupings[gname]
    K = len(idxs)
    for oname, (Y, P, a, b, bk) in ORACLES.items():
        Ys = [Y[i] for i in idxs]; Ps = [P[i] for i in idxs]
        mu_all = float(np.concatenate(Ys).mean()); r = 0.05 * abs(mu_all)
        for kind in ('having', 'order', 'topk'):
            name = f'{gname}|{oname}|{kind}'
            qi += 1
            if name in res:
                continue
            kw = dict(c=mu_all) if kind == 'having' else (dict(k_top=3 if K < 10 else 5) if kind == 'topk' else {})
            q = Query(Ys, Ps, beta[bk], a, b, kind, r, name=name, **kw)
            t0 = time.time()
            out = run_query(q, R=R, seed=9000 + 17 * qi, cap=CAP)
            res[name] = dict(K=K, N=int(q.N.sum()), r=r, c=kw.get('c'), k=kw.get('k_top'), beta=beta[bk], summary=out)
            json.dump(res, open(OUT, 'w'), indent=1)
            print(f'{name:38s} K={K:2d} ' + ' '.join(f"{m}={v['cost']:.3f}/{v['err']:.2f}" for m, v in out.items())
                  + f' ({time.time()-t0:.0f}s)', flush=True)
print('done')
