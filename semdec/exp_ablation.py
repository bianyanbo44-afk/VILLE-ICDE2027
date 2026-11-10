"""Ablations on AMZ-Stars (sensitivity to proxy quality, misleading proxies, margin r, alpha, number of groups).
Run: python -I exp_ablation.py <sem1_amazon.npz> <proxy_sweep.npz|-> <out.json> <R> <study>
studies: proxy, bad, margin, alpha, groups
"""
import sys, os, json, time
import hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import Query, run_query

sys.stdout.reconfigure(encoding='utf-8')
NPZ, SWEEP, OUT, R, STUDY = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
d = np.load(NPZ)
keep = ~d['train']
cat, year, asin = d['cat'][keep], d['year'][keep], d['asin'][keep]
rating = d['rating'][keep].astype(np.float64)
cats = list(d['cats']); beta = json.loads(str(d['beta']))
neg = (rating <= 2).astype(np.float64)
yrs = [y for y in np.unique(year) if (year == y).sum() >= 2000]
G_year = [np.flatnonzero(year == y) for y in yrs]
ci = int(np.argmax(np.bincount(cat)))
sizes = np.bincount(cat, minlength=len(cats))
ci = [c for c in np.argsort(-sizes) if cats[c] == 'Appliances'][0]
idx = np.flatnonzero(cat == ci)
a_, cnt = np.unique(asin[idx], return_counts=True)
top = [x for x in a_[np.argsort(-cnt)][:30] if (asin[idx] == x).sum() >= 2000]
G_prod = [idx[asin[idx] == x] for x in top]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
METHODS = ['CLT-P', 'HS', 'BS-P', 'WSR', 'WSR-P', 'BET', 'BET-P']


def run(name, idxs, Y, P, a, b, bb, kind, rmult=0.05, alpha=0.05, seed=0):
    if name in res:
        return
    Ys = [Y[i] for i in idxs]; Ps = [P[i] for i in idxs]
    mu = float(np.concatenate(Ys).mean())
    kw = dict(c=mu) if kind == 'having' else (dict(k_top=5) if kind == 'topk' else {})
    q = Query(Ys, Ps, bb, a, b, kind, rmult * abs(mu), name=name, **kw)
    t0 = time.time()
    out = run_query(q, R=R, alpha=alpha, seed=seed, methods=METHODS, return_runs=True)
    records_dir = Path(OUT).parent / (Path(OUT).stem + '_runs')
    records_dir.mkdir(exist_ok=True)
    record_name = hashlib.sha256(name.encode()).hexdigest()[:16] + '.json'
    records = {method: values.pop('runs') for method, values in out.items()}
    with (records_dir / record_name).open('w', encoding='utf8') as stream:
        json.dump(dict(query=name, seed=seed, methods=records), stream, separators=(',', ':'))
    res[name] = dict(K=len(idxs), r=rmult * abs(mu), alpha=alpha, beta=bb, summary=out,
                     seed=seed, execution_record=f'{records_dir.name}/{record_name}')
    temporary = Path(OUT).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(res, indent=1), encoding='utf8')
    temporary.replace(OUT)
    print(f'{name:44s} ' + ' '.join(f"{m}={v['cost']:.3f}/{v['err']:.2f}" for m, v in out.items()) + f' ({time.time()-t0:.0f}s)', flush=True)


P5 = d['p_neg'][keep].astype(np.float64)
if STUDY == 'proxy':
    S = np.load(SWEEP)
    for n in (100, 300, 1000):
        p = S[f'p_neg_{n}'][keep].astype(np.float64)
        bn = float(S[f'beta_neg_{n}'])   # cross-validated on the n labeled reviews, as in the main pipeline
        for kind in ('having', 'topk'):
            run(f'proxy{n}|year|NEG|{kind}', G_year, neg, p, 0.0, 1.0, bn, kind, seed=101 + n)
            run(f'proxy{n}|prod|NEG|{kind}', G_prod, neg, p, 0.0, 1.0, bn, kind, seed=201 + n)
    for kind in ('having', 'topk'):
        run(f'proxy5000|year|NEG|{kind}', G_year, neg, P5, 0.0, 1.0, beta['neg'], kind, seed=105)
        run(f'proxy5000|prod|NEG|{kind}', G_prod, neg, P5, 0.0, 1.0, beta['neg'], kind, seed=205)
elif STUDY == 'bad':
    rng = np.random.default_rng(7)
    variants = {'inverted': 1.0 - P5, 'random': rng.random(len(P5)), 'shuffled': rng.permutation(P5)}
    for vn, p in variants.items():
        for kind in ('having', 'order', 'topk'):
            run(f'bad-{vn}|year|NEG|{kind}', G_year, neg, p, 0.0, 1.0, 1.0, kind, seed=301)
elif STUDY == 'margin':
    for rm in (0.025, 0.05, 0.10):
        for kind in ('having', 'topk'):
            run(f'r{rm}|year|NEG|{kind}', G_year, neg, P5, 0.0, 1.0, beta['neg'], kind, rmult=rm, seed=401)
            run(f'r{rm}|prod|NEG|{kind}', G_prod, neg, P5, 0.0, 1.0, beta['neg'], kind, rmult=rm, seed=501)
elif STUDY == 'alpha':
    for al in (0.01, 0.05, 0.10):
        for kind in ('having', 'topk'):
            run(f'alpha{al}|year|NEG|{kind}', G_year, neg, P5, 0.0, 1.0, beta['neg'], kind, alpha=al, seed=601)
            run(f'alpha{al}|prod|NEG|{kind}', G_prod, neg, P5, 0.0, 1.0, beta['neg'], kind, alpha=al, seed=701)
elif STUDY == 'groups':
    for K in (5, 10, 20, 30):
        for kind in ('having', 'order', 'topk'):
            run(f'K{K}|prod|NEG|{kind}', G_prod[:K], neg, P5, 0.0, 1.0, beta['neg'], kind, seed=801 + K)
print('done')
