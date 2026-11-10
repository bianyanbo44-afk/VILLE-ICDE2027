"""SEM2: confirmatory run on real LLM judgments (see probes/SEM2_PROTOCOL.md).
Run: python -I sem2_run.py <data/sem2 dir> <out.json> [R]
"""
import sys, os, json, time
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import Query, run_query

sys.stdout.reconfigure(encoding='utf-8')
D, OUT = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 30
meta = np.load(os.path.join(D, 'meta.npz'))
lab = np.load(os.path.join(D, 'llm_labels.npz'))
assert np.array_equal(lab['ids'], np.arange(len(meta['cat'])))
texts = [json.loads(l)['text'] for l in open(os.path.join(D, 'texts.jsonl'), encoding='utf-8')]
train = meta['train']
Y = {'DEFECT': (lab['p_defect'][:, 0] > 0.5).astype(np.float64),
     'VALUE': (lab['p_value'][:, 0] > 0.5).astype(np.float64),
     'SENT': (lab['p_sent'].argmax(1) + 1).astype(np.float64)}
RANGE = {'DEFECT': (0.0, 1.0), 'VALUE': (0.0, 1.0), 'SENT': (1.0, 5.0)}

# ---- proxies trained on the 3,000 held-out LLM-labeled reviews
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm=None)
tf = TfidfTransformer(sublinear_tf=True)
tr_idx = np.flatnonzero(train); q_idx = np.flatnonzero(~train)
Xtr = tf.fit_transform(hv.transform([texts[i] for i in tr_idx]))
Xq = tf.transform(hv.transform([texts[i] for i in q_idx]))
PROXY, BETA, INFO = {}, {}, {}
for name in Y:
    y = Y[name][tr_idx]
    if name == 'SENT':
        m = Ridge(alpha=1.0); pcv = np.clip(cross_val_predict(m, Xtr, y, cv=5), 1, 5)
        m.fit(Xtr, y); pq = np.clip(m.predict(Xq), 1, 5)
    else:
        m = LogisticRegression(C=4.0, max_iter=2000); pcv = cross_val_predict(m, Xtr, y, cv=5, method='predict_proba')[:, 1]
        m.fit(Xtr, y); pq = m.predict_proba(Xq)[:, 1]
    BETA[name] = float(np.clip(np.cov(y, pcv)[0, 1] / max(np.var(pcv, ddof=1), 1e-12), 0, 1))
    full = np.zeros(len(texts)); full[q_idx] = pq; PROXY[name] = full
    yq = Y[name][q_idx]
    INFO[name] = dict(rate_or_mean=float(yq.mean()), beta=BETA[name], corr=float(np.corrcoef(yq, pq)[0, 1]),
                      auc=(float(roc_auc_score(yq, pq)) if name != 'SENT' else None))
    print(name, INFO[name], flush=True)

cat, asin, year = meta['cat'], meta['asin'], meta['year']
cats = list(meta['cats'])
groupings = {}
for ci, cn in enumerate(cats):
    idx = q_idx[cat[q_idx] == ci]
    a, cnt = np.unique(asin[idx], return_counts=True)
    top = a[np.argsort(-cnt)][:30]
    groupings[f'prod-{cn}'] = [idx[asin[idx] == x] for x in top]
yrs = [y for y in np.unique(year[q_idx]) if (year[q_idx] == y).sum() >= 1000]
groupings['year'] = [q_idx[year[q_idx] == y] for y in yrs]
print('groupings', {k: (len(v), int(sum(len(i) for i in v))) for k, v in groupings.items()}, flush=True)
np.savez(os.path.join(D, 'proxies.npz'), **{f'p_{k.lower()}': v for k, v in PROXY.items()},
         **{f'beta_{k.lower()}': np.array(v) for k, v in BETA.items()})

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res['_info'] = INFO
qi = 0
for gname, idxs in groupings.items():
    K = len(idxs)
    for oname in Y:
        a, b = RANGE[oname]
        Ys = [Y[oname][i] for i in idxs]; Ps = [PROXY[oname][i] for i in idxs]
        mu_all = float(np.concatenate(Ys).mean()); r = 0.05 * abs(mu_all)
        for kind in ('having', 'order', 'topk'):
            name = f'{gname}|{oname}|{kind}'
            qi += 1
            if name in res:
                continue
            kw = dict(c=mu_all) if kind == 'having' else (dict(k_top=5) if kind == 'topk' else {})
            q = Query(Ys, Ps, BETA[oname], a, b, kind, r, name=name, **kw)
            t0 = time.time()
            out = run_query(q, R=R, seed=12000 + 31 * qi)
            res[name] = dict(K=K, N=int(q.N.sum()), r=r, c=kw.get('c'), k=kw.get('k_top'), beta=BETA[oname], summary=out)
            json.dump(res, open(OUT, 'w'), indent=1)
            print(f'{name:38s} K={K:2d} ' + ' '.join(f"{m}={v['cost']:.3f}/{v['err']:.2f}" for m, v in out.items())
                  + f' ({time.time()-t0:.0f}s)', flush=True)
print('done')
