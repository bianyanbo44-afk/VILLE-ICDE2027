"""How often would answering from the proxy alone be wrong?  For every main-experiment query, replace each
group mean by the group's mean proxy score and check r-correctness against the true answer (same groups,
thresholds, margins, and k as the main experiments).
Run: python -I proxy_only.py <sem1_amazon.npz> <data/sem2> <data/sem3> <out.json>
"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
NPZ, D2, D3, OUT = sys.argv[1:5]


def check(Ys, Ps, kind, k):
    mu = np.array([y.mean() for y in Ys]); pb = np.array([p.mean() for p in Ps])
    c = float(np.concatenate(Ys).mean()); r = 0.05 * abs(c)
    if kind == 'having':
        bad = [(abs(m - c) >= r) and ((p > c) != (m > c)) for m, p in zip(mu, pb)]
        return bool(any(bad)), int(sum(bad))
    if kind == 'order':
        K = len(mu); bad = 0
        for i in range(K):
            for j in range(K):
                if mu[i] >= mu[j] + r and pb[i] < pb[j]:
                    bad += 1
        return bad > 0, bad
    S = np.argsort(-pb)[:k]; kth = np.sort(mu)[::-1][k - 1]
    bad = int(sum(mu[g] < kth - r for g in S))
    return bad > 0, bad


out = {}
# AMZ-Stars (star-derived judgments; groupings as in exp_d1.py)
d = np.load(NPZ); keep = ~d['train']
cat, year, asin = d['cat'][keep], d['year'][keep], d['asin'][keep]
rating = d['rating'][keep].astype(np.float64); cats = list(d['cats'])
OR = {'NEG': ((rating <= 2) * 1.0, d['p_neg'][keep].astype(np.float64)), 'POS': ((rating == 5) * 1.0, d['p_pos'][keep].astype(np.float64)),
      'RAT': (rating, d['p_rat'][keep].astype(np.float64))}
G = {'cat': [np.flatnonzero(cat == i) for i in range(len(cats))],
     'year': [np.flatnonzero(year == y) for y in np.unique(year) if (year == y).sum() >= 2000]}
sizes = np.bincount(cat, minlength=len(cats))
for ci in np.argsort(-sizes)[:3]:
    idx = np.flatnonzero(cat == ci); a, cnt = np.unique(asin[idx], return_counts=True)
    top = [x for x in a[np.argsort(-cnt)][:30] if (asin[idx] == x).sum() >= 2000]
    if len(top) >= 5:
        G[f'prod-{cats[ci]}'] = [idx[asin[idx] == x] for x in top]
for gn, idxs in G.items():
    for on, (Y, P) in OR.items():
        for kind in ('having', 'order', 'topk'):
            k = 3 if len(idxs) == 8 else 5
            out[f'S:{gn}|{on}|{kind}'] = check([Y[i] for i in idxs], [P[i] for i in idxs], kind, k)
# AMZ-LLM and ArXiv-LLM (real LLM judgments; groupings as in sem2_run.py / sem3_run.py)
for tag, D, preds in (('L', D2, (('DEFECT', 'p_defect'), ('VALUE', 'p_value'), ('SENT', 'p_sent'))),
                      ('A', D3, (('SOTA', 'p_sota'), ('EMP', 'p_emp')))):
    if not os.path.exists(os.path.join(D, 'proxies.npz')):
        continue
    meta = np.load(os.path.join(D, 'meta.npz')); lab = np.load(os.path.join(D, 'llm_labels.npz'))
    prox = np.load(os.path.join(D, 'proxies.npz')); q_idx = np.flatnonzero(~meta['train'])
    if tag == 'L':
        G = {}
        for ci, cn in enumerate(meta['cats']):
            idx = q_idx[meta['cat'][q_idx] == ci]; a, cnt = np.unique(meta['asin'][idx], return_counts=True)
            G[f'prod-{cn}'] = [idx[meta['asin'][idx] == x] for x in a[np.argsort(-cnt)][:30]]
        yr = meta['year']
        G['year'] = [q_idx[yr[q_idx] == y] for y in np.unique(yr[q_idx]) if (yr[q_idx] == y).sum() >= 1000]
    else:
        yr, ca = meta['year'], meta['cat']
        G = {'year': [q_idx[yr[q_idx] == y] for y in np.unique(yr[q_idx])]}
        cnt = np.bincount(ca[q_idx], minlength=len(meta['cats']))
        G['area'] = [q_idx[ca[q_idx] == c] for c in np.argsort(-cnt)[:16]]
    for on, key in preds:
        if key not in lab.files or key.replace('p_', 'p_') not in prox.files:
            continue
        Y = (lab[key].argmax(1) + 1.0) if on == 'SENT' else (lab[key][:, 0] > 0.5) * 1.0
        P = prox[key]
        for gn, idxs in G.items():
            for kind in ('having', 'order', 'topk'):
                out[f'{tag}:{gn}|{on}|{kind}'] = check([Y[i] for i in idxs], [P[i] for i in idxs], kind, 5)
json.dump(out, open(OUT, 'w'), indent=1)
for tag in ('S', 'L', 'A'):
    v = [x for k, x in out.items() if k.startswith(tag + ':')]
    if v:
        print(tag, 'queries', len(v), 'proxy-only answer wrong on', sum(x[0] for x in v),
              {kind: f"{sum(x[0] for k, x in out.items() if k.startswith(tag + ':') and k.endswith(kind))}/{sum(1 for k in out if k.startswith(tag + ':') and k.endswith(kind))}" for kind in ('having', 'order', 'topk')})
