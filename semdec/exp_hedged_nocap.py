"""Robustness of the Evergreen-style baseline: hedged-capital betting CS (Waudby-Smith & Ramdas) instead of the
closed-form predictable-plug-in EB CS, on the AMZ-Stars HAVING queries (WoR).
A HAVING decision from a CS only needs the CS at c and c +- r: "above" fires when the hedged wealth at c reaches
1/a with the running mean above c; "below" symmetric; "within r" when both c-r and c+r are excluded with the
running mean between them.  a = alpha / K (two-sided CS per group).
Run: python -I exp_hedged.py <sem1_amazon.npz> <out.json> <R>
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import looks_for

sys.stdout.reconfigure(encoding='utf-8')
NPZ, OUT, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = np.load(NPZ)
keep = ~d['train']
cat, year, asin = d['cat'][keep], d['year'][keep], d['asin'][keep]
rating = d['rating'][keep].astype(np.float64)
cats = list(d['cats'])
ORACLES = {'NEG': ((rating <= 2).astype(float), 0.0, 1.0), 'POS': ((rating == 5).astype(float), 0.0, 1.0), 'RAT': (rating, 1.0, 5.0)}
G = {'cat': [np.flatnonzero(cat == i) for i in range(len(cats))],
     'year': [np.flatnonzero(year == y) for y in np.unique(year) if (year == y).sum() >= 2000]}
sizes = np.bincount(cat, minlength=len(cats))
for ci in np.argsort(-sizes)[:3]:
    idx = np.flatnonzero(cat == ci)
    a_, cnt = np.unique(asin[idx], return_counts=True)
    top = [x for x in a_[np.argsort(-cnt)][:30] if (asin[idx] == x).sum() >= 2000]
    if len(top) >= 5:
        G[f'prod-{cats[ci]}'] = [idx[asin[idx] == x] for x in top]


def hedged_first(y, N, m, a, looks, c=0.5):
    """Two-sided hedged-capital WoR test of mu = m on [0,1]-scaled data; returns first look index rejecting, else -1."""
    n = len(y)
    t = np.arange(1, n + 1, dtype=np.float64)
    S = np.cumsum(y); S_prev = np.concatenate([[0.0], S[:-1]])
    mu_hat = (0.5 + S) / (t + 1)
    sig2 = (0.25 + np.cumsum((y - mu_hat) ** 2)) / (t + 1)
    sig2_prev = np.concatenate([[0.25], sig2[:-1]])
    lam = np.sqrt(2 * math.log(2 / a) / (sig2_prev * t * np.log1p(t)))
    m_t = (N * m - S_prev) / np.maximum(N - t + 1, 1)
    up = np.clip(lam, 0, c / np.maximum(m_t, 1e-12)); dn = np.clip(lam, 0, c / np.maximum(1 - m_t, 1e-12))
    imp = (m_t < 0) | (m_t > 1)
    lp = np.cumsum(np.where(imp, 0.0, np.log1p(up * (y - m_t))))
    ln = np.cumsum(np.where(imp, 0.0, np.log1p(-dn * (y - m_t))))
    K = 0.5 * np.exp(np.minimum(lp, 700)) + 0.5 * np.exp(np.minimum(ln, 700))
    K = np.where(np.cumsum(imp) > 0, np.inf, K)
    hit = K[looks - 1] >= 1 / a
    return int(np.argmax(hit)) if hit.any() else -1


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
rng = np.random.default_rng(77)
for gname, idxs in G.items():
    for oname, (Y, lo, hi) in ORACLES.items():
        name = f'{gname}|{oname}|having'
        if name in res:
            continue
        Ys = [(Y[i] - lo) / (hi - lo) for i in idxs]
        mu_all = float(np.concatenate(Ys).mean()); r = 0.05 * abs(np.concatenate([Y[i] for i in idxs]).mean()) / (hi - lo)
        K = len(Ys); a = 0.05 / K
        N = np.array([len(y) for y in Ys]); L = N.copy(); mus = np.array([y.mean() for y in Ys])
        costs, errs = [], []
        t0 = time.time()
        for run in range(R):
            tot, err = 0.0, False
            for g in range(K):
                y = Ys[g][rng.permutation(N[g])[:L[g]]]
                lk = looks_for(L[g])
                run_mean = (np.cumsum(y) / np.arange(1, len(y) + 1))[lk - 1]
                tc = hedged_first(y, N[g], mu_all, a, lk)
                tp, tm = hedged_first(y, N[g], mu_all + r, a, lk), hedged_first(y, N[g], mu_all - r, a, lk)
                cands = []
                if tc >= 0:
                    cands.append((tc, 'above' if run_mean[tc] > mu_all else 'below'))
                if tp >= 0 and tm >= 0:
                    ti = max(tp, tm)
                    if mu_all - r < run_mean[ti] < mu_all + r:
                        cands.append((ti, 'ind'))
                if not cands or (lk[min(cands)[0]] >= L[g] and L[g] == N[g]):
                    tot += N[g]; continue
                t_, dec = min(cands); tot += lk[t_]
                mu = mus[g]
                if abs(mu - mu_all) >= r and ((dec == 'ind') or (dec == 'above' and mu < mu_all) or (dec == 'below' and mu > mu_all)):
                    err = True
            costs.append(tot / N.sum()); errs.append(err)
        res[name] = dict(cost=float(np.mean(costs)), err=float(np.mean(errs)))
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'{name:40s} hedged-CS cost={np.mean(costs):.4f} err={np.mean(errs):.2f} ({time.time()-t0:.0f}s)', flush=True)
print('done')
