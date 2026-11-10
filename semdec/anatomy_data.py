"""Data for the 'anatomy of a decision' figure: one HAVING decision on AMZ-Stars (year x NEG), one run.
Run: python -I anatomy_data.py <sem1_amazon.npz> <out.json>
"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import looks_for, iv_wsr

NPZ, OUT = sys.argv[1], sys.argv[2]
d = np.load(NPZ); keep = ~d['train']
year = d['year'][keep]; rating = d['rating'][keep].astype(float)
y_all = (rating <= 2).astype(float); p_all = d['p_neg'][keep].astype(float)
beta = json.loads(str(d['beta']))['neg']
yrs = [y for y in np.unique(year) if (year == y).sum() >= 2000]
idxs = [np.flatnonzero(year == y) for y in yrs]
c = float(y_all[np.concatenate(idxs)].mean()); r = 0.05 * c; K = len(idxs); alpha = 0.05; a = alpha / (2 * K)
mus = np.array([y_all[i].mean() for i in idxs]); Ns = np.array([len(i) for i in idxs])
# pick a large group whose mean lies 1.5r to 3r above the threshold (a decision that needs real evidence)
cand = [g for g in range(K) if 1.5 * r <= mus[g] - c <= 3 * r and Ns[g] > 100_000]
g = [i for i, v in enumerate(yrs) if v == int(sys.argv[3])][0] if len(sys.argv) > 3 else (max(cand, key=lambda g: Ns[g]) if cand else int(np.argmin(np.abs(mus - c - 2 * r))))
rng = np.random.default_rng(3)
L = min(200_000, int(Ns[g]))
sgn = 1.0 if mus[g] > c else -1.0
idx = idxs[g][rng.permutation(Ns[g])[:L]]
y, p = y_all[idx], p_all[idx]; N = int(Ns[g]); pbar = float(p_all[idxs[g]].mean())


def logwealth(x, m, lo, hi, cap=0.5):
    t = np.arange(1, len(x) + 1, dtype=float)
    S = np.cumsum(x); S_prev = np.concatenate([[0.0], S[:-1]])
    rem = N - t + 1.0
    m_t = (N * m - S_prev) / rem
    mu_hat = (m + S_prev) / t
    pred = (N * mu_hat - S_prev) / rem
    dev = (x - (m + S) / (t + 1)) ** 2
    s0 = ((hi - lo) / 4) ** 2
    sig2_prev = np.concatenate([[s0], ((s0 + np.cumsum(dev)) / (t + 1))[:-1]])
    gap = pred - m_t
    lam = np.clip(gap / (sig2_prev + gap ** 2), 0.0, cap / np.maximum(m_t - lo, 1e-12))
    return np.cumsum(np.log1p(np.maximum(lam * (x - m_t), -0.999999)))


z = y - beta * (p - pbar)
zl, zh = 0 - beta * (1 - pbar), 1 - beta * (0 - pbar)
lw_p = logwealth(sgn * z, sgn * c, min(sgn * zl, sgn * zh), max(sgn * zl, sgn * zh))
lw_y = logwealth(sgn * y, sgn * c, min(0.0, sgn * 1.0), max(0.0, sgn * 1.0))
lk = looks_for(L)
Lb, Ub = iv_wsr(y, N, 0.0, 1.0, K, alpha, lk)
first = lambda arr: int(lk[np.argmax(arr)]) if arr.any() else None
out = dict(group_year=int(yrs[g]), N=N, mu=float(mus[g]), c=c, r=r, K=K, a=a, threshold=math.log(1 / a),
           t=lk.tolist(), logwealth_ville=lw_p[lk - 1].tolist(), logwealth_ville_noproxy=lw_y[lk - 1].tolist(),
           cs_lower=Lb.tolist(), cs_upper=Ub.tolist(),
           fire_ville=first(lw_p[lk - 1] >= math.log(1 / a)), fire_ville_noproxy=first(lw_y[lk - 1] >= math.log(1 / a)),
           fire_cs=first(Lb > c) if sgn > 0 else first(Ub < c), direction='above' if sgn > 0 else 'below')
json.dump(out, open(OUT, 'w'))
print({k: v for k, v in out.items() if not isinstance(v, list)})
