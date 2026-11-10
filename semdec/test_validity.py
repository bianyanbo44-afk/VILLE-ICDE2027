"""Type-I error check of the WoR betting tests on finite populations sitting exactly on the null boundary."""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import wor_bet_first, wor_bet_upper, wor_pair_first, looks_for

rng = np.random.default_rng(1)
a = 0.05; R = 2000
# one-sample: Bernoulli population with mean exactly m, and a skewed [0,1] population
for name, pop in (('bern0.15', (np.arange(20000) < 3000).astype(float)),
                  ('beta', np.clip(rng.beta(0.3, 2.0, 5000), 0, 1))):
    N = len(pop); m = pop.mean(); L = N; lk = looks_for(L)
    rej = sum(wor_bet_first(pop[rng.permutation(N)], N, m, 0.0, 1.0, a, lk) >= 0 for _ in range(R))
    rej2 = sum(wor_bet_upper(pop[rng.permutation(N)], N, m, 0.0, 1.0, a, lk) >= 0 for _ in range(R))
    print(f'one-sample {name}: N={N} reject(H0: mu<=m)={rej/R:.4f}  reject(H0: mu>=m)={rej2/R:.4f}  (should be <= {a})')
# paired: two groups of different sizes with mu_i - mu_j = d0 exactly
for d0 in (0.0, 0.05):
    pj = (np.arange(8000) < 1600).astype(float)            # mean 0.20
    pi = (np.arange(3000) < int(3000 * (0.20 + d0))).astype(float)
    di = pi.mean() - pj.mean()
    n = min(len(pi), len(pj)); lk = looks_for(n)
    rej = 0
    for _ in range(R):
        zi = pi[rng.permutation(len(pi))][:n]; zj = pj[rng.permutation(len(pj))][:n]
        rej += wor_pair_first(zi, zj, len(pi), len(pj), di, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), a, lk) >= 0
    print(f'paired d0={di:.3f}: reject(H0: mu_i-mu_j<=d0)={rej/R:.4f} (should be <= {a})')
