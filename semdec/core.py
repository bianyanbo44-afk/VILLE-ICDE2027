"""Certified decision queries over expensive per-tuple evaluations (LLM predicates / scores).

All sampling is without replacement (WoR) inside each group.  A group g has N_g tuples with
oracle values y in [a, b] (one LLM call each) and proxy values p in [a, b] (cheap, known for all tuples).
Proxy-rectified values  z = y - beta * (p - mean_g(p))  have the same group mean as y.

Decision tests are e-processes (nonnegative supermartingales under H0); by Ville's inequality a test that
rejects when its wealth reaches 1/a has type-I error <= a at any stopping time.
"""
import math
import numpy as np
from scipy.stats import norm

BATCH = 100


def looks_for(L):
    lk = np.arange(BATCH, L + 1, BATCH)
    if len(lk) == 0 or lk[-1] != L:
        lk = np.append(lk, L)
    return lk


# ----------------------------------------------------------------------------------------------
# one-sample WoR betting test of H0: mu <= m   (values in [lo, hi], population size N)
# ----------------------------------------------------------------------------------------------
def wor_bet_first(x, N, m, lo, hi, a, looks, cap=0.5):
    """Index into `looks` of the first look whose wealth >= 1/a; -1 if none.  x: WoR draws (len L <= N)."""
    if m < lo:
        return 0
    if m >= hi:
        return -1
    L = len(x)
    t = np.arange(1, L + 1, dtype=np.float64)
    S = np.cumsum(x)
    S_prev = np.concatenate([[0.0], S[:-1]])
    rem = N - t + 1.0
    m_t = (N * m - S_prev) / rem                       # conditional null mean of draw t
    mu_hat = (m + S_prev) / t                          # running population-mean estimate (pseudo-obs at m)
    pred = (N * mu_hat - S_prev) / rem                 # predicted conditional mean under the alternative
    dev = (x - (m + S) / (t + 1)) ** 2
    s0 = ((hi - lo) / 4) ** 2
    sig2_prev = np.concatenate([[s0], ((s0 + np.cumsum(dev)) / (t + 1))[:-1]])
    gap = pred - m_t
    room = np.maximum(m_t - lo, 1e-12)
    lam = np.clip(gap / (sig2_prev + gap ** 2), 0.0, cap / room)
    lam = np.where(m_t >= hi, 0.0, lam)
    inc = np.log1p(np.maximum(lam * (x - m_t), -0.999999))
    inc = np.where(m_t < lo - 1e-12, np.inf, inc)      # H0 impossible given the data so far
    logK = np.cumsum(inc)
    hit = logK[looks - 1] >= math.log(1.0 / a)
    return int(np.argmax(hit)) if hit.any() else -1


def wor_bet_upper(x, N, m, lo, hi, a, looks, cap=0.5):
    """First look rejecting H0: mu >= m."""
    return wor_bet_first(-x, N, -m, -hi, -lo, a, looks, cap)


# ----------------------------------------------------------------------------------------------
# paired WoR betting test of H0: mu_i - mu_j <= d0
# ----------------------------------------------------------------------------------------------
def _wor_unbiased(z, N, n):
    """Per-draw statistics u_t = (z_t + B_t)/A_t with E[u_t | past] = group mean (WoR, any t <= N).
    After the group is exhausted (t > N) u_t is the exact group mean.  Returns (u, umin_coef, B/A, 1/A)."""
    L = len(z)
    t = np.arange(1, n + 1, dtype=np.float64)
    m = min(L, n)
    S = np.cumsum(z[:m]); S_prev = np.concatenate([[0.0], S[:-1]])
    tt = t[:m]
    inv_A = (N - tt + 1.0) / N
    BoA = S_prev / N
    u = np.empty(n); u[:m] = z[:m] * inv_A + BoA
    ia = np.zeros(n); ia[:m] = inv_A
    ba = np.empty(n); ba[:m] = BoA
    if n > m:                                   # exhausted group: constant exact mean (only if m == N)
        u[m:] = S[-1] / N; ba[m:] = S[-1] / N
    return u, ia, ba


def wor_pair_first(zi, zj, Ni, Nj, d0, rng_i, rng_j, mu_rng, a, looks, cap=0.5, n=None):
    """Lockstep WoR draws from both groups; X_t = u^i_t - u^j_t has conditional mean mu_i - mu_j exactly.
    A group whose stream covers all N tuples contributes its exact mean after exhaustion."""
    if n is None:
        n = min(len(zi), len(zj))
    ui, iai, bai = _wor_unbiased(zi, Ni, n)
    uj, iaj, baj = _wor_unbiased(zj, Nj, n)
    X = ui - uj
    xmin = (rng_i[0] * iai + bai) - (rng_j[1] * iaj + baj)       # predictable lower bound of X_t
    xmax = (rng_i[1] * iai + bai) - (rng_j[0] * iaj + baj)
    t = np.arange(1, n + 1, dtype=np.float64)
    SX = np.cumsum(X); SX_p = np.concatenate([[0.0], SX[:-1]])
    mu_hat = (d0 + SX_p) / t                                      # one pseudo-observation at the null
    dev = (X - (d0 + SX) / (t + 1)) ** 2
    s0 = ((mu_rng[1] - mu_rng[0]) / 2) ** 2
    sig2_prev = np.concatenate([[s0], ((s0 + np.cumsum(dev)) / (t + 1))[:-1]])
    gap = mu_hat - d0
    room = np.maximum(d0 - xmin, 1e-12)
    lam = np.clip(gap / (sig2_prev + gap ** 2), 0.0, cap / room)
    lam = np.where(xmax <= d0, 0.0, lam)
    inc = np.log1p(np.maximum(lam * (X - d0), -0.999999))
    inc = np.where(xmin > d0 + 1e-12, np.inf, inc)               # H0 impossible given the data so far
    logK = np.cumsum(inc)
    hit = logK[looks - 1] >= math.log(1.0 / a)
    return int(np.argmax(hit)) if hit.any() else -1


def wor_pair_upper(zi, zj, Ni, Nj, d0, rng_i, rng_j, mu_rng, a, looks, cap=0.5, n=None):
    """First look rejecting H0: mu_i - mu_j >= d0."""
    return wor_pair_first(zj, zi, Nj, Ni, -d0, rng_j, rng_i, mu_rng, a, looks, cap, n)


def pair_first_from(X, xmin, xmax, d0, mu_rng, a, looks, cap=0.5):
    """Same test as wor_pair_first, from precomputed X_t = u^i_t - u^j_t and its predictable bounds."""
    n = len(X)
    t = np.arange(1, n + 1, dtype=np.float64)
    SX = np.cumsum(X); SX_p = np.concatenate([[0.0], SX[:-1]])
    mu_hat = (d0 + SX_p) / t
    dev = (X - (d0 + SX) / (t + 1)) ** 2
    s0 = ((mu_rng[1] - mu_rng[0]) / 2) ** 2
    sig2_prev = np.concatenate([[s0], ((s0 + np.cumsum(dev)) / (t + 1))[:-1]])
    gap = mu_hat - d0
    room = np.maximum(d0 - xmin, 1e-12)
    lam = np.clip(gap / (sig2_prev + gap ** 2), 0.0, cap / room)
    lam = np.where(xmax <= d0, 0.0, lam)
    inc = np.log1p(np.maximum(lam * (X - d0), -0.999999))
    inc = np.where(xmin > d0 + 1e-12, np.inf, inc)
    logK = np.cumsum(inc)
    hit = logK[looks - 1] >= math.log(1.0 / a)
    return int(np.argmax(hit)) if hit.any() else -1


# ----------------------------------------------------------------------------------------------
# interval baselines on the group-mean scale (all evaluated at `looks`)
# ----------------------------------------------------------------------------------------------
def iv_clt(x, N, k_mult, alpha, looks):
    t = np.arange(1, len(x) + 1, dtype=np.float64)
    S, S2 = np.cumsum(x), np.cumsum(x * x)
    mean = S / t
    var = np.maximum(S2 / t - mean ** 2, 0) * t / np.maximum(t - 1, 1)
    fpc = np.maximum(1 - (t - 1) / max(N - 1, 1), 0)
    hw = norm.ppf(1 - alpha / (2 * k_mult)) * np.sqrt(var / t * fpc)
    hw = np.where(t >= N, 0.0, hw)
    return (mean - hw)[looks - 1], (mean + hw)[looks - 1]


def iv_hs(x, N, lo, hi, k_mult, alpha, looks):
    """IFOCUS-style: Hoeffding-Serfling with a union bound over looks."""
    n = looks.astype(np.float64)
    mean = (np.cumsum(x) / np.arange(1, len(x) + 1))[looks - 1]
    mm = np.arange(1, len(looks) + 1)
    am = 6 * alpha / (math.pi ** 2 * k_mult * mm ** 2)
    rho = np.maximum(1 - (n - 1) / N, 0)
    eps = (hi - lo) * np.sqrt(rho * np.log(2 / am) / (2 * n))
    eps = np.where(n >= N, 0.0, eps)
    return mean - eps, mean + eps


def iv_bs(x, N, lo, hi, k_mult, alpha, looks):
    """BlazeIt/Macke-style empirical Bernstein stopping: Maurer-Pontil bound at doubling rounds, union bound."""
    t = np.arange(1, len(x) + 1, dtype=np.float64)
    S, S2 = np.cumsum(x), np.cumsum(x * x)
    mean = S / t
    var = np.maximum(S2 / t - mean ** 2, 0) * t / np.maximum(t - 1, 1)
    Lb = np.full(len(looks), -np.inf); Ub = np.full(len(looks), np.inf)
    j = 0; cur = (-np.inf, np.inf)
    for li, n in enumerate(looks):
        if n >= N:
            cur = (mean[n - 1], mean[n - 1])
        elif n >= BATCH * 2 ** j:
            d = 6 * alpha / (math.pi ** 2 * k_mult * (j + 1) ** 2)
            e = math.sqrt(2 * var[n - 1] * math.log(4 / d) / n) + 7 * (hi - lo) * math.log(4 / d) / (3 * max(n - 1, 1))
            cur = (mean[n - 1] - e, mean[n - 1] + e); j += 1
        Lb[li], Ub[li] = cur
    return Lb, Ub


def iv_wsr(x, N, lo, hi, k_mult, alpha, looks, c=0.5):
    """Evergreen-style: Waudby-Smith & Ramdas WoR predictable-plug-in empirical-Bernstein CS (two-sided)."""
    a2 = alpha / k_mult
    y = (x - lo) / (hi - lo)
    n = len(y)
    t = np.arange(1, n + 1, dtype=np.float64)
    S = np.cumsum(y)
    mu_hat = (0.5 + S) / (t + 1)
    mu_prev = np.concatenate([[0.5], mu_hat[:-1]])
    sig2 = (0.25 + np.cumsum((y - mu_hat) ** 2)) / (t + 1)
    sig2_prev = np.concatenate([[0.25], sig2[:-1]])
    lam = np.minimum(np.sqrt(2 * math.log(2 / a2) / (sig2_prev * t * np.log1p(t))), c)
    v = 4 * (y - mu_prev) ** 2
    psi = (-np.log1p(-lam) - lam) / 4
    S_prev = np.concatenate([[0.0], S[:-1]])
    rem = np.maximum(N - t + 1, 1.0)
    Zw = y + S_prev / rem
    w = N / rem
    den = np.cumsum(lam * w)
    center = np.cumsum(lam * Zw) / den
    margin = (math.log(2 / a2) + np.cumsum(v * psi)) / den
    l = np.maximum.accumulate(np.clip(center - margin, 0, 1))
    u = np.minimum.accumulate(np.clip(center + margin, 0, 1))
    Lb, Ub = lo + l * (hi - lo), lo + u * (hi - lo)
    ex = t >= N
    Lb = np.where(ex, S / t * (hi - lo) + lo, Lb); Ub = np.where(ex, S / t * (hi - lo) + lo, Ub)
    return Lb[looks - 1], Ub[looks - 1]


def first_true(mask):
    return int(np.argmax(mask)) if mask.any() else -1
