"""Simulation runner for certified decision queries (HAVING / ORDER BY / TOP-k) with WoR sampling.

Cost = number of oracle (LLM) calls, reported as a fraction of all tuples in the query's groups.
Methods
  CLT, CLT-P   : CLT intervals (with FPC) checked at every look -- NOT anytime-valid (uncertified)
  HS           : IFOCUS-style Hoeffding-Serfling + union bound over looks, oracle values only
  BS-P         : BlazeIt-style control variates (proxy) + empirical-Bernstein stopping
  WSR, WSR-P   : Evergreen-style Waudby-Smith--Ramdas WoR confidence sequence, without / with proxy
  BET, BET-P   : boundary betting e-processes (ours), without / with proxy
"""
import math
import numpy as np
from query_schedule import DecisionState, error_flags
from core import (looks_for, wor_bet_first, wor_bet_upper, wor_pair_first, wor_pair_upper, pair_first_from, _wor_unbiased,
                  iv_clt, iv_hs, iv_bs, iv_wsr, first_true, BATCH)

ALL_METHODS = ['CLT', 'CLT-P', 'HS', 'BS-P', 'WSR', 'WSR-P', 'BET', 'BET-P']
IV = {'CLT': 'clt', 'CLT-P': 'clt', 'HS': 'hs', 'BS-P': 'bs', 'WSR': 'wsr', 'WSR-P': 'wsr'}


def _interval(kind, x, N, lo, hi, k_mult, alpha, looks):
    if kind == 'clt':
        return iv_clt(x, N, k_mult, alpha, looks)
    if kind == 'hs':
        return iv_hs(x, N, lo, hi, k_mult, alpha, looks)
    if kind == 'bs':
        return iv_bs(x, N, lo, hi, k_mult, alpha, looks)
    return iv_wsr(x, N, lo, hi, k_mult, alpha, looks)


class Query:
    def __init__(self, Ys, Ps, beta, a, b, kind, r, c=None, k_top=None, name=''):
        self.Ys, self.Ps, self.beta, self.a, self.b = Ys, Ps, beta, a, b
        self.kind, self.r, self.c, self.k_top, self.name = kind, r, c, k_top, name
        self.N = np.array([len(y) for y in Ys])
        self.mu = np.array([float(y.mean()) for y in Ys])
        self.muP = np.array([float(p.mean()) for p in Ps])
        self.K = len(Ys)
        # ranges of y and of the rectified z = y - beta (p - muP)
        self.rng_y = [(a, b)] * self.K
        self.rng_z = [(a - beta * (b - mp), b - beta * (a - mp)) for mp in self.muP]


def _having_group(meth, x, N, rg, q, alpha, looks):
    c, r, K = q.c, q.r, q.K
    lo, hi = rg
    if meth.startswith('BET'):
        a = alpha / (2 * K)
        t_above = wor_bet_first(x, N, c, lo, hi, a, looks)
        t_below = wor_bet_upper(x, N, c, lo, hi, a, looks)
        t_lt = wor_bet_upper(x, N, c + r, lo, hi, a, looks)
        t_gt = wor_bet_first(x, N, c - r, lo, hi, a, looks)
        t_ind = max(t_lt, t_gt) if (t_lt >= 0 and t_gt >= 0) else -1
        cands = [(t_above, 'above'), (t_below, 'below'), (t_ind, 'ind')]
    else:
        Lb, Ub = _interval(IV[meth], x, N, lo, hi, K, alpha, looks)
        cands = [(first_true(Lb > c), 'above'), (first_true(Ub < c), 'below'),
                 (first_true((Ub < c + r) & (Lb > c - r)), 'ind')]
    cands = [z for z in cands if z[0] >= 0]
    return min(cands) if cands else None


def _pair(meth, xi, xj, Ni, Nj, rgi, rgj, q, alpha, looks, P, ivs=None):
    r = q.r
    if meth.startswith('BET'):
        a = alpha / (2 * P)
        mu_rng = (q.a, q.b)
        n = int(looks[-1])
        t_gt = wor_pair_first(xi, xj, Ni, Nj, 0.0, rgi, rgj, mu_rng, a, looks, n=n)
        t_lt = wor_pair_upper(xi, xj, Ni, Nj, 0.0, rgi, rgj, mu_rng, a, looks, n=n)
        t_a = wor_pair_upper(xi, xj, Ni, Nj, r, rgi, rgj, mu_rng, a, looks, n=n)
        t_b = wor_pair_first(xi, xj, Ni, Nj, -r, rgi, rgj, mu_rng, a, looks, n=n)
        t_tie = max(t_a, t_b) if (t_a >= 0 and t_b >= 0) else -1
        cands = [(t_gt, 'gt'), (t_lt, 'lt'), (t_tie, 'tie')]
    else:
        (Li, Ui), (Lj, Uj) = ivs
        m_ = len(looks)
        Li, Ui, Lj, Uj = Li[:m_], Ui[:m_], Lj[:m_], Uj[:m_]
        cands = [(first_true(Li > Uj), 'gt'), (first_true(Lj > Ui), 'lt'),
                 (first_true((Ui - Lj < r) & (Uj - Li < r)), 'tie')]
    cands = [z for z in cands if z[0] >= 0]
    return min(cands) if cands else None


def pair_horizon(Li, Ni, Lj, Nj):
    """Lockstep draws; a group whose stream covers all its tuples contributes its exact mean once exhausted."""
    ei, ej = Li >= Ni, Lj >= Nj
    if ei and ej:
        return int(max(Ni, Nj))
    if ei:
        return int(Lj)
    if ej:
        return int(Li)
    return int(min(Li, Lj))


def _pair_events(q, X, RG, meth, L, alpha, pairs, P):
    """Precompute certificate events on fixed streams without charging observations.

    Scheduling reveals an event only after both actual prefixes reach its horizon.
    """
    K, N = q.K, q.N
    ivs = None
    if not meth.startswith('BET'):
        ivs = [_interval(IV[meth], X[g][:L[g]], N[g], RG[g][0], RG[g][1], K, alpha, looks_for(L[g])) for g in range(K)]
    else:
        H = max(pair_horizon(L[i], N[i], L[j], N[j]) for (i, j) in pairs)
        U = [_wor_unbiased(X[g][:L[g]], N[g], H if L[g] >= N[g] else int(L[g])) for g in range(K)]
        lo_b = [RG[g][0] * U[g][1] + U[g][2] for g in range(K)]
        hi_b = [RG[g][1] * U[g][1] + U[g][2] for g in range(K)]
        a_p = alpha / (2 * P); mu_rng = (q.a, q.b)
    tpair, dpair = {}, {}
    for (i, j) in pairs:
        n = pair_horizon(L[i], N[i], L[j], N[j])
        lk = looks_for(n)
        if meth.startswith('BET'):
            # Search geometrically through the same batch endpoints. Once any
            # decision fires, later observations cannot change its first event.
            prefix = min(10 * BATCH, n)
            while True:
                lk = looks_for(prefix)
                Xp = U[i][0][:prefix] - U[j][0][:prefix]
                xmin = lo_b[i][:prefix] - hi_b[j][:prefix]
                xmax = hi_b[i][:prefix] - lo_b[j][:prefix]
                t_gt = pair_first_from(Xp, xmin, xmax, 0.0, mu_rng, a_p, lk)
                t_lt = pair_first_from(-Xp, -xmax, -xmin, 0.0, mu_rng, a_p, lk)
                t_a = pair_first_from(-Xp, -xmax, -xmin, -q.r, mu_rng, a_p, lk)
                t_b = pair_first_from(Xp, xmin, xmax, -q.r, mu_rng, a_p, lk)
                t_tie = max(t_a, t_b) if t_a >= 0 and t_b >= 0 else -1
                candidates = [(t_gt, 'gt'), (t_lt, 'lt'), (t_tie, 'tie')]
                candidates = [event for event in candidates if event[0] >= 0]
                d = min(candidates) if candidates else None
                if d is not None or prefix == n:
                    break
                prefix = min(n, 4 * prefix)
        else:
            lki, lkj = looks_for(L[i]), looks_for(L[j])
            ii = np.minimum(np.searchsorted(lki, np.minimum(lk, L[i])), len(lki) - 1)
            jj = np.minimum(np.searchsorted(lkj, np.minimum(lk, L[j])), len(lkj) - 1)
            sub = ((ivs[i][0][ii], ivs[i][1][ii]), (ivs[j][0][jj], ivs[j][1][jj]))
            d = _pair(meth, None, None, N[i], N[j], RG[i], RG[j], q, alpha, lk, P, sub)
        if d is None:
            tpair[(i, j)] = None; dpair[(i, j)] = None
        else:
            tpair[(i, j)] = int(lk[d[0]]); dpair[(i, j)] = d[1]
    return {pair: (tpair[pair], dpair[pair]) for pair in pairs}


def _eval(q, X, RG, meth, L, alpha, pairs, P, return_record=False):
    """Replay actual batch reads and share round transitions with live execution.

    Prefixes in X are precomputed oracle responses. Only the lengths advanced by
    the scheduler may affect candidates, decisions, or charged calls.
    """
    available = np.asarray(L, dtype=int)
    lengths = np.zeros(q.K, dtype=int)
    state = DecisionState(q, q.N, q.k_top)
    cumulative = [np.cumsum(values[:limit]) for values, limit in zip(X, available)]
    group_events = {}
    pair_events = {}
    if q.kind == 'having':
        for group in range(q.K):
            looks = looks_for(available[group])
            decision = _having_group(meth, X[group][:available[group]], q.N[group],
                                     RG[group], q, alpha, looks)
            group_events[group] = None if decision is None else (int(looks[decision[0]]), decision[1])
    else:
        pair_events = _pair_events(q, X, RG, meth, available, alpha, pairs, P)

    def group_certificate(group):
        event = group_events[group]
        return event[1] if event is not None and event[0] <= lengths[group] else None

    def pair_certificate(i, j):
        time, side = pair_events[i, j]
        horizon = pair_horizon(lengths[i], q.N[i], lengths[j], q.N[j])
        return side if time is not None and time <= horizon else None

    trace = []
    while not state.done:
        reading = sorted(state.active)
        next_lengths = lengths.copy()
        for group in reading:
            next_lengths[group] = min(lengths[group] + BATCH, available[group])
        if np.array_equal(next_lengths, lengths):
            break
        lengths = next_lengths
        estimates = np.array([cumulative[group][lengths[group] - 1] / lengths[group]
                              for group in range(q.K)])
        previous_pairs = set(state.pair_decisions)
        previous_groups = set(state.group_decisions)
        state.update(lengths, estimates, group_certificate, pair_certificate)
        trace.append(dict(round=len(trace) + 1, calls=int(lengths.sum()),
                          per_group_calls=lengths.tolist(), read_groups=reading,
                          next_active=sorted(state.active), answer=state.answer.copy(),
                          new_pair_decisions={f'{i},{j}': side
                                              for (i, j), side in state.pair_decisions.items()
                                              if (i, j) not in previous_pairs},
                          new_group_decisions={str(group): side
                                               for group, side in state.group_decisions.items()
                                               if group not in previous_groups}))
    answer_error, decision_error = error_flags(q, state) if state.done else (False, False)
    record = dict(calls=int(lengths.sum()), per_group_calls=lengths.tolist(),
                  answer=state.answer, answer_err=answer_error, decision_err=decision_error,
                  err=answer_error or decision_error, complete=bool(state.done), rounds=trace)
    if return_record:
        return record
    return lengths.sum() / float(q.N.sum()), record['err'], not state.done


def run_query(q, R=50, alpha=0.05, seed=0, cap=10 ** 12,
              methods=ALL_METHODS, start=50_000, grow=4, return_runs=False):
    """Run shared permutations with actual prefix accounting and separate errors."""
    rng = np.random.default_rng(seed)
    N = q.N
    maximum = np.minimum(N, cap)
    pairs = [(i, j) for i in range(q.K) for j in range(i + 1, q.K)]
    pair_count = max(len(pairs), 1)
    records = {method: [] for method in methods}
    for repetition in range(R):
        raw, rectified = [], []
        for group in range(q.K):
            indices = rng.permutation(N[group])[:maximum[group]]
            y = q.Ys[group][indices]
            p = q.Ps[group][indices]
            raw.append(y)
            rectified.append(y - q.beta * (p - q.muP[group]))
        for method in methods:
            use_proxy = method.endswith('-P')
            values = rectified if use_proxy else raw
            ranges = q.rng_z if use_proxy else q.rng_y
            available = np.minimum(maximum, start)
            while True:
                record = _eval(q, values, ranges, method, available, alpha,
                               pairs, pair_count, return_record=True)
                if record['complete'] or np.all(available >= maximum):
                    break
                available = np.minimum(maximum, available * grow)
            if not record['complete']:
                raise RuntimeError(f'{q.name} {method} reached its prefix cap without an answer')
            record['repetition'] = repetition
            records[method].append(record)
    summary = {}
    for method, runs in records.items():
        costs = [record['calls'] / float(N.sum()) for record in runs]
        summary[method] = dict(
            cost=float(np.mean(costs)), cost_med=float(np.median(costs)),
            err=float(np.mean([record['err'] for record in runs])),
            answer_err=float(np.mean([record['answer_err'] for record in runs])),
            decision_err=float(np.mean([record['decision_err'] for record in runs])),
            cost_sd=float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
            cost_q10=float(np.quantile(costs, 0.1)), cost_q90=float(np.quantile(costs, 0.9)))
        if return_runs:
            summary[method]['runs'] = runs
    return summary
