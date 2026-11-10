"""Regressions for exhausted rankings and actual-prefix Top-k execution."""

import ast
import contextlib
import io
import itertools
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import looks_for
from query_schedule import DecisionState, error_flags
from run import (ALL_METHODS, Query, _eval, _having_group, _interval,
                 _pair, _pair_events, pair_horizon)


def live_on_arrays(streams, proxies, query, method):
    """Call the real live entry point with a fixed oracle and a no-op CUDA sync."""
    source = Path(__file__).with_name('online_benchmark.py').read_text(encoding='utf8')
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'live_query')
    namespace = dict(np=np, SimpleNamespace=SimpleNamespace, time=time,
                     torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
                     DecisionState=DecisionState, _having_group=_having_group,
                     _interval=_interval, _pair=_pair, pair_horizon=pair_horizon,
                     looks_for=looks_for, BATCH=100)
    exec(compile(ast.Module(body=[function], type_ignores=[]), source, 'exec'), namespace)
    full = np.concatenate(streams)
    full_proxy = np.concatenate(proxies)
    ends = np.cumsum([0] + [len(values) for values in streams])
    permutations = [np.arange(ends[group], ends[group + 1]) for group in range(query.K)]

    class Oracle:
        calls = 0
        seconds = 0.0

        def __call__(self, indices):
            self.calls += len(indices)
            return full[indices]

    with contextlib.redirect_stdout(io.StringIO()):
        result, _ = namespace['live_query'](
            Oracle(), permutations, full_proxy, query.beta, query.kind, method,
            .05, query.r, query.c, query.k_top)
    return result


def replay(streams, proxies, kind, method, beta=0.0, margin=.13, alpha=.05):
    query = Query(streams, proxies, beta, 0., 1., kind, margin, c=.2, k_top=3)
    values = streams
    ranges = query.rng_y
    if method.endswith('-P'):
        values = [y - beta * (p - p.mean()) for y, p in zip(streams, proxies)]
        ranges = query.rng_z
    pairs = [(i, j) for i in range(query.K) for j in range(i + 1, query.K)]
    record = _eval(query, values, ranges, method, query.N, alpha,
                   pairs, max(len(pairs), 1), return_record=True)
    return query, record


class QueryScheduleRegression(unittest.TestCase):
    def test_exhausted_order_exactly_enumerated(self):
        populations = [[.15, .15, 1.], [.6, .6, 0.], [.05, .05, 1.]]
        permutations = [sorted(set(itertools.permutations(values))) for values in populations]
        for arrangement in itertools.product(*permutations):
            streams = [np.array(values) for values in arrangement]
            query, result = replay(streams, [np.zeros(3)] * 3, 'order', 'BET',
                                   margin=.01, alpha=.01)
            self.assertTrue(result['complete'])
            self.assertFalse(result['answer_err'], arrangement)
            self.assertEqual(result['answer'], [0, 1, 2])
            self.assertEqual(result['calls'], sum(query.N))

    def test_topk_counts_reactivation(self):
        rng = np.random.default_rng(0)
        sizes = [1100, 1400, 1700, 2000, 2300, 2600]
        rates = [.70, .56, .51, .49, .46, .20]
        streams = [rng.binomial(1, rate, size).astype(float)
                   for rate, size in zip(rates, sizes)]
        proxies = [np.zeros(size) for size in sizes]
        query, cached = replay(streams, proxies, 'topk', 'BET')
        live = live_on_arrays(streams, proxies, query, 'BET')
        self.assertEqual(cached['calls'], 3100)
        self.assertEqual(cached['per_group_calls'], [500, 800, 600, 700, 400, 100])
        self.assertEqual(cached['calls'], live['calls'])
        self.assertEqual(cached['answer'], live['answer'])

    def test_live_cached_match_on_unequal_partial_batches(self):
        rng = np.random.default_rng(41)
        sizes = [173, 251, 310, 433, 527]
        streams = [rng.binomial(1, rate, size).astype(float)
                   for rate, size in zip([.1, .25, .4, .6, .8], sizes)]
        proxies = [np.clip(.2 + .6 * values + rng.normal(0, .1, len(values)), 0, 1)
                   for values in streams]
        for kind in ['having', 'order', 'topk']:
            for method in ['BET', 'BET-P', 'WSR', 'WSR-P']:
                with self.subTest(kind=kind, method=method):
                    query, cached = replay(streams, proxies, kind, method, beta=.8)
                    live = live_on_arrays(streams, proxies, query, method)
                    self.assertEqual(cached['calls'], live['calls'])
                    self.assertEqual(cached['per_group_calls'], live['per_group_calls'])
                    self.assertEqual(cached['answer'], live['answer'])

    def test_all_methods_charge_each_observation_once(self):
        rng = np.random.default_rng(6)
        streams = [rng.binomial(1, rate, 700).astype(float)
                   for rate in [.1, .2, .3, .4, .5, .6]]
        proxies = [np.zeros(len(values)) for values in streams]
        for method in ALL_METHODS:
            _, result = replay(streams, proxies, 'topk', method)
            previous = np.zeros(6, dtype=int)
            for round_record in result['rounds']:
                current = np.array(round_record['per_group_calls'])
                self.assertTrue(np.all(current >= previous))
                self.assertTrue(np.all(current - previous <= 100))
                self.assertEqual(round_record['calls'], int(current.sum()))
                previous = current
            self.assertEqual(result['calls'], sum(result['per_group_calls']))

    def test_answer_and_decision_errors_are_distinct(self):
        streams = [np.array([0., 1., 1., 1.])]
        query = Query(streams, [np.zeros(4)], 0., 0., 1., 'having', .1, c=.2)
        state = DecisionState(query, query.N)
        state.update(np.array([1]), np.array([.3]), lambda group: 'ind',
                     lambda i, j: None)
        self.assertEqual(error_flags(query, state), (False, True))

    def test_progressive_events_equal_full_stream_events(self):
        rng = np.random.default_rng(81)
        sizes = [173, 1001, 5511, 7307]
        streams = [rng.binomial(1, rate, size).astype(float)
                   for rate, size in zip([.12, .4, .42, .78], sizes)]
        query = Query(streams, [np.zeros(size) for size in sizes], 0., 0., 1., 'order', .03)
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        events = _pair_events(query, streams, query.rng_y, 'BET', query.N, .05, pairs, len(pairs))
        for i, j in pairs:
            looks = looks_for(max(sizes[i], sizes[j]))
            decision = _pair('BET', streams[i], streams[j], sizes[i], sizes[j],
                             query.rng_y[i], query.rng_y[j], query, .05, looks, len(pairs))
            expected = (None, None) if decision is None else (int(looks[decision[0]]), decision[1])
            self.assertEqual(events[i, j], expected)

    def test_selecting_all_groups_needs_no_labels(self):
        streams = [np.array([0., 1.]) for _ in range(3)]
        proxies = [np.zeros(2) for _ in streams]
        query, result = replay(streams, proxies, 'topk', 'BET')
        self.assertEqual(result['calls'], 0)
        self.assertEqual(result['answer'], [0, 1, 2])
        self.assertFalse(result['answer_err'])
        self.assertEqual(live_on_arrays(streams, proxies, query, 'BET')['calls'], 0)


if __name__ == '__main__':
    unittest.main()
