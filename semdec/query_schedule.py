"""Shared round transitions for cached and live decision queries."""

import numpy as np


class DecisionState:
    """Track decisions using only the prefixes actually read by the caller."""

    def __init__(self, query, sizes, topk=None):
        self.query = query
        self.sizes = np.asarray(sizes, dtype=int)
        self.count = len(self.sizes)
        self.topk = topk
        self.pairs = [(i, j) for i in range(self.count)
                      for j in range(i + 1, self.count)]
        self.active = set(range(self.count))
        self.group_decisions = {}
        self.pair_decisions = {}
        self.answer = []
        self.done = False
        if query.kind == 'topk' and topk == self.count:
            self.answer = list(range(self.count))
            self.active = set()
            self.done = True

    def update(self, lengths, estimates, group_certificate, pair_certificate):
        """Resolve one observed round and choose the next active groups.

        Certificate callbacks return a side or None using the supplied prefixes.
        Exact population decisions take precedence on the exhaustion round.
        """
        if self.query.kind == 'having':
            for group in sorted(self.active):
                if lengths[group] == self.sizes[group]:
                    side = 'above' if estimates[group] > self.query.c else 'below'
                else:
                    side = group_certificate(group)
                if side is not None:
                    self.group_decisions[group] = side
            self.active = set(range(self.count)) - self.group_decisions.keys()
            self.answer = sorted(group for group, side in self.group_decisions.items()
                                 if side == 'above' or
                                 (side == 'ind' and estimates[group] > self.query.c))
            self.done = not self.active
            return

        for i, j in self.pairs:
            if (i, j) in self.pair_decisions:
                continue
            if lengths[i] == self.sizes[i] and lengths[j] == self.sizes[j]:
                difference = estimates[i] - estimates[j]
                side = 'gt' if difference > 0 else ('lt' if difference < 0 else 'tie')
            else:
                side = pair_certificate(i, j)
            if side is not None:
                self.pair_decisions[i, j] = side

        if self.query.kind == 'order':
            wins = np.zeros(self.count, dtype=int)
            for (i, j), side in self.pair_decisions.items():
                if side == 'gt':
                    wins[i] += 1
                elif side == 'lt':
                    wins[j] += 1
            self.answer = np.lexsort((-estimates, -wins)).tolist()
            self.done = len(self.pair_decisions) == len(self.pairs)
            self.active = {group for pair in self.pairs
                           if pair not in self.pair_decisions for group in pair}
            return

        candidate = set(np.argsort(-estimates)[:self.topk].tolist())
        boundary = [(i, j) for i, j in self.pairs
                    if (i in candidate) != (j in candidate)]
        self.done = all(self.pair_decisions.get((i, j)) in
                        ('tie', 'gt' if i in candidate else 'lt')
                        for i, j in boundary)
        self.active = {group for pair in boundary
                       if pair not in self.pair_decisions for group in pair}
        if not self.active and not self.done:
            self.active = set(np.flatnonzero(lengths < self.sizes).tolist())
        if np.all(lengths == self.sizes):
            candidate = set(np.argsort(-estimates)[:self.topk].tolist())
            self.done = True
        self.answer = sorted(candidate)


def error_flags(query, state):
    """Separate the returned answer's correctness from certificate correctness."""
    if not state.done:
        raise ValueError('An incomplete query has no final-answer error flag.')
    means = query.mu
    margin = query.r
    selected = set(state.answer)
    if query.kind == 'having':
        answer_error = any((mean >= query.c + margin and group not in selected) or
                           (mean <= query.c - margin and group in selected)
                           for group, mean in enumerate(means))
        decision_error = any(
            abs(means[group] - query.c) >= margin and
            (side == 'ind' or
             (side == 'above' and means[group] < query.c) or
             (side == 'below' and means[group] > query.c))
            for group, side in state.group_decisions.items())
    else:
        relevant = state.pair_decisions.items()
        if query.kind == 'topk':
            relevant = [((i, j), side) for (i, j), side in relevant
                        if (i in selected) != (j in selected)]
        decision_error = any(
            abs(means[i] - means[j]) >= margin and
            (side == 'tie' or
             (side == 'gt' and means[i] < means[j]) or
             (side == 'lt' and means[i] > means[j]))
            for (i, j), side in relevant)
        if query.kind == 'order':
            positions = {group: rank for rank, group in enumerate(state.answer)}
            answer_error = any(means[i] >= means[j] + margin and
                               positions[i] > positions[j]
                               for i in range(query.K) for j in range(query.K))
        else:
            kth = np.sort(means)[-query.k_top]
            answer_error = (len(selected) != query.k_top or
                            min(means[list(selected)]) < kth - margin)
    return bool(answer_error), bool(decision_error)
