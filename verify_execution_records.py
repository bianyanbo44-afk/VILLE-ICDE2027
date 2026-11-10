"""Check every charged prefix and summary in the corrected cached execution records."""

import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
names = ['d1_main', 'd2_wor', 'd3_arxiv', 'abl_proxy', 'abl_margin',
         'abl_alpha', 'abl_groups', 'abl_bad']
totals = defaultdict(int)
errors = defaultdict(lambda: defaultdict(int))
reports = []

for name in names:
    summary = json.loads((root / (name + '.json')).read_text(encoding='utf-8-sig'))
    for key, entry in summary.items():
        if key.startswith('_'):
            continue
        path = root / entry['execution_record']
        record = json.loads(path.read_text(encoding='utf-8-sig'))
        assert record['query'] == key, (name, key)
        if 'seed' in record and 'seed' in entry:
            assert record['seed'] == entry['seed']
        method_counts = {}
        for method, runs in record['methods'].items():
            expected = 30 if name.startswith('d') else 10
            assert len(runs) == expected, (name, key, method, len(runs))
            for run in runs:
                assert run['complete']
                assert run['calls'] == sum(run['per_group_calls'])
                assert len(run['per_group_calls']) == entry['K']
                assert run['err'] == (run['answer_err'] or run['decision_err'])
                previous = [0] * entry['K']
                for step in run['rounds']:
                    now = step['per_group_calls']
                    increments = [a-b for a, b in zip(now, previous)]
                    assert all(0 <= value <= 100 for value in increments)
                    assert step['calls'] == sum(now)
                    actual_reads = {i for i, v in enumerate(increments) if v}
                    scheduled = set(step['read_groups'])
                    assert actual_reads <= scheduled
                    # Some trails retain exhausted groups in the active schedule.
                    # They add zero calls and never advance beyond their final prefix.
                    assert all(now[i] == run['per_group_calls'][i]
                               for i in scheduled - actual_reads)
                    previous = now
                    totals['rounds'] += 1
                assert previous == run['per_group_calls']
                totals['executions'] += 1
                totals['charged_calls'] += run['calls']
                for flag in ('answer_err', 'decision_err'):
                    errors[method][flag] += int(run[flag])
            for flag in ('answer_err', 'decision_err', 'err'):
                assert math.isclose(sum(run[flag] for run in runs)/expected,
                                    entry['summary'][method][flag], abs_tol=1e-12)
            if 'N' in entry:
                cost = sum(run['calls'] for run in runs)/expected/entry['N']
                assert math.isclose(cost, entry['summary'][method]['cost'], abs_tol=1e-12)
            method_counts[method] = len(runs)
        reports.append({'source': name, 'query': key, 'methods': method_counts,
                        'execution_record': entry['execution_record'], 'pass': True})
        totals['queries'] += 1
        del record
        gc.collect()
    print(name, 'complete', dict(totals), flush=True)

result = {'pass': True, 'totals': dict(totals), 'errors': dict(errors), 'queries': reports}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print('PASS', dict(totals), flush=True)
