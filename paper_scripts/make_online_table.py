"""Build the live systems table and macros from completed measured runs.

python -I make_online_table.py RESULTS_DIR PAPER_DIR
"""
import json
import math
from pathlib import Path
import sys


def build(results_dir, paper_dir):
    (paper_dir / 'tables').mkdir(parents=True, exist_ok=True)
    records = []
    configs = []
    labels = [('qwen', 'Qwen2.5-1.5B'), ('smol', 'SmolLM2-1.7B')]
    query_names = {'having': 'HAVING', 'order': 'ORDER BY', 'topk': 'Top-5'}
    for model_name, label in labels:
        directory = results_dir / f'live_{model_name}_20261109'
        config = json.loads((directory / 'config.json').read_text())
        exact = json.loads((directory / 'exact.json').read_text())
        configs.append(config)
        for kind in query_names:
            ville = json.loads((directory / f'BET-P_{kind}.json').read_text())
            baseline = json.loads((directory / f'WSR_{kind}.json').read_text())
            for result in [ville, baseline]:
                if not result['live_labels_match_exact']:
                    raise ValueError('Live and exact runs must use identical per-tuple judgments.')
                if result['calls'] != sum(result['per_group_calls']):
                    raise ValueError('Every actual model call must be counted.')
            records.append(dict(model=label, kind=kind, ville=ville, baseline=baseline, exact=exact))
    if len({config['workload_sha256'] for config in configs}) != 1:
        raise ValueError('The model families must use the same workload.')
    total = configs[0]['N']
    rows = []
    for index, record in enumerate(records):
        ville, baseline = record['ville'], record['baseline']
        model = record['model'] if index % 3 == 0 else ''
        if index == 3:
            rows.append(r'\midrule')
        rows.append(f"{model} & {query_names[record['kind']]} & "
                    f"{ville['calls']:,} & {baseline['calls']:,} & "
                    f"{ville['seconds']:.1f} & {baseline['seconds']:.1f} & "
                    f"{ville['exact_seconds']:.1f} & {ville['speedup_vs_exact']:.2f}$\\times$ "
                    + r'\\')
    table = r'''\begin{table*}[t]
\centering
\caption{Live execution over the same 32,000 reviews in 16 product groups with two model families.
CS denotes the Evergreen-style confidence-sequence baseline. Calls count fresh model evaluations by each method.
Latencies include prompt processing, model inference and certificate updates, with the model
resident and proxies prepared. Each model's complete exact scan is measured once and supplies the shared reference
latency for the three queries over the same predicate. The SQL aggregation time is added per query.
Speedup is measured exact-query latency divided by VILLE latency.}
\label{tab:online}
\footnotesize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{llrrrrrr}
\toprule
& & \multicolumn{2}{c}{LLM calls} & \multicolumn{3}{c}{Measured latency (s)} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
Model & Query & \sys{} & CS & \sys{} & CS & Exact & Speedup \\
\midrule
'''
    if total != 32000 or configs[0]['K'] != 16:
        raise ValueError('Update the table caption to match the frozen workload.')
    table += '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular}
\end{table*}
'''
    (paper_dir / 'tables/tab_online.tex').write_text(table, encoding='utf-8')
    having_speedups = [record['ville']['speedup_vs_exact'] for record in records
                       if record['kind'] == 'having']
    max_overhead_percent = max(
        100 * max(record['ville']['seconds'] - record['ville']['llm_seconds'], 0) /
        record['ville']['seconds'] for record in records)
    overhead_upper = math.ceil(max_overhead_percent * 10) / 10
    macros = dict(LiveHavingMin=f'{min(having_speedups):.1f}',
                  LiveHavingMax=f'{max(having_speedups):.1f}',
                  LiveMaxOverhead=f'{overhead_upper:.1f}',
                  LiveN=f'{total:,}'.replace(',', '{,}'))
    (paper_dir / 'numbers_online.tex').write_text(
        '% generated from completed live measurements by make_online_table.py\n' +
        '\n'.join(f'\\newcommand{{\\num{name}}}{{{value}}}' for name, value in macros.items()) + '\n',
        encoding='utf-8')
    summary = dict(workload=configs[0], records=records, macros=macros,
                   ville_wrong=sum(record['ville']['wrong'] for record in records),
                   baseline_wrong=sum(record['baseline']['wrong'] for record in records))
    (results_dir / 'online_systems_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(macros=macros, ville_wrong=summary['ville_wrong'],
                         baseline_wrong=summary['baseline_wrong']), indent=2))


if __name__ == '__main__':
    build(Path(sys.argv[1]), Path(sys.argv[2]))
