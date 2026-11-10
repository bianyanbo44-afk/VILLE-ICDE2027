"""Recompute the main comparisons using saved labels, proxies and original seeds.

Usage: python -I replay_saved.py stars|llm|arxiv DATA RESULTS OUTPUT [--runs 30]
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import Query, run_query


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1), encoding='utf8')
    temporary.replace(path)


def load_inputs(dataset, data_root):
    if dataset == 'stars':
        data = np.load(data_root / 'sem1_amazon.npz')
        keep = ~data['train']
        categories = list(data['cats'])
        category = data['cat'][keep]
        years = data['year'][keep]
        product = data['asin'][keep]
        rating = data['rating'][keep].astype(float)
        labels = {'NEG': (rating <= 2).astype(float),
                  'POS': (rating == 5).astype(float), 'RAT': rating}
        proxies = {name: data[f'p_{key}'][keep].astype(float)
                   for name, key in [('NEG', 'neg'), ('POS', 'pos'), ('RAT', 'rat')]}
        groups = {'cat': [np.flatnonzero(category == i) for i in range(len(categories))],
                  'year': [np.flatnonzero(years == year) for year in np.unique(years)
                           if (years == year).sum() >= 2000]}
        sizes = np.bincount(category, minlength=len(categories))
        for number in np.argsort(-sizes)[:3]:
            indices = np.flatnonzero(category == number)
            names, counts = np.unique(product[indices], return_counts=True)
            chosen = [name for name in names[np.argsort(-counts)][:30]
                      if (product[indices] == name).sum() >= 2000]
            if len(chosen) >= 5:
                groups[f'prod-{categories[number]}'] = [indices[product[indices] == name]
                                                        for name in chosen]
        names = [f'{group}|{label}|{kind}'
                 for group in ['cat', 'year', 'prod-Amazon_Fashion', 'prod-Appliances']
                 for label in labels for kind in ['having', 'order', 'topk']]
        seeds = {name: 9000 + 17 * (index + 1) for index, name in enumerate(names)}
    else:
        folder = data_root / ('sem2' if dataset == 'llm' else 'sem3')
        if not folder.exists():
            folder = data_root / ('amz_llm' if dataset == 'llm' else 'arxiv_llm')
        meta = np.load(folder / 'meta.npz')
        judgments = np.load(folder / 'llm_labels.npz')
        stored_proxy = np.load(folder / 'proxies.npz')
        indices = np.flatnonzero(~meta['train'])
        if dataset == 'llm':
            labels = {'DEFECT': (judgments['p_defect'][:, 0] > .5).astype(float),
                      'VALUE': (judgments['p_value'][:, 0] > .5).astype(float),
                      'SENT': (judgments['p_sent'].argmax(1) + 1).astype(float)}
            groups = {}
            for number, category in enumerate(meta['cats']):
                rows = indices[meta['cat'][indices] == number]
                names, counts = np.unique(meta['asin'][rows], return_counts=True)
                chosen = names[np.argsort(-counts)][:30]
                groups[f'prod-{category}'] = [rows[meta['asin'][rows] == name] for name in chosen]
            years = meta['year'][indices]
            groups['year'] = [indices[years == year] for year in np.unique(years)
                              if (years == year).sum() >= 1000]
            names = [f'{group}|{label}|{kind}' for group in groups for label in labels
                     for kind in ['having', 'order', 'topk']]
            seeds = {name: 12000 + 31 * (index + 1) for index, name in enumerate(names)}
        else:
            labels = {name: (judgments[key][:, 0] > .5).astype(float)
                      for name, key in [('SOTA', 'p_sota'), ('EMP', 'p_emp')]}
            years = meta['year'][indices]
            groups = {'year': [indices[years == year] for year in np.unique(years)]}
            counts = np.bincount(meta['cat'][indices], minlength=len(meta['cats']))
            groups['area'] = [indices[meta['cat'][indices] == number]
                              for number in np.argsort(-counts)[:16]]
            names = [f'{group}|{label}|{kind}' for label in labels for group in groups
                     for kind in ['having', 'order', 'topk']]
            seeds = {name: 13000 + 31 * (index + 1) for index, name in enumerate(names)}
        proxies = {name: stored_proxy[f'p_{name.lower()}'].astype(float) for name in labels}
    return labels, proxies, groups, seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['stars', 'llm', 'arxiv'])
    parser.add_argument('data', type=Path)
    parser.add_argument('previous', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--grouping', help='Replay one grouping, retaining the original query seeds')
    parser.add_argument('--label', help='Replay one saved attribute, retaining the original seeds')
    args = parser.parse_args()
    original = json.loads(args.previous.read_text(encoding='utf8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_dir = args.output.parent / (args.output.stem + '_runs')
    records_dir.mkdir(exist_ok=True)
    results = json.loads(args.output.read_text(encoding='utf8')) if args.output.exists() else {}
    labels, proxies, groups, seeds = load_inputs(args.dataset, args.data)
    results['_execution'] = dict(version='actual-prefix-v2', runs=args.runs,
                                 numpy=np.__version__, seed_rule='original per-query seeds',
                                 source_sha256=hashlib.sha256(
                                     Path(__file__).with_name('run.py').read_bytes()).hexdigest())
    if '_info' in original:
        results['_info'] = original['_info']
    for name, metadata in original.items():
        if name.startswith('_') or name in results:
            continue
        group_name, label_name, kind = name.split('|')
        if args.grouping and group_name != args.grouping:
            continue
        if args.label and label_name != args.label:
            continue
        ids = groups[group_name]
        ys = [labels[label_name][rows] for rows in ids]
        ps = [proxies[label_name][rows] for rows in ids]
        low, high = (1., 5.) if label_name in ['RAT', 'SENT'] else (0., 1.)
        query = Query(ys, ps, metadata['beta'], low, high, kind,
                      metadata['r'], c=metadata['c'], k_top=metadata['k'], name=name)
        if query.K != metadata['K'] or int(query.N.sum()) != metadata['N']:
            raise ValueError(f'Cached workload changed for {name}')
        started = time.perf_counter()
        summary = run_query(query, R=args.runs, seed=seeds[name], return_runs=True)
        audit = dict(query=name, seed=seeds[name], methods={})
        for method, values in summary.items():
            audit['methods'][method] = values.pop('runs')
        audit_name = hashlib.sha256(name.encode()).hexdigest()[:16] + '.json'
        save_json(records_dir / audit_name, audit)
        results[name] = {**metadata, 'summary': summary, 'seed': seeds[name],
                         'execution_record': f'{records_dir.name}/{audit_name}'}
        save_json(args.output, results)
        status = ' '.join(f'{method}={values["cost"]:.4f}/{values["answer_err"]:.3f}'
                          for method, values in summary.items())
        print(f'{name} {status} ({time.perf_counter()-started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
