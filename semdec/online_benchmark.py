"""Live, DuckDB-backed decision queries with measured exact-scan references.

Run with a CUDA environment, for example
  python -I online_benchmark.py --model MODEL_DIR --name qwen --data DATA_DIR --out OUT_DIR
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import duckdb
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import looks_for
from query_schedule import DecisionState
from run import _having_group, _interval, _pair, pair_horizon

SYSTEM = 'You are a precise annotator of customer product reviews.'
QUESTION = ('Does the reviewer report that the product was defective, broken, damaged, '
            'or stopped working? Answer with only Yes or No.')
BATCH = 100


def active_time():
    """Windows interrupt time excludes host standby, allowing interrupted runs to be detected."""
    if os.name == 'nt':
        ticks = ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks)):
            raise OSError('Cannot check active host time.')
        return ticks.value / 10_000_000
    return time.perf_counter()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class LocalJudge:
    def __init__(self, model_dir, texts, token_budget):
        self.texts = texts
        self.token_budget = token_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, attn_implementation='sdpa',
            local_files_only=True).cuda().eval()
        answer_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in ['Yes', 'No']]
        if any(len(ids) != 1 for ids in answer_ids):
            raise ValueError('The Yes/No decision requires single-token answers.')
        self.answer_ids = torch.tensor([ids[0] for ids in answer_ids], device='cuda')
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, indices):
        torch.cuda.synchronize()
        started = time.perf_counter()
        active_started = active_time()
        prompts = [self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM},
             {'role': 'user', 'content': f'Review:\n{self.texts[i][:1200]}\n\nQuestion: {QUESTION}'}],
            tokenize=False, add_generation_prompt=True) for i in indices]
        lengths = np.array([len(prompt) for prompt in prompts])
        order = np.argsort(lengths, kind='stable')
        result = np.zeros(len(indices))
        start = 0
        while start < len(indices):
            end, longest = start, 0
            while end < len(indices):
                estimate = int(lengths[order[end]]) // 3 + 8
                if end > start and (end - start + 1) * max(longest, estimate) > self.token_budget:
                    break
                longest = max(longest, estimate)
                end += 1
            selected = order[start:end]
            encoded = self.tokenizer([prompts[i] for i in selected], padding=True,
                                     return_tensors='pt').to('cuda')
            with torch.inference_mode():
                logits = self.model(**encoded, logits_to_keep=1).logits[:, -1, :].float()
                result[selected] = (logits[:, self.answer_ids[0]] >
                                    logits[:, self.answer_ids[1]]).float().cpu().numpy()
            start = end
        torch.cuda.synchronize()
        if time.perf_counter() - started - (active_time() - active_started) > 5:
            raise RuntimeError('Host standby interrupted the model call; rerun this incomplete measurement.')
        self.calls += len(indices)
        self.seconds += time.perf_counter() - started
        return result


def load_workload(data_dir, max_groups, cap):
    meta = np.load(data_dir / 'meta.npz')
    texts = [json.loads(line)['text'] for line in
             (data_dir / 'texts.jsonl').open(encoding='utf-8')]
    connection = duckdb.connect()
    connection.execute('CREATE TABLE reviews(row_id INTEGER, product VARCHAR)')
    query_rows = np.flatnonzero((~meta['train']) & (meta['cat'] == 0))
    connection.executemany('INSERT INTO reviews VALUES (?, ?)',
                          [(int(i), str(meta['asin'][i])) for i in query_rows])
    grouped = connection.execute(
        'SELECT product, list(row_id ORDER BY row_id) AS ids FROM reviews '
        'GROUP BY product ORDER BY count(*) DESC, product LIMIT ?', [max_groups]).fetchall()
    selection_rng = np.random.default_rng(20261109)
    groups, names = [], []
    for product, indices in grouped:
        indices = np.array(indices, dtype=np.int64)
        if len(indices) > cap:
            indices = np.sort(selection_rng.choice(indices, cap, replace=False))
        groups.append(indices)
        names.append(product)
    connection.close()
    train = np.flatnonzero(meta['train'])
    if np.intersect1d(train, np.concatenate(groups)).size:
        raise ValueError('Proxy training and query rows must be disjoint.')
    return texts, groups, names, train


def fit_proxy(judge, texts, train, groups, out_dir):
    cache = out_dir / 'proxy.npz'
    if cache.exists() and (out_dir / 'proxy.json').exists():
        data = np.load(cache)
        return data['p'], float(data['beta']), json.loads((out_dir / 'proxy.json').read_text())
    started = time.perf_counter()
    y = np.concatenate([judge(train[start:start + BATCH]) for start in range(0, len(train), BATCH)])
    vectorizer = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2),
                                  alternate_sign=False, norm=None)
    tfidf = TfidfTransformer(sublinear_tf=True)
    training_features = tfidf.fit_transform(vectorizer.transform([texts[i] for i in train]))
    model = LogisticRegression(C=4.0, max_iter=2000, random_state=20261109)
    cv_scores = cross_val_predict(model, training_features, y, cv=5,
                                  method='predict_proba')[:, 1]
    beta = float(np.clip(np.cov(y, cv_scores)[0, 1] /
                         max(np.var(cv_scores, ddof=1), 1e-12), 0, 1))
    model.fit(training_features, y)
    query_rows = np.concatenate(groups)
    query_features = tfidf.transform(vectorizer.transform([texts[i] for i in query_rows]))
    scores = np.zeros(len(texts))
    scores[query_rows] = model.predict_proba(query_features)[:, 1]
    info = dict(training_calls=len(train), seconds=time.perf_counter() - started,
                beta=beta, training_positive_fraction=float(y.mean()))
    np.savez(cache, p=scores, beta=beta, train_ids=train, train_y=y)
    save_json(out_dir / 'proxy.json', info)
    return scores, beta, info


def live_query(judge, permutations, proxy, beta, kind, method, alpha, margin, threshold, topk):
    count = len(permutations)
    sizes = np.array([len(group) for group in permutations])
    means_proxy = np.array([proxy[group].mean() for group in permutations])
    ranges = [(0.0, 1.0)] * count
    use_proxy = method.endswith('-P')
    if use_proxy:
        ranges = [(-beta * (1 - mean), 1 + beta * mean) for mean in means_proxy]
    query = SimpleNamespace(K=count, kind=kind, r=margin, c=threshold, a=0.0, b=1.0)
    observed = [[] for _ in permutations]
    state = DecisionState(query, sizes, topk)
    pairs = state.pairs
    traces = []
    initial_calls, initial_llm_seconds = judge.calls, judge.seconds
    torch.cuda.synchronize()
    started = time.perf_counter()
    while not state.done:
        for group in sorted(state.active):
            begin = len(observed[group])
            indices = permutations[group][begin:begin + BATCH]
            if len(indices):
                observed[group].extend(judge(indices).tolist())
        values = []
        for group, ys in enumerate(observed):
            values.append(np.array(ys, dtype=np.float64))
            if use_proxy:
                ids = permutations[group][:len(ys)]
                values[-1] -= beta * (proxy[ids] - means_proxy[group])
        lengths = np.array([len(ys) for ys in observed])
        estimates = np.array([values[group].mean() for group in range(count)])
        intervals = None
        if kind != 'having' and not method.startswith('BET'):
            intervals = [_interval('wsr', values[group], sizes[group], *ranges[group],
                                   count, alpha, looks_for(lengths[group]))
                         for group in range(count)]

        def group_certificate(group):
            decision = _having_group(method, values[group], sizes[group], ranges[group],
                                     query, alpha, looks_for(lengths[group]))
            return decision[1] if decision is not None else None

        def pair_certificate(i, j):
            horizon = pair_horizon(lengths[i], sizes[i], lengths[j], sizes[j])
            looks = looks_for(horizon)
            selected_intervals = None
            if intervals is not None:
                selected_intervals = []
                for group in [i, j]:
                    group_looks = looks_for(lengths[group])
                    positions = np.searchsorted(group_looks, np.minimum(looks, lengths[group]))
                    selected_intervals.append(tuple(bounds[positions] for bounds in intervals[group]))
            decision = _pair(method, values[i], values[j], sizes[i], sizes[j], ranges[i], ranges[j],
                             query, alpha, looks, len(pairs), selected_intervals)
            return decision[1] if decision is not None else None

        state.update(lengths, estimates, group_certificate, pair_certificate)
        elapsed = time.perf_counter() - started
        traces.append(dict(round=len(traces) + 1, calls=judge.calls - initial_calls,
                           seconds=elapsed, active=len(state.active), per_group_calls=lengths.tolist(),
                           next_active=sorted(state.active), answer=state.answer.copy()))
        print(f'{method} {kind} round={len(traces)} calls={judge.calls-initial_calls} '
              f'active={len(state.active)} seconds={elapsed:.1f}', flush=True)
        if state.done:
            break
    torch.cuda.synchronize()
    result = dict(kind=kind, method=method, calls=judge.calls - initial_calls,
                  seconds=time.perf_counter() - started,
                  llm_seconds=judge.seconds - initial_llm_seconds, answer=state.answer,
                  per_group_calls=[len(y) for y in observed], rounds=traces)
    return result, observed


def validate(result, observed, full, permutations, margin, threshold, topk):
    means = np.array([full[group].mean() for group in permutations])
    for group, ys in enumerate(observed):
        if not np.array_equal(ys, full[permutations[group][:len(ys)]]):
            raise ValueError('A live judgment differs from the full pass using the same request blocks.')
    selected = set(result['answer'])
    if result['kind'] == 'having':
        wrong = any((mean >= threshold + margin and group not in selected) or
                    (mean <= threshold - margin and group in selected)
                    for group, mean in enumerate(means))
    elif result['kind'] == 'order':
        position = {group: rank for rank, group in enumerate(result['answer'])}
        wrong = any(means[i] >= means[j] + margin and position[i] > position[j]
                    for i in range(len(means)) for j in range(len(means)))
    else:
        wrong = len(selected) != topk or min(means[list(selected)]) < np.sort(means)[-topk] - margin
    result['wrong'] = bool(wrong)
    result['live_labels_match_exact'] = True
    result['full_means'] = means.tolist()
    if wrong:
        print('INCORRECT_ANSWER', result['kind'], result['method'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-groups', type=int, default=16)
    parser.add_argument('--cap-per-group', type=int, default=2000)
    parser.add_argument('--token-budget', type=int, default=8000)
    args = parser.parse_args()
    if os.name == 'nt':
        # Process-scoped inhibition is released by Windows when this process exits.
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError('Cannot keep the host awake during the benchmark.')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261109)
    torch.set_num_threads(4)
    texts, groups, names, train = load_workload(args.data, args.max_groups, args.cap_per_group)
    rng = np.random.default_rng(2027)
    permutations = [group[rng.permutation(len(group))] for group in groups]
    workload_hash = hashlib.sha256(np.concatenate(permutations).tobytes()).hexdigest()
    config = dict(name=args.name, model=str(args.model), N=sum(map(len, groups)), K=len(groups),
                  products=names, group_sizes=list(map(len, groups)), workload_sha256=workload_hash,
                  alpha=0.05, margin=0.01, threshold=0.20, topk=5, batch=BATCH,
                  token_budget=args.token_budget, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), proxy_training_calls=len(train))
    config_path = args.out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output belongs to a different experiment configuration.')
    save_json(config_path, config)
    np.savez(args.out / 'workload.npz', **{f'group_{i}': group for i, group in enumerate(permutations)})
    judge = LocalJudge(args.model, texts, args.token_budget)
    judge(permutations[0][:BATCH])
    proxy, beta, info = fit_proxy(judge, texts, train, groups, args.out)
    print('PROXY', json.dumps(info), flush=True)
    exact_path = args.out / 'exact.json'
    if exact_path.exists() and (args.out / 'exact_labels.npz').exists():
        exact_result = json.loads(exact_path.read_text())
        full = np.load(args.out / 'exact_labels.npz')['y']
    else:
        full = np.full(len(texts), np.nan)
        initial_calls, initial_llm_seconds = judge.calls, judge.seconds
        started = time.perf_counter()
        for group, permutation in enumerate(permutations):
            for start in range(0, len(permutation), BATCH):
                indices = permutation[start:start + BATCH]
                full[indices] = judge(indices)
            print('EXACT', group + 1, len(groups), 'calls', judge.calls - initial_calls,
                  'seconds', round(time.perf_counter() - started, 1), flush=True)
        torch.cuda.synchronize()
        exact_result = dict(calls=judge.calls - initial_calls, seconds=time.perf_counter() - started,
                            llm_seconds=judge.seconds - initial_llm_seconds)
        np.savez(args.out / 'exact_labels.npz', y=full)
        save_json(exact_path, exact_result)
    connection = duckdb.connect()
    connection.execute('CREATE TABLE judged(product INTEGER, y DOUBLE)')
    connection.executemany('INSERT INTO judged VALUES (?, ?)',
                          [(g, float(full[i])) for g, group in enumerate(groups) for i in group])
    references = {}
    sql = dict(having='SELECT product FROM judged GROUP BY product HAVING AVG(y)>0.20 ORDER BY product',
               order='SELECT product FROM judged GROUP BY product ORDER BY AVG(y) DESC, product',
               topk='SELECT product FROM judged GROUP BY product ORDER BY AVG(y) DESC, product LIMIT 5')
    for kind, statement in sql.items():
        started = time.perf_counter()
        references[kind] = dict(answer=[row[0] for row in connection.execute(statement).fetchall()],
                                sql_seconds=time.perf_counter() - started, sql=statement)
    connection.close()
    save_json(args.out / 'references.json', references)
    for kind in ['having', 'order', 'topk']:
        for method in ['BET-P', 'WSR']:
            path = args.out / f'{method}_{kind}.json'
            judgments_path = args.out / f'{method}_{kind}_judgments.npz'
            if path.exists() and judgments_path.exists():
                continue
            result, observed = live_query(judge, permutations, proxy, beta, kind, method,
                                           0.05, 0.01, 0.20, 5)
            validate(result, observed, full, permutations, 0.01, 0.20, 5)
            result['fraction'] = result['calls'] / config['N']
            result['exact_seconds'] = exact_result['seconds'] + references[kind]['sql_seconds']
            result['speedup_vs_exact'] = result['exact_seconds'] / result['seconds']
            np.savez(judgments_path,
                     **{f'group_{i}': np.array(ys) for i, ys in enumerate(observed)})
            save_json(path, result)
            print('COMPLETED', kind, method, result['calls'], result['seconds'], flush=True)
    print('BENCHMARK_COMPLETE', args.name, flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
