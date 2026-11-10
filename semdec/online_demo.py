"""End-to-end run of VILLE with a live local LLM on one AMZ-LLM HAVING query (no precomputed judgments are used
for decisions; they are only used afterwards to check correctness).
Run (CUDA env): python -I online_demo.py <model_dir> <data/sem2> <out.json> [category_index] [predicate]
"""
import sys, os, json, time, math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import wor_bet_first, wor_bet_upper, looks_for

sys.stdout.reconfigure(encoding='utf-8')
MODEL, D, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CAT = int(sys.argv[4]) if len(sys.argv) > 4 else 0
PRED = sys.argv[5] if len(sys.argv) > 5 else 'defect'
SYSTEM = 'You are a precise annotator of customer product reviews.'
Q = {'defect': 'Does the reviewer report that the product was defective, broken, damaged, or stopped working? Answer with only Yes or No.',
     'value': "Does the reviewer comment on the product's price or value for money? Answer with only Yes or No."}[PRED]
ALPHA, BATCH, MAXCH = 0.05, 100, 1200

meta = np.load(os.path.join(D, 'meta.npz')); prox = np.load(os.path.join(D, 'proxies.npz'))
labels = np.load(os.path.join(D, 'llm_labels.npz'))
texts = [json.loads(l)['text'][:MAXCH] for l in open(os.path.join(D, 'texts.jsonl'), encoding='utf-8')]
q_idx = np.flatnonzero((~meta['train']) & (meta['cat'] == CAT))
asin = meta['asin']
a_, cnt = np.unique(asin[q_idx], return_counts=True)
groups = [q_idx[asin[q_idx] == x] for x in a_[np.argsort(-cnt)][:30]]
K = len(groups)
P = prox[f'p_{PRED}']; beta = float(prox[f'beta_{PRED}'])
exact = (labels[f'p_{PRED}'][:, 0] > 0.5).astype(float)              # used only for c and for the final check
c = float(np.concatenate([exact[g] for g in groups]).mean()); r = 0.05 * c
a = ALPHA / (2 * K)

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
yes_no = torch.tensor([tok.encode('Yes', add_special_tokens=False)[0], tok.encode('No', add_special_tokens=False)[0]], device='cuda')


LLM_SECONDS = [0.0]
TOKBUDGET = 12000                      # same adaptive batching as the offline labeling run


def judge(idx):
    t_start = time.time()
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Review:\n{texts[i]}\n\nQuestion: {Q}'}],
                                       tokenize=False, add_generation_prompt=True) for i in idx]
    out = np.zeros(len(idx))
    lens = np.array([len(p) for p in prompts]); order = np.argsort(lens)
    i, n = 0, len(idx)
    while i < n:
        j, maxlen = i, 0
        while j < n:
            l_est = lens[order[j]] // 3 + 8
            if (j - i + 1) * max(maxlen, l_est) > TOKBUDGET and j > i:
                break
            maxlen = max(maxlen, l_est); j += 1
        sub = order[i:j]
        enc = tok([prompts[k] for k in sub], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out[sub] = (torch.softmax(lg[:, yes_no], -1)[:, 0] > 0.5).float().cpu().numpy()
        i = j
    LLM_SECONDS[0] += time.time() - t_start
    return out


rng = np.random.default_rng(2027)
perm = [g[rng.permutation(len(g))] for g in groups]
pbar = [float(P[g].mean()) for g in groups]
rngz = [(0 - beta * (1 - pb), 1 - beta * (0 - pb)) for pb in pbar]
judged = [[] for _ in range(K)]
decision = [None] * K
calls = 0; t0 = time.time(); rounds = 0
while any(d is None for d in decision):
    active = [g for g in range(K) if decision[g] is None]
    batch = []
    for g in active:
        n0 = len(judged[g]); batch += [(g, i) for i in perm[g][n0:n0 + BATCH]]
    ys = judge([i for _, i in batch]); calls += len(batch); rounds += 1
    for (g, i), yv in zip(batch, ys):
        judged[g].append((i, yv))
    for g in active:
        idx = np.array([i for i, _ in judged[g]]); y = np.array([v for _, v in judged[g]])
        z = y - beta * (P[idx] - pbar[g]); N = len(groups[g]); lk = looks_for(len(z))
        lo, hi = rngz[g]
        if len(z) >= N:
            decision[g] = ('exact', 'above' if y.mean() > c else 'below', len(z)); continue
        if wor_bet_first(z, N, c, lo, hi, a, lk) >= 0:
            decision[g] = ('certified', 'above', len(z))
        elif wor_bet_upper(z, N, c, lo, hi, a, lk) >= 0:
            decision[g] = ('certified', 'below', len(z))
        elif wor_bet_upper(z, N, c + r, lo, hi, a, lk) >= 0 and wor_bet_first(z, N, c - r, lo, hi, a, lk) >= 0:
            decision[g] = ('certified', 'within r', len(z))
    print(f'round {rounds}: calls={calls} active={sum(d is None for d in decision)} elapsed={time.time()-t0:.0f}s', flush=True)
elapsed = time.time() - t0
Ntot = sum(len(g) for g in groups)
mus = [float(exact[g].mean()) for g in groups]
wrong = [g for g in range(K) if abs(mus[g] - c) >= r and ((decision[g][1] == 'above' and mus[g] < c) or (decision[g][1] == 'below' and mus[g] > c) or decision[g][1] == 'within r')]
res = dict(category=str(meta['cats'][CAT]), predicate=PRED, K=K, N=Ntot, c=c, r=r, calls=calls, fraction=calls / Ntot, rounds=rounds,
           seconds=elapsed, llm_seconds=LLM_SECONDS[0], calls_per_second=calls / elapsed, est_full_seconds=Ntot / (calls / elapsed), wrong=len(wrong),
           decisions=[dict(group=g, kind=decision[g][0], side=decision[g][1], judged=decision[g][2], N=len(groups[g]), mu=mus[g]) for g in range(K)])
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'decisions'}, indent=1))
