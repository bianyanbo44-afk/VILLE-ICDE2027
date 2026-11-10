"""Proxies of different quality for AMZ-Stars: retrain on 100/300/1000 of the 5,000 held-out labeled reviews.
Run: python -I proxy_sweep.py <amazon_dir> <sem1_amazon.npz> <out.npz> cat1 cat2 ...  (same category order as sem1_prep)
"""
import sys, os, gzip, json, time
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

sys.stdout.reconfigure(encoding='utf-8')
D, NPZ, OUT, CATS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
SIZES = [100, 300, 1000]
d = np.load(NPZ)
train_mask, rating = d['train'], d['rating'].astype(np.float64)
N = len(rating)


def rows():
    for cat in CATS:
        with gzip.open(os.path.join(D, cat + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                yield (r.get('title') or '') + ' . ' + (r.get('text') or '')


t0 = time.time()
tr_idx = np.flatnonzero(train_mask)
txt = [None] * len(tr_idx); pos = {g: k for k, g in enumerate(tr_idx)}
for gi, tx in enumerate(rows()):
    if gi in pos:
        txt[pos[gi]] = tx
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm=None)
rng = np.random.default_rng(31)
perm = rng.permutation(len(tr_idx))
models = {}
BETAS = {}
for n in SIZES:
    sub = perm[:n]
    tf = TfidfTransformer(sublinear_tf=True)
    X = tf.fit_transform(hv.transform([txt[k] for k in sub]))
    y = rating[tr_idx[sub]]
    yb = (y <= 2).astype(int)
    pcv = cross_val_predict(LogisticRegression(C=4.0, max_iter=2000), X, yb, cv=5, method='predict_proba')[:, 1]
    BETAS[f'beta_neg_{n}'] = float(np.clip(np.cov(yb, pcv)[0, 1] / max(np.var(pcv, ddof=1), 1e-12), 0, 1))
    m_neg = LogisticRegression(C=4.0, max_iter=2000).fit(X, yb)
    m_rat = Ridge(alpha=1.0).fit(X, y)
    models[n] = (tf, m_neg, m_rat)
print('trained', f'({time.time()-t0:.0f}s)', flush=True)
P = {f'p_neg_{n}': np.zeros(N, dtype=np.float32) for n in SIZES}
P.update({f'p_rat_{n}': np.zeros(N, dtype=np.float32) for n in SIZES})
buf, start = [], 0


def flush(buf, start):
    H = hv.transform(buf)
    for n, (tf, m_neg, m_rat) in models.items():
        X = tf.transform(H)
        P[f'p_neg_{n}'][start:start + len(buf)] = m_neg.predict_proba(X)[:, 1]
        P[f'p_rat_{n}'][start:start + len(buf)] = np.clip(m_rat.predict(X), 1, 5)


for gi, tx in enumerate(rows()):
    buf.append(tx)
    if len(buf) == 100_000:
        flush(buf, start); start += len(buf); buf = []
if buf:
    flush(buf, start)
ev = ~train_mask
for n in SIZES:
    print(n, 'AUC neg', round(roc_auc_score(rating[ev] <= 2, P[f'p_neg_{n}'][ev]), 3),
          'corr rat', round(np.corrcoef(rating[ev], P[f'p_rat_{n}'][ev])[0, 1], 3), flush=True)
print(BETAS)
np.savez(OUT, **P, **{k: np.array(v) for k, v in BETAS.items()})
print('saved', f'({time.time()-t0:.0f}s)')
