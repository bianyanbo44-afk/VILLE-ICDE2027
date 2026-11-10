"""SEM1 data prep: Amazon Reviews 2023 -> per-row arrays + cheap proxy scores.
Run: python -I sem1_prep.py <amazon_dir> <out_npz> cat1 cat2 ...
"""
import sys, os, gzip, json, time
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
D, OUT, CATS = sys.argv[1], sys.argv[2], sys.argv[3:]
NTRAIN = 5000
rng = np.random.default_rng(13)


def rows(cat):
    with gzip.open(os.path.join(D, cat + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


# pass 1: metadata
t0 = time.time()
cat_l, rat_l, year_l, asin_l = [], [], [], []
asin_ids = {}
for ci, cat in enumerate(CATS):
    n0 = len(rat_l)
    for r in rows(cat):
        cat_l.append(ci)
        rat_l.append(r.get('rating', 0))
        year_l.append(time.gmtime(r.get('timestamp', 0) / 1000).tm_year)
        a = r.get('parent_asin') or r.get('asin') or ''
        asin_l.append(asin_ids.setdefault(a, len(asin_ids)))
    print(f'{cat}: {len(rat_l)-n0} rows ({time.time()-t0:.0f}s)', flush=True)
cat_a = np.array(cat_l, dtype=np.int8); rat_a = np.array(rat_l, dtype=np.float32)
year_a = np.array(year_l, dtype=np.int16); asin_a = np.array(asin_l, dtype=np.int32)
del cat_l, rat_l, year_l, asin_l
N = len(rat_a)
train_idx = np.sort(rng.choice(N, NTRAIN, replace=False))
train_mask = np.zeros(N, dtype=bool); train_mask[train_idx] = True


def text_of(r):
    return (r.get('title') or '') + ' . ' + (r.get('text') or '')


# pass 2: training texts
txt = []
gi = 0
for cat in CATS:
    for r in rows(cat):
        if train_mask[gi]:
            txt.append(text_of(r))
        gi += 1
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm=None)
tf = TfidfTransformer(sublinear_tf=True)
Xtr = tf.fit_transform(hv.transform(txt))
ytr = rat_a[train_idx].astype(np.float64)
models, beta = {}, {}
for name, y, kind in (('neg', (ytr <= 2).astype(int), 'clf'), ('pos', (ytr == 5).astype(int), 'clf'), ('rat', ytr, 'reg')):
    if kind == 'clf':
        m = LogisticRegression(C=4.0, max_iter=2000)
        p_cv = cross_val_predict(m, Xtr, y, cv=5, method='predict_proba')[:, 1]
    else:
        m = Ridge(alpha=1.0)
        p_cv = np.clip(cross_val_predict(m, Xtr, y, cv=5), 1, 5)
    m.fit(Xtr, y)
    models[name] = m
    b = np.cov(y, p_cv)[0, 1] / max(np.var(p_cv, ddof=1), 1e-12)
    beta[name] = float(np.clip(b, 0, 1))
    print(f'proxy {name}: cv beta={beta[name]:.3f} cv corr={np.corrcoef(y, p_cv)[0,1]:.3f}', flush=True)

# pass 3: proxy scores for all rows
P = {k: np.zeros(N, dtype=np.float32) for k in models}
buf, gi0, gi = [], 0, 0


def flush(buf, start):
    X = tf.transform(hv.transform(buf))
    P['neg'][start:start + len(buf)] = models['neg'].predict_proba(X)[:, 1]
    P['pos'][start:start + len(buf)] = models['pos'].predict_proba(X)[:, 1]
    P['rat'][start:start + len(buf)] = np.clip(models['rat'].predict(X), 1, 5)


for cat in CATS:
    for r in rows(cat):
        buf.append(text_of(r)); gi += 1
        if len(buf) == 100_000:
            flush(buf, gi0); gi0 = gi; buf = []
    print(f'scored through {cat} ({time.time()-t0:.0f}s)', flush=True)
if buf:
    flush(buf, gi0)
ev = ~train_mask
print(f"population AUC neg={roc_auc_score(rat_a[ev] <= 2, P['neg'][ev]):.3f} pos={roc_auc_score(rat_a[ev] == 5, P['pos'][ev]):.3f} "
      f"rat corr={np.corrcoef(rat_a[ev], P['rat'][ev])[0,1]:.3f}", flush=True)
np.savez(OUT, cat=cat_a, rating=rat_a, year=year_a, asin=asin_a, train=train_mask,
         p_neg=P['neg'], p_pos=P['pos'], p_rat=P['rat'], cats=np.array(CATS), beta=json.dumps(beta))
print('saved', OUT, f'({time.time()-t0:.0f}s)')
