"""SEM2 data prep: pick query products from two unused categories, write texts for LLM labeling.
Run: python -I sem2_prep.py <amazon_dir> <out_dir>
"""
import sys, os, gzip, json, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
D, OUT = sys.argv[1], sys.argv[2]
CATS = ['Video_Games', 'Musical_Instruments']
TARGET, LOW, HIGH, NTRAIN_PER_CAT = 60_000, 1000, 4000, 1500
rng = np.random.default_rng(23)
os.makedirs(OUT, exist_ok=True)


def rows(cat):
    with gzip.open(os.path.join(D, cat + '.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


meta = []          # (cat, asin, year, rating)
texts_out = open(os.path.join(OUT, 'texts.jsonl'), 'w', encoding='utf-8')
gid = 0
for ci, cat in enumerate(CATS):
    t0 = time.time()
    cnt = {}
    for r in rows(cat):
        a = r.get('parent_asin') or r.get('asin')
        cnt[a] = cnt.get(a, 0) + 1
    cand = sorted([(c, a) for a, c in cnt.items() if LOW <= c <= HIGH], reverse=True)
    chosen, tot = set(), 0
    for c, a in cand:
        if tot >= TARGET:
            break
        chosen.add(a); tot += c
    n_other = sum(cnt.values()) - tot
    p_train = min(1.0, 3 * NTRAIN_PER_CAT / max(n_other, 1))
    tr_buf = []
    for r in rows(cat):
        a = r.get('parent_asin') or r.get('asin')
        text = (r.get('title') or '') + '. ' + (r.get('text') or '')
        if a in chosen:
            y = time.gmtime(r.get('timestamp', 0) / 1000).tm_year
            meta.append((ci, a, y, r.get('rating', 0), 0))
            texts_out.write(json.dumps({'id': gid, 'text': text}) + '\n'); gid += 1
        elif rng.random() < p_train:
            tr_buf.append(text)
    pick = rng.choice(len(tr_buf), min(NTRAIN_PER_CAT, len(tr_buf)), replace=False)
    for k in pick:
        meta.append((ci, '__train__', 0, 0, 1))
        texts_out.write(json.dumps({'id': gid, 'text': tr_buf[k]}) + '\n'); gid += 1
    print(f'{cat}: products={len(chosen)} query_rows={tot} train={len(pick)} ({time.time()-t0:.0f}s)', flush=True)
texts_out.close()
cat_a = np.array([m[0] for m in meta], dtype=np.int8)
asins = sorted({m[1] for m in meta})
amap = {a: i for i, a in enumerate(asins)}
np.savez(os.path.join(OUT, 'meta.npz'), cat=cat_a, asin=np.array([amap[m[1]] for m in meta], dtype=np.int32),
         year=np.array([m[2] for m in meta], dtype=np.int16), rating=np.array([m[3] for m in meta], dtype=np.float32),
         train=np.array([m[4] for m in meta], dtype=bool), cats=np.array(CATS))
print('rows', len(meta))
