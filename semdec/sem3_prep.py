"""SEM3 data prep (ArXiv-LLM): computer-science abstracts 2010-2021 for LLM labeling.
Run: python -I sem3_prep.py <arxiv-abstracts.jsonl.gz> <out_dir>
Keeps papers whose primary category is cs.* and whose identifier is new-style (YYMM.NNNNN), so that the
submission year is read from the identifier.  3,000 random papers form the proxy training set and are
excluded from every query group.
"""
import sys, os, gzip, json, re, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
INP, OUT = sys.argv[1], sys.argv[2]
Y0, Y1, NTRAIN = 2010, 2021, 3000
os.makedirs(OUT, exist_ok=True)
pat = re.compile(r'^(\d{2})(\d{2})\.\d{4,5}$')
ws = re.compile(r'\s+')
t0 = time.time()
years, prims, texts = [], [], []
with gzip.open(INP, 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        m = pat.match(str(r.get('id', '')))
        if not m:
            continue
        y = 2000 + int(m.group(1))
        if not (Y0 <= y <= Y1):
            continue
        cats = r.get('categories')
        cats = cats if isinstance(cats, list) else [str(cats)]
        toks = ' '.join(cats).replace('[', ' ').replace(']', ' ').replace("'", ' ').split()
        if not toks or not toks[0].startswith('cs.'):
            continue
        title = ws.sub(' ', str(r.get('title', ''))).strip()
        abstract = ws.sub(' ', str(r.get('abstract', ''))).strip()
        if len(abstract) < 100:
            continue
        years.append(y); prims.append(toks[0]); texts.append(title + '. ' + abstract)
n = len(texts)
rng = np.random.default_rng(2027)
train = np.zeros(n, dtype=bool); train[rng.choice(n, NTRAIN, replace=False)] = True
cats = sorted(set(prims)); cmap = {c: i for i, c in enumerate(cats)}
with open(os.path.join(OUT, 'texts.jsonl'), 'w', encoding='utf-8') as f:
    for i, tx in enumerate(texts):
        f.write(json.dumps({'id': i, 'text': tx}) + '\n')
np.savez(os.path.join(OUT, 'meta.npz'), year=np.array(years, dtype=np.int16),
         cat=np.array([cmap[c] for c in prims], dtype=np.int16), cats=np.array(cats), train=train)
print(f'papers {n} train {train.sum()} categories {len(cats)} ({time.time()-t0:.0f}s)')
