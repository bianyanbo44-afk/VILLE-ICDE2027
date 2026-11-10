"""Record the arXiv identifiers of the ArXiv-LLM papers, in the order of sem3_prep.py's texts.jsonl.
Run: python -I sem3_ids.py <arxiv-abstracts.jsonl.gz> <data/sem3>
Applies exactly the filters of sem3_prep.py and checks every reconstructed text against texts.jsonl.
"""
import sys, os, gzip, json, re
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
INP, OUT = sys.argv[1], sys.argv[2]
pat = re.compile(r'^(\d{2})(\d{2})\.\d{4,5}$')
ws = re.compile(r'\s+')
ids = []
ref = open(os.path.join(OUT, 'texts.jsonl'), encoding='utf-8')
with gzip.open(INP, 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        m = pat.match(str(r.get('id', '')))
        if not m or not (2010 <= 2000 + int(m.group(1)) <= 2021):
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
        row = json.loads(ref.readline())
        assert row['id'] == len(ids) and row['text'] == title + '. ' + abstract, len(ids)
        ids.append(str(r['id']))
assert ref.readline() == ''
np.save(os.path.join(OUT, 'arxiv_ids.npy'), np.array(ids))
print('ids', len(ids), ids[:3], ids[-3:])
