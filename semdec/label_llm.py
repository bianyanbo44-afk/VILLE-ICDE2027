"""Zero-shot LLM judgments (the expensive oracle) for review texts.
Run with a CUDA torch + transformers env:  python -I label_llm.py <model_dir> <texts.jsonl> <out.npz> [limit]
Each line of texts.jsonl: {"id": int, "text": str}.  Outputs per question the probability of the first answer
token(s) and the argmax answer.
"""
import sys, json, time, math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')
MODEL, INP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
MAXCH = 1200
SYSTEM = 'You are a precise annotator of customer product reviews.'
QUESTIONS = {
    'defect': ('Does the reviewer report that the product was defective, broken, damaged, or stopped working? '
               'Answer with only Yes or No.', ['Yes', 'No']),
    'value': ("Does the reviewer comment on the product's price or value for money? "
              'Answer with only Yes or No.', ['Yes', 'No']),
    'sent': ('On a scale from 1 (very negative) to 5 (very positive), how positive is this review overall? '
             'Answer with a single digit.', ['1', '2', '3', '4', '5']),
}

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
ids = []
texts = []
for line in open(INP, encoding='utf-8'):
    r = json.loads(line); ids.append(r['id']); texts.append(r['text'][:MAXCH])
    if LIMIT and len(texts) >= LIMIT:
        break
n = len(texts)
print('texts', n, flush=True)


def ans_ids(words):
    out = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        assert len(t) == 1, (w, t)
        out.append(t[0])
    return out


res = {}
t_start = time.time()
for qn, (q, words) in QUESTIONS.items():
    aid = torch.tensor(ans_ids(words), device='cuda')
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                                        {'role': 'user', 'content': f'Review:\n{tx}\n\nQuestion: {q}'}],
                                       tokenize=False, add_generation_prompt=True) for tx in texts]
    lens = np.array([len(p) for p in prompts])
    order = np.argsort(lens)
    probs = np.zeros((n, len(words)), dtype=np.float32)
    i = 0; t0 = time.time(); TOKBUDGET = 12000
    while i < n:
        # adaptive batch: keep (batch * max_len_tokens) under the budget
        j = i; maxlen = 0
        while j < n:
            l_est = lens[order[j]] // 3 + 8
            if (j - i + 1) * max(maxlen, l_est) > TOKBUDGET and j > i:
                break
            maxlen = max(maxlen, l_est); j += 1
        idx = order[i:j]
        enc = tok([prompts[k] for k in idx], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        pr = torch.softmax(logits[:, aid], dim=-1).cpu().numpy()
        probs[idx] = pr
        i = j
        if (i // 2000) != ((i - len(idx)) // 2000):
            print(f'{qn}: {i}/{n} ({i/(time.time()-t0):.1f}/s)', flush=True)
    res[qn] = probs
    print(f'{qn} done in {time.time()-t0:.0f}s', flush=True)
np.savez(OUT, ids=np.array(ids), **{f'p_{k}': v for k, v in res.items()})
print('saved', OUT, f'total {time.time()-t_start:.0f}s', flush=True)
