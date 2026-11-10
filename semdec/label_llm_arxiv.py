"""Zero-shot LLM judgments (the expensive oracle) for paper abstracts (ArXiv-LLM).
Run with a CUDA torch + transformers env:  python -I label_llm_arxiv.py <model_dir> <texts.jsonl> <out.npz> [limit]
Same model, decoding, and batching as label_llm.py; only the system prompt and the questions differ.
The output file is rewritten after every question, so a finished question survives a later failure.
"""
import sys, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8')
MODEL, INP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = (int(sys.argv[4]) or None) if len(sys.argv) > 4 else None
ONLY = sys.argv[5].split(',') if len(sys.argv) > 5 else None          # optional subset of questions
START = int(sys.argv[6]) if len(sys.argv) > 6 else 0                  # optional chunk: first line and line count
COUNT = int(sys.argv[7]) if len(sys.argv) > 7 else None
MAXCH = 2000
SYSTEM = 'You are a precise annotator of scientific paper abstracts.'
QUESTIONS = {
    'sota': ('Does the abstract claim that the proposed approach outperforms existing methods or achieves '
             'state-of-the-art results? Answer with only Yes or No.', ['Yes', 'No']),
    'emp': ('Does the abstract report results from experiments or an empirical evaluation? Answer with only Yes or No.', ['Yes', 'No']),
}
if ONLY:
    QUESTIONS = {k: v for k, v in QUESTIONS.items() if k in ONLY}

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
ids, texts = [], []
for li, line in enumerate(open(INP, encoding='utf-8')):
    if li < START:
        continue
    if COUNT is not None and li >= START + COUNT:
        break
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
                                        {'role': 'user', 'content': f'Abstract:\n{tx}\n\nQuestion: {q}'}],
                                       tokenize=False, add_generation_prompt=True) for tx in texts]
    lens = np.array([len(p) for p in prompts])
    order = np.argsort(lens)
    probs = np.zeros((n, len(words)), dtype=np.float32)
    i = 0; nb = 0; t0 = time.time(); TOKBUDGET = 12000
    while i < n:
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
        probs[idx] = torch.softmax(logits[:, aid], dim=-1).cpu().numpy()
        del enc, logits
        nb += 1
        if nb % 100 == 0:
            torch.cuda.empty_cache()            # cached blocks also hold Windows commit; release them regularly
        i = j
        if (i // 5000) != ((i - len(idx)) // 5000):
            print(f'{qn}: {i}/{n} ({i/(time.time()-t0):.1f}/s)', flush=True)
    res[qn] = probs
    np.savez(OUT, ids=np.array(ids), **{f'p_{k}': v for k, v in res.items()})
    print(f'{qn} done in {time.time()-t0:.0f}s; saved', flush=True)
print('saved', OUT, f'total {time.time()-t_start:.0f}s', flush=True)
