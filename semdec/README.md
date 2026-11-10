# VILLE: certified decision queries over LLM-evaluated data

Code for the experiments in "VILLE: Certified Decision Queries over LLM-Evaluated Data with Proxy-Rectified E-Processes".

## Contents
| File | Purpose |
|---|---|
| `core.py` | Without-replacement betting tests (one-sample and paired), Evergreen-style WSR confidence sequence, IFOCUS-style Hoeffding–Serfling, BlazeIt-style empirical-Bernstein stopping, CLT |
| `query_schedule.py`, `replay_saved.py` | Shared actual-prefix scheduler and deterministic cached replay |
| `run.py` | Query runner: HAVING / ORDER BY / Top-k compilation, per-group stopping, cost, run-to-run spread, and error accounting (decisions and returned answers) |
| `test_validity.py` | Type-I error check of the tests on finite populations on the null boundary |
| `sem1_prep.py`, `exp_d1.py` | AMZ-Stars: star-derived judgments, n-gram proxies, main experiment |
| `sem2_prep.py`, `label_llm.py`, `sem2_run.py` | AMZ-LLM: product selection, zero-shot LLM judgments (Qwen2.5-1.5B-Instruct), proxies and queries |
| `sem3_prep.py`, `sem3_ids.py`, `label_llm_arxiv.py`, `sem3_run.py` | ArXiv-LLM: computer-science abstracts 2010–2021, chunked LLM judgments, proxies and queries |
| `online_demo.py` | End-to-end run of VILLE with the LLM in the loop (no stored judgment is used for decisions) |
| `online_benchmark.py` | DuckDB-backed live HAVING, ORDER BY and Top-5 with Qwen2.5 and SmolLM2, actual-call accounting and measured exact scans |
| `proxy_only.py` | Correctness of answers computed from proxy scores alone |
| `proxy_sweep.py`, `exp_ablation.py` | Sensitivity studies (proxy quality, misleading proxies, margin, alpha, number of groups) |
| `exp_hedged_nocap.py` | Evergreen-style baseline with the hedged-capital betting confidence sequence |
| `gates.py`, `anatomy_data.py` | Pre-registered gate evaluation; data for the single-decision figure |
| `../paper_scripts/` | Figures, tables, and every number quoted in the paper, from the result JSON files |

## Data
- **AMZ-Stars, AMZ-LLM**: Amazon Reviews 2023 (McAuley Lab), categories listed in the paper, from the public dataset site.
- **ArXiv-LLM**: the public arXiv metadata snapshot `gfissore/arxiv-abstracts-2021` (CC0) on Hugging Face. `data/arxiv_llm/arxiv_ids.npy` lists the arXiv identifier of every row, in order.
- `data/amz_llm/` and `data/arxiv_llm/` hold the LLM judgments (`llm_labels.npz`), group metadata (`meta.npz`), and proxy scores (`proxies.npz`) used in the paper.

## Reproduce
```
# AMZ-Stars
python sem1_prep.py <amazon_dir> sem1_amazon.npz Gift_Cards Magazine_Subscriptions Digital_Music Health_and_Personal_Care All_Beauty Handmade_Products Appliances Amazon_Fashion
python exp_d1.py sem1_amazon.npz d1_main.json 30 cat year prod-Amazon_Fashion prod-Appliances   # one call: seeds follow this order
# AMZ-LLM
python sem2_prep.py <amazon_dir> sem2/
python label_llm.py <Qwen2.5-1.5B-Instruct dir> sem2/texts.jsonl sem2/llm_labels.npz
python sem2_run.py sem2/ d2_wor.json 30
# ArXiv-LLM (labeling in chunks of 20,000 abstracts)
python sem3_prep.py arxiv-abstracts.jsonl.gz sem3/
python label_llm_arxiv.py <Qwen2.5-1.5B-Instruct dir> sem3/texts.jsonl sem3/chunks/sota_0.npz 0 sota 0 20000   # repeat for every start and for emp
python sem3_run.py sem3/ d3_arxiv.json 30
# live run, proxy-only answers, figures and tables
python online_demo.py <Qwen2.5-1.5B-Instruct dir> sem2/ online_demo.json 0 defect
python proxy_only.py sem1_amazon.npz sem2/ sem3/ proxy_only.json
python ../paper_scripts/make_figs.py <results_dir> <out_dir>
python ../paper_scripts/make_tables.py <results_dir> <out_dir>
python ../paper_scripts/make_macros.py <results_dir> numbers_auto.tex   # every aggregate number quoted in the text
```
For the supplied corrected cached results, use `replay_saved.py` and the pinned environment documented in the package root. It preserves the original per-query seeds. The commands above describe the source-data and model-judgment preparation route.

## Live systems measurements

The supplementary live workload uses 16 Video_Games products, at most 2,000 evaluation reviews per product,
and the existing disjoint 3,000-review proxy-training split. The protocol is
`../probes/ONLINE_SYSTEMS_PROTOCOL_20261109.md` in the source project and
`../ONLINE_SYSTEMS_PROTOCOL_20261109.md` in the artifact.
Use a CUDA environment with PyTorch, Transformers, NumPy, SciPy, scikit-learn and DuckDB.
The raw review texts are reconstructed by `sem2_prep.py` using the public Amazon source.

```
python -I online_benchmark.py --model <Qwen2.5-1.5B-Instruct dir> --name qwen --data <sem2 dir> --out <results dir>/live_qwen_20261109
python -I online_benchmark.py --model <SmolLM2-1.7B-Instruct dir> --name smol --data <sem2 dir> --out <results dir>/live_smol_20261109
python -I ../paper_scripts/make_online_table.py <results dir> <paper dir>
```

SmolLM2 is pinned to revision `31b70e2e869a7173562077fd711b654946d38674` of
`HuggingFaceTB/SmolLM2-1.7B-Instruct`. Each completed method/query writes its own JSON and judgments.
Use a fresh results directory for a new timing run. Reusing a directory resumes completed configurations.
The full pass and all sampled methods issue fresh LLM calls. An exact pass supplies a measured reference
for all three queries over the same attribute. Queries use a resident model, and proxy preparation is
recorded separately from the warm-query measurements. The live result directories retain workload identifiers, calibration
cost, full labels, per-round progress and checks against the exact answers.
