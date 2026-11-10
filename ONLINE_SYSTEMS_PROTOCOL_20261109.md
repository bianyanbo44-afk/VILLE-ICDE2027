# Live decision-query experiment

Frozen before the new model is evaluated, 2026-11-09.

## Question

Do saved LLM calls translate into measured query latency for HAVING, ORDER BY and Top-5 across two independent model families?

## Workload and methods

- Use the existing AMZ-LLM Video_Games evaluation split. Select its 16 largest products by count, with a uniform seed-20261109 sample of at most 2,000 reviews per product. Selection uses metadata alone. Preserve the existing disjoint 3,000-review proxy-training split.
- Evaluate the existing Defect prompt, with the same 1,200-character truncation and constrained Yes/No first-token decision.
- Use Qwen/Qwen2.5-1.5B-Instruct and HuggingFaceTB/SmolLM2-1.7B-Instruct. The latter is pinned to revision 31b70e2e869a7173562077fd711b654946d38674.
- HAVING threshold 0.20, indifference margin 0.01, alpha 0.05, Top-5 and batches of 100. These values are fixed before reading the additional model's judgments.
- Compare VILLE (BET-P), the existing Evergreen-style confidence sequence (WSR), and exact evaluation. Use the same seed-2027 group permutations for all methods. Charge every tuple actually sent to the model, including tuples acquired before the final Top-5 boundary is established.
- Use DuckDB to materialize the workload, group its row identifiers and compute the reference HAVING, ORDER BY and Top-5 answers after full evaluation. VILLE is the decision layer over these relation groups.

## Model and timing contract

- Fit a separate n-gram proxy to each model's judgments on the disjoint training split, reusing the main study's logistic regression and five-fold beta calibration. Report setup separately.
- Load and warm up the model before timing queries. Keep the model resident. Include live model execution, prompt processing and certificate updates in query latency. Precomputed group sizes and proxy scores follow the main paper's amortized-proxy setting.
- Execute every method with fresh model calls. Disable reuse of judgments between methods. Within each group, fixed 100-tuple request blocks and identical internal batching make full and sampled runs use the same deterministic model function.
- Measure one complete exact pass per model. The same predicate's fully judged relation supplies reference answers for all three query forms. Report this measured pass explicitly as their shared exact-scan reference, with no throughput-based extrapolation.
- Save model/environment metadata, actual calls, measured seconds, per-round progress, selected row identifiers, reference labels and correctness checks. Compare every live judgment with the full pass, and fail if batching changes a judgment.
- Resume at completed method/query runs only. A failed run is rerun from the beginning and its partial time is excluded from reported completed-run latency.

## Interpretation

Keep this experiment separate from the existing 75-query replay study. Report all three query forms and both models. Center the latency result on query forms that certify early, and retain the full table so the relation between sample fraction and runtime is inspectable. Empirical agreement supplements the formal guarantee.

## Execution record

The first Qwen timing pass was interrupted by host standby. Its full scan and log are retained with `interrupted_sleep` names and excluded from the reported measurements. The rerun inhibits automatic idle sleep for the process lifetime and checks model calls against Windows active interrupt time to detect another standby interval. The completed proxy calibration preceded the interruption and is reused as setup data. Inference uses a padded-token budget of 8,000 for Qwen and 6,000 for SmolLM2 to fit the 8 GB device, held constant across methods within each model. Both retain 100-tuple request blocks.
