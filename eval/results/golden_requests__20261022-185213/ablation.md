| config | hit@1 | hit@5 | mrr@10 | recall@5 | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness | refusal_rate | context_tokens | p50_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.367 | 0.733 | 0.527 | 0.683 | 0.885 | 0.686 | 0.717 | 0.453 | 0.318 | 1.0 | 388 | 1.6 |
| headers | 0.467 | 0.933 | 0.641 | 0.867 | 0.936 | 0.763 | 0.725 | 0.536 | 0.508 | 1.0 | 647 | 1.72 |
| small_to_big | 0.467 | 0.933 | 0.641 | 0.867 | 0.925 | 0.793 | 0.735 | 0.744 | 0.501 | 1.0 | 1454 | 2.09 |

## By chunk type

| config | hit@1 | noncode@1 | noncode_share@5 | code_hit@1 | code_mrr@10 |
|---|---|---|---|---|---|
| baseline | 0.367 | 0.533 | 0.627 | 0.733 | 0.819 |
| headers | 0.467 | 0.467 | 0.453 | 0.767 | 0.878 |
| small_to_big | 0.467 | 0.467 | 0.453 | 0.767 | 0.878 |

## 95% bootstrap intervals (paired differences vs headers)

| config | hit@1 | Δhit@1 vs headers | mrr@10 | Δmrr@10 vs headers |
|---|---|---|---|---|
| baseline | 0.367 [0.20, 0.53] | -0.100 [-0.27, +0.07] | 0.527 [0.39, 0.66] | -0.114 [-0.25, +0.02] |
| headers | 0.467 [0.30, 0.63] |  | 0.641 [0.51, 0.77] |  |
| small_to_big | 0.467 [0.30, 0.63] | +0.000 [+0.00, +0.00] | 0.641 [0.51, 0.77] | +0.000 [+0.00, +0.00] |

| config | factual_correctness | Δfactual_correctness vs headers | faithfulness | Δfaithfulness vs headers | context_recall | Δcontext_recall vs headers |
|---|---|---|---|---|---|---|
| baseline | 0.318 [0.23, 0.41] | -0.189 [-0.30, -0.09] | 0.885 [0.82, 0.94] | -0.052 [-0.12, +0.02] | 0.453 [0.31, 0.60] | -0.083 [-0.24, +0.07] |
| headers | 0.508 [0.44, 0.58] |  | 0.936 [0.89, 0.97] |  | 0.536 [0.38, 0.68] |  |
| small_to_big | 0.501 [0.42, 0.57] | -0.007 [-0.07, +0.06] | 0.925 [0.87, 0.97] | -0.012 [-0.08, +0.06] | 0.744 [0.61, 0.87] | +0.208 [+0.07, +0.36] |
