| config | hit@1 | hit@5 | mrr@10 | recall@5 | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness | refusal_rate | context_tokens | p50_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.35 | 0.725 | 0.518 | 0.688 | 0.893 | 0.772 | 0.769 | 0.379 | 0.348 | 0.75 | 371 | 1.76 |
| headers | 0.5 | 0.775 | 0.648 | 0.738 | 0.933 | 0.856 | 0.831 | 0.64 | 0.502 | 0.75 | 653 | 1.58 |
| small_to_big | 0.5 | 0.775 | 0.648 | 0.738 | 0.97 | 0.882 | 0.84 | 0.835 | 0.496 | 0.75 | 1565 | 1.94 |

## By chunk type

| config | hit@1 | noncode@1 | noncode_share@5 | code_hit@1 | code_mrr@10 |
|---|---|---|---|---|---|
| baseline | 0.35 | 0.525 | 0.58 | 0.75 | 0.827 |
| headers | 0.5 | 0.45 | 0.52 | 0.875 | 0.9 |
| small_to_big | 0.5 | 0.45 | 0.52 | 0.875 | 0.9 |

## 95% bootstrap intervals (paired differences vs headers)

| config | hit@1 | Δhit@1 vs headers | mrr@10 | Δmrr@10 vs headers |
|---|---|---|---|---|
| baseline | 0.350 [0.20, 0.50] | -0.150 [-0.30, +0.00] | 0.518 [0.40, 0.63] | -0.130 [-0.22, -0.05] |
| headers | 0.500 [0.35, 0.65] |  | 0.648 [0.53, 0.76] |  |
| small_to_big | 0.500 [0.35, 0.65] | +0.000 [+0.00, +0.00] | 0.648 [0.53, 0.76] | +0.000 [+0.00, +0.00] |

| config | factual_correctness | Δfactual_correctness vs headers | faithfulness | Δfaithfulness vs headers | context_recall | Δcontext_recall vs headers |
|---|---|---|---|---|---|---|
| baseline | 0.348 [0.27, 0.42] | -0.154 [-0.24, -0.08] | 0.893 [0.85, 0.94] | -0.040 [-0.09, +0.01] | 0.379 [0.25, 0.51] | -0.260 [-0.40, -0.11] |
| headers | 0.502 [0.42, 0.59] |  | 0.933 [0.87, 0.98] |  | 0.640 [0.52, 0.75] |  |
| small_to_big | 0.496 [0.41, 0.58] | -0.007 [-0.07, +0.05] | 0.970 [0.93, 1.00] | +0.038 [+0.00, +0.08] | 0.835 [0.73, 0.93] | +0.196 [+0.11, +0.29] |
