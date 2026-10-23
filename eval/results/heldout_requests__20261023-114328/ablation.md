| config | hit@1 | hit@5 | mrr@10 | recall@5 | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness | context_tokens | p50_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| headers | 0.5 | 0.9 | 0.634 | 0.683 | 0.84 | 0.814 | 0.823 | 0.625 | 0.546 | 646 | 1.5 |
| small_to_big | 0.5 | 0.9 | 0.634 | 0.683 | 0.938 | 0.825 | 0.834 | 0.767 | 0.436 | 1124 | 1.3 |

## By chunk type

| config | hit@1 | noncode@1 | noncode_share@5 | code_hit@1 | code_mrr@10 |
|---|---|---|---|---|---|
| headers | 0.5 | 0.5 | 0.6 | 0.9 | 0.933 |
| small_to_big | 0.5 | 0.5 | 0.6 | 0.9 | 0.933 |

## 95% bootstrap intervals (paired differences vs headers)

| config | hit@1 | mrr@10 | Δhit@1 vs headers | Δmrr@10 vs headers |
|---|---|---|---|---|
| headers | 0.500 [0.20, 0.80] | 0.634 [0.39, 0.86] |  |  |
| small_to_big | 0.500 [0.20, 0.80] | 0.634 [0.39, 0.86] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] |

| config | factual_correctness | faithfulness | context_recall | Δfactual_correctness vs headers | Δfaithfulness vs headers | Δcontext_recall vs headers |
|---|---|---|---|---|---|---|
| headers | 0.546 [0.44, 0.65] | 0.840 [0.70, 0.96] | 0.625 [0.43, 0.81] |  |  |  |
| small_to_big | 0.436 [0.31, 0.55] | 0.938 [0.85, 1.00] | 0.767 [0.62, 0.92] | -0.110 [-0.30, +0.05] | +0.099 [-0.07, +0.28] | +0.142 [-0.05, +0.38] |
