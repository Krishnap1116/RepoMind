| config | hit@1 | hit@5 | mrr@10 | recall@5 | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness | context_tokens | p50_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| headers | 0.3 | 0.9 | 0.52 | 0.733 | 0.871 | 0.879 | 0.741 | 0.45 | 0.404 | 665 | 1.56 |
| small_to_big | 0.3 | 0.9 | 0.52 | 0.733 | 0.956 | 0.869 | 0.745 | 0.667 | 0.485 | 1564 | 1.54 |

## By chunk type

| config | hit@1 | noncode@1 | noncode_share@5 | code_hit@1 | code_mrr@10 |
|---|---|---|---|---|---|
| headers | 0.3 | 0.6 | 0.48 | 0.8 | 0.883 |
| small_to_big | 0.3 | 0.6 | 0.48 | 0.8 | 0.883 |

## 95% bootstrap intervals (paired differences vs headers)

| config | hit@1 | mrr@10 | Δhit@1 vs headers | Δmrr@10 vs headers |
|---|---|---|---|---|
| headers | 0.300 [0.00, 0.60] | 0.520 [0.33, 0.73] |  |  |
| small_to_big | 0.300 [0.00, 0.60] | 0.520 [0.33, 0.73] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] |

| config | factual_correctness | faithfulness | context_recall | Δfactual_correctness vs headers | Δfaithfulness vs headers | Δcontext_recall vs headers |
|---|---|---|---|---|---|---|
| headers | 0.404 [0.25, 0.54] | 0.871 [0.78, 0.96] | 0.450 [0.20, 0.72] |  |  |  |
| small_to_big | 0.485 [0.38, 0.60] | 0.956 [0.88, 1.00] | 0.667 [0.37, 0.93] | +0.081 [-0.10, +0.28] | +0.084 [+0.01, +0.18] | +0.217 [+0.03, +0.43] |
