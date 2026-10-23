| config | hit@1 | hit@5 | mrr@10 | recall@5 | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness | refusal_rate | context_tokens | p50_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| headers | 0.5 | 0.775 | 0.648 | 0.738 | 0.874 | 0.837 | 0.836 | 0.602 | 0.543 | 1.0 | 653 | 1.61 |
| small_to_big | 0.5 | 0.775 | 0.648 | 0.738 | 0.946 | 0.869 | 0.851 | 0.81 | 0.571 | 1.0 | 1565 | 1.69 |

## By chunk type

| config | hit@1 | noncode@1 | noncode_share@5 | code_hit@1 | code_mrr@10 |
|---|---|---|---|---|---|
| headers | 0.5 | 0.45 | 0.52 | 0.875 | 0.9 |
| small_to_big | 0.5 | 0.45 | 0.52 | 0.875 | 0.9 |

## 95% bootstrap intervals (paired differences vs headers)

| config | hit@1 | mrr@10 | Δhit@1 vs headers | Δmrr@10 vs headers |
|---|---|---|---|---|
| headers | 0.500 [0.35, 0.65] | 0.648 [0.53, 0.76] |  |  |
| small_to_big | 0.500 [0.35, 0.65] | 0.648 [0.53, 0.76] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] |

| config | factual_correctness | faithfulness | context_recall | Δfactual_correctness vs headers | Δfaithfulness vs headers | Δcontext_recall vs headers |
|---|---|---|---|---|---|---|
| headers | 0.543 [0.46, 0.62] | 0.874 [0.81, 0.93] | 0.602 [0.48, 0.72] |  |  |  |
| small_to_big | 0.571 [0.50, 0.64] | 0.946 [0.91, 0.98] | 0.810 [0.70, 0.91] | +0.028 [-0.03, +0.09] | +0.072 [+0.00, +0.14] | +0.208 [+0.08, +0.33] |
