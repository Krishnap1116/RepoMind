| config | hit@1 | hit@5 | mrr@10 | recall@5 | faithfulness | answer_relevancy | context_precision | context_recall | factual_correctness | refusal_rate | context_tokens | p50_latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| headers | 0.467 | 0.933 | 0.641 | 0.867 | 0.929 | 0.848 | 0.709 | 0.525 | 0.56 | 1.0 | 647 | 1.61 |
| small_to_big | 0.467 | 0.933 | 0.641 | 0.867 | 0.899 | 0.797 | 0.714 | 0.744 | 0.577 | 1.0 | 1454 | 1.75 |

## By chunk type

| config | hit@1 | noncode@1 | noncode_share@5 | code_hit@1 | code_mrr@10 |
|---|---|---|---|---|---|
| headers | 0.467 | 0.467 | 0.453 | 0.767 | 0.878 |
| small_to_big | 0.467 | 0.467 | 0.453 | 0.767 | 0.878 |

## 95% bootstrap intervals (paired differences vs headers)

| config | hit@1 | mrr@10 | Δhit@1 vs headers | Δmrr@10 vs headers |
|---|---|---|---|---|
| headers | 0.467 [0.30, 0.63] | 0.641 [0.51, 0.77] |  |  |
| small_to_big | 0.467 [0.30, 0.63] | 0.641 [0.51, 0.77] | +0.000 [+0.00, +0.00] | +0.000 [+0.00, +0.00] |

| config | factual_correctness | faithfulness | context_recall | Δfactual_correctness vs headers | Δfaithfulness vs headers | Δcontext_recall vs headers |
|---|---|---|---|---|---|---|
| headers | 0.560 [0.49, 0.63] | 0.929 [0.88, 0.97] | 0.525 [0.38, 0.67] |  |  |  |
| small_to_big | 0.577 [0.52, 0.63] | 0.899 [0.84, 0.95] | 0.744 [0.61, 0.87] | +0.017 [-0.06, +0.10] | -0.030 [-0.08, +0.02] | +0.219 [+0.10, +0.35] |
