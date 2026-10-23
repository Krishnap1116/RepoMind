"""Pool per-question results from several runs (e.g. two repos) and report paired bootstrap intervals.

    python eval/pool_results.py eval/results/golden_flask__... eval/results/golden_requests__...

Append ":label" to a directory to tag its configs (e.g. by generator model), which lets runs of the
same questions with different models be compared:

    python eval/pool_results.py dirA:4o-mini dirB:4o-mini dirC:4.1-mini dirD:4.1-mini \
        --reference headers@4o-mini --metrics factual_correctness

--did A,B,C,D also bootstraps the paired difference-in-difference (A - B) - (C - D) per question, e.g.
whether small_to_big's gain over headers is larger with one generator than with another.
"""
import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_eval import bootstrap_ci, to_markdown, uncertainty  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Pool eval runs and bootstrap paired differences")
    parser.add_argument("result_dirs", nargs="+", help="results directories, each optionally suffixed with :label")
    parser.add_argument("--reference", default="headers")
    parser.add_argument("--only", default="",
                        help="Keep only question ids starting with this prefix, e.g. 'heldout_'")
    parser.add_argument("--metrics", default="hit@1,mrr@10",
                        help="Comma-separated per-question columns, e.g. factual_correctness,faithfulness")
    parser.add_argument("--did", default="",
                        help="A,B,C,D: interval for (A - B) - (C - D), e.g. "
                             "small_to_big@4.1-mini,headers@4.1-mini,small_to_big@4o-mini,headers@4o-mini")
    args = parser.parse_args()

    frames = []
    for entry in args.result_dirs:
        result_dir, _, label = entry.partition(":")
        df = pd.read_csv(os.path.join(result_dir, "per_question.csv"))
        # question ids repeat across golden sets, so namespace them by golden set (the run tag);
        # rescore outputs are already namespaced
        golden_set = os.path.basename(result_dir.rstrip("/")).split("__")[0]
        df["id"] = df["id"].where(df["id"].str.contains(":"), golden_set + ":" + df["id"])
        if label:
            df["config"] = df["config"] + "@" + label
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled[pooled["id"].str.startswith(args.only)]
    n = pooled[pooled["type"] != "unanswerable"]["id"].nunique()
    print(f"{n} answerable questions pooled from {len(frames)} runs\n")
    print(to_markdown(uncertainty(pooled, reference=args.reference, metrics=args.metrics.split(","))))
    if args.did:
        a, b, c, d = args.did.split(",")
        answerable = pooled[pooled["type"] != "unanswerable"]
        for metric in args.metrics.split(","):
            wide = answerable.pivot(index="id", columns="config", values=metric)
            diffs = ((wide[a] - wide[b]) - (wide[c] - wide[d])).dropna().to_numpy(dtype=float)
            low, high = bootstrap_ci(diffs)
            print(f"\n{metric}: ({a} - {b}) - ({c} - {d}) = {diffs.mean():+.3f} [{low:+.2f}, {high:+.2f}], "
                  f"{len(diffs)} questions")


if __name__ == "__main__":
    main()
