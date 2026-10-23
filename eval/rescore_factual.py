"""Re-score cached answers with RAGAS factual correctness in recall or precision mode.

The default F1 mode also penalizes true claims that the (short) reference answer doesn't mention.
Recall mode asks only whether the answer covers the reference's facts.

    python eval/rescore_factual.py --mode recall --configs baseline,headers,small_to_big
"""
import os
import sys
import glob
import json
import argparse
from collections import Counter
from datetime import datetime

import pandas as pd

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

from run_eval import (  # noqa: E402
    GEN_MODEL, JUDGE_MODEL, cache_path, make_chat_model, read_cache, run_cost, to_markdown, uncertainty, write_cache,
)


def load_golden_sets():
    """{run_tag: {question_id: row}} for every golden and held-out question file."""
    sets = {}
    for path in glob.glob(os.path.join(EVAL_DIR, "golden_*.jsonl")) + glob.glob(os.path.join(EVAL_DIR, "heldout_*.jsonl")):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        sets[os.path.splitext(os.path.basename(path))[0]] = {row["id"]: row for row in rows}
    return sets


def score(records, golden, mode, usage):
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import FactualCorrectness

    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"], "reference": golden[r["id"]]["reference_answer"]}
        for r in records
    ])
    result = evaluate(
        dataset,
        metrics=[FactualCorrectness(mode=mode)],
        llm=LangchainLLMWrapper(make_chat_model(JUDGE_MODEL)),
        run_config=RunConfig(max_workers=4, timeout=300, max_wait=90),
        token_usage_parser=get_token_usage_for_openai,
    )
    tokens = result.total_tokens()
    usage["input_tokens"] += tokens.input_tokens
    usage["output_tokens"] += tokens.output_tokens
    return [None if pd.isna(v) else float(v) for v in result.to_pandas().iloc[:, -1]]


def main():
    parser = argparse.ArgumentParser(description="Re-score cached answers with factual correctness recall/precision")
    parser.add_argument("--mode", choices=["recall", "precision"], default="recall")
    parser.add_argument("--configs", default="baseline,headers,small_to_big")
    parser.add_argument("--reference", default="headers")
    args = parser.parse_args()

    column = f"factual_{args.mode}"
    golden_sets = load_golden_sets()
    usage = Counter()
    rows = []
    for name in args.configs.split(","):
        # only answers from the current GEN_MODEL; each generator has its own cache file
        for path in sorted(glob.glob(cache_path(name, "*"))):
            run_tag = os.path.basename(path).split(f"__{name}__")[0]
            if run_tag not in golden_sets:
                continue
            golden = golden_sets[run_tag]
            cached = read_cache(path)
            answerable = [r for r in cached.values() if golden[r["id"]]["type"] != "unanswerable"]
            todo = [r for r in answerable if r.get(f"{column}_judge") != JUDGE_MODEL or r.get(column) is None]
            if todo:
                print(f"  scoring {len(todo)} answers: {run_tag} / {name}")
                for record, value in zip(todo, score(todo, golden, args.mode, usage)):
                    record[column] = value
                    record[f"{column}_judge"] = JUDGE_MODEL
                write_cache(path, cached)
            rows += [{"id": f"{run_tag}:{r['id']}", "config": name, "type": golden[r["id"]]["type"],
                      column: r.get(column)} for r in answerable]

    df = pd.DataFrame(rows)
    table = to_markdown(uncertainty(df, reference=args.reference, metrics=[column]))
    model = GEN_MODEL.replace("/", "_")
    out_dir = os.path.join(EVAL_DIR, "results", f"{column}__{model}__{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir)
    df.to_csv(os.path.join(out_dir, "per_question.csv"), index=False)
    with open(os.path.join(out_dir, "intervals.md"), "w") as f:
        f.write(f"{df['id'].nunique()} answerable questions, generator {GEN_MODEL}, judge {JUDGE_MODEL}\n\n{table}\n")
    if usage:
        cost = run_cost(usage)
        print(f"  new API usage: {usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens"
              + (f" ≈ ${cost:.3f}" if cost is not None else ""))
    print(f"\n{df['id'].nunique()} answerable questions, generator {GEN_MODEL}\n\n{table}\n\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
