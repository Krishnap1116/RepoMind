"""Run the golden set through each retrieval config and report retrieval + RAGAS metrics.

    python eval/run_eval.py                          # all configs, full golden set
    python eval/run_eval.py --configs baseline --limit 5 --no-ragas
"""
import os
import re
import sys
import json
import time
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))

from repomind import (  # noqa: E402
    CONFIGS, EMBED_MODEL, GEN_MODEL, JUDGE_MODEL, RERANK_MODEL, RAGPipeline, chunk_kind, count_tokens,
    format_chunk, generate_answer, make_chat_model, make_embeddings,
)

DEFAULT_REPO = "https://github.com/pallets/flask"
DEFAULT_REF = "3.1.0"
METRIC_K = 10    # depth of the ranked list used for retrieval metrics
RETRIEVE_K = 20  # depth collected; = k_candidates so the reranker pool is unchanged, extra depth feeds code-only scoring
CACHE_DIR = os.path.join(EVAL_DIR, ".cache")
# $ per token (input, output); used to report what a run cost
PRICES = {"gpt-4o-mini": (0.15e-6, 0.60e-6), "gpt-4.1-mini": (0.40e-6, 1.60e-6), "gpt-4o": (2.50e-6, 10.00e-6)}

RETRIEVAL_COLUMNS = ["hit@1", "hit@5", "mrr@10", "recall@5"]
KIND_COLUMNS = ["noncode@1", "noncode_share@5", "code_hit@1", "code_mrr@10"]
RAGAS_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "factual_correctness(mode=f1)": "factual_correctness",
}
JUDGED_FIELDS = [*RAGAS_COLUMNS.values(), "refused", "ragas_judge"]


def load_golden(path, limit=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def retrieval_metrics(sources, gold_files):
    """File-level metrics over the ranked chunk list: a chunk is a hit if it comes from a gold file."""
    gold = set(gold_files)
    hits = [source in gold for source in sources[:METRIC_K]]
    first_hit = next((i for i, hit in enumerate(hits) if hit), None)
    return {
        "hit@1": float(any(hits[:1])),
        "hit@5": float(any(hits[:5])),
        "mrr@10": 1 / (first_hit + 1) if first_hit is not None else 0.0,
        "recall@5": len(gold & set(sources[:5])) / len(gold),
    }


def kind_metrics(sources, gold_files):
    """Tests the 'prose outranks code' diagnosis on the same ranked list.

    noncode@1 / noncode_share@5: how much of the top of the ranking is docs, tests, changelog, etc.
    code_hit@1 / code_mrr@10: the same ranking with every non-code chunk removed. If a config loses
    on hit@1 but not on code_hit@1, non-code chunks are what pushed the gold code down.
    """
    kinds = [chunk_kind(source) for source in sources]
    code_only = [source for source, kind in zip(sources, kinds) if kind == "code"]
    code = retrieval_metrics(code_only, gold_files)
    return {
        "top1_kind": kinds[0] if kinds else None,
        "noncode@1": float(bool(kinds) and kinds[0] != "code"),
        "noncode_share@5": sum(kind != "code" for kind in kinds[:5]) / max(len(kinds[:5]), 1),
        "code_hit@1": code["hit@1"],
        "code_mrr@10": code["mrr@10"],
    }


def cache_path(config_name, run_tag):
    """run_tag identifies the golden set + exclusions, so different runs never share answers."""
    model = re.sub(r"[^\w.-]", "_", GEN_MODEL)
    return os.path.join(CACHE_DIR, f"{run_tag}__{config_name}__{model}.jsonl")


def read_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {row["id"]: row for row in map(json.loads, f)}


def write_cache(path, records):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        for record in records.values():
            f.write(json.dumps(record) + "\n")


def retrieve_records(config, golden, repo, ref, exclude):
    """Retrieval only: no LLM calls, so it runs fully offline once the index exists."""
    pipeline = RAGPipeline(repo, config, ref=ref, exclude=exclude)
    records = []
    for row in golden:
        start = time.perf_counter()
        ranked = pipeline.retrieve(row["question"], k=RETRIEVE_K)
        records.append(dict(row, sources=[chunk.metadata["source"] for chunk in ranked],
                            latency_s=round(time.perf_counter() - start, 3)))
    return records


def generate_records(config, golden, repo, ref, exclude, run_tag, usage):
    """Retrieve + answer every golden question, reusing cached answers from earlier runs."""
    path = cache_path(config.name, run_tag)
    cached = read_cache(path)
    todo = [row for row in golden if cached.get(row["id"], {}).get("question") != row["question"]]

    if todo:
        pipeline = RAGPipeline(repo, config, ref=ref, exclude=exclude)
        for n, row in enumerate(todo, 1):
            start = time.perf_counter()
            ranked = pipeline.retrieve(row["question"], k=RETRIEVE_K)
            chunks = pipeline.contexts(ranked)[:config.k_final]
            answer = generate_answer(row["question"], chunks, pipeline.llm, usage)
            contexts = [format_chunk(chunk) for chunk in chunks]
            cached[row["id"]] = {
                "id": row["id"],
                "question": row["question"],
                "answer": answer,
                "contexts": contexts,
                "context_tokens": count_tokens("\n\n".join(contexts)),
                "sources": [chunk.metadata["source"] for chunk in ranked],
                "latency_s": round(time.perf_counter() - start, 3),
            }
            print(f"  [{config.name}] {n}/{len(todo)} {row['id']}")
            write_cache(path, cached)

    return [dict(cached[row["id"]], **row) for row in golden]


def judge_refusal(records, judge, usage):
    """For unanswerable questions: did the answer say the code doesn't cover it instead of inventing one?"""
    for record in records:
        if record["type"] != "unanswerable" or "refused" in record:
            continue
        prompt = f"""
        A user asked a code assistant a question that the codebase cannot answer.
        Did the assistant clearly say the information is not in the code (instead of inventing an answer)?
        Question: {record['question']}
        Answer: {record['answer']}
        Reply with only "yes" or "no".
        """
        response = judge.invoke(prompt)
        if response.usage_metadata:
            usage["input_tokens"] += response.usage_metadata["input_tokens"]
            usage["output_tokens"] += response.usage_metadata["output_tokens"]
        record["refused"] = float(response.content.strip().lower().startswith("yes"))


def score_with_ragas(records, judge_llm, workers, usage):
    """Add RAGAS scores to answerable records that don't have them yet for this judge model."""
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference,
        LLMContextRecall, ResponseRelevancy,
    )

    # rescore anything judged by another model or left empty by a failed (e.g. rate-limited) call
    todo = [
        r for r in records
        if r["type"] != "unanswerable"
        and (r.get("ragas_judge") != JUDGE_MODEL or any(r.get(c) is None for c in RAGAS_COLUMNS.values()))
    ]
    if not todo:
        return
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["reference_answer"],
        }
        for r in todo
    ])
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
            LLMContextRecall(), FactualCorrectness(),
        ],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(make_embeddings()),
        run_config=RunConfig(max_workers=workers, timeout=300, max_wait=90),
        token_usage_parser=get_token_usage_for_openai,
    )
    tokens = result.total_tokens()
    usage["input_tokens"] += tokens.input_tokens
    usage["output_tokens"] += tokens.output_tokens
    scores = result.to_pandas()
    for record, (_, row) in zip(todo, scores.iterrows()):
        for column, name in RAGAS_COLUMNS.items():
            value = row.get(column)
            record[name] = None if pd.isna(value) else float(value)
        record["ragas_judge"] = JUDGE_MODEL


def summarize(df):
    answerable = df[df["type"] != "unanswerable"]
    rows = []
    for name, group in answerable.groupby("config", sort=False):
        summary = {"config": name}
        for column in RETRIEVAL_COLUMNS + KIND_COLUMNS + [c for c in RAGAS_COLUMNS.values() if c in group]:
            summary[column] = round(group[column].mean(), 3)
        unanswerable = df[(df["config"] == name) & (df["type"] == "unanswerable")]
        if "refused" in unanswerable and len(unanswerable):
            summary["refusal_rate"] = round(unanswerable["refused"].mean(), 3)
        if "context_tokens" in group:
            summary["context_tokens"] = round(group["context_tokens"].mean())
        summary["p50_latency_s"] = round(df[df["config"] == name]["latency_s"].median(), 2)
        rows.append(summary)
    return pd.DataFrame(rows)


def run_cost(usage, model=JUDGE_MODEL):
    """Dollar cost of the tokens in usage at model's price, or None for an unknown model."""
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return usage["input_tokens"] * input_price + usage["output_tokens"] * output_price


def describe_usage(gen_usage, judge_usage):
    parts, total = [], 0.0
    for label, usage, model in (("generation", gen_usage, GEN_MODEL), ("judge", judge_usage, JUDGE_MODEL)):
        if usage:
            cost = run_cost(usage, model)
            total = None if cost is None or total is None else total + cost
            parts.append(f"{label} {usage['input_tokens']:,} in + {usage['output_tokens']:,} out ({model})")
    return "; ".join(parts) + (f" ≈ ${total:.3f}" if total is not None else "")


def bootstrap_ci(values, n_resamples=10000, seed=0):
    """95% percentile bootstrap interval for the mean of per-question scores."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def uncertainty(df, reference="headers", metrics=("hit@1", "mrr@10")):
    """Mean with 95% CI per config, plus the paired difference vs the reference config.

    Every config answers the same questions, so differences are bootstrapped per question
    (paired), which is tighter and more honest than comparing two independent intervals.
    """
    answerable = df[df["type"] != "unanswerable"]
    scores = {name: group.set_index("id") for name, group in answerable.groupby("config", sort=False)}
    rows = []
    for name, group in scores.items():
        row = {"config": name}
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            low, high = bootstrap_ci(values)
            row[metric] = f"{values.mean():.3f} [{low:.2f}, {high:.2f}]"
            if reference in scores and name != reference:
                diffs = (group[metric] - scores[reference][metric].reindex(group.index)).dropna().to_numpy(dtype=float)
                low, high = bootstrap_ci(diffs)
                row[f"Δ{metric} vs {reference}"] = f"{diffs.mean():+.3f} [{low:+.2f}, {high:+.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RepoMind on a golden set")
    parser.add_argument("--golden", default=os.path.join(EVAL_DIR, "golden_flask.jsonl"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--configs", default="all",
                        help=f"Comma-separated subset of {', '.join(CONFIGS)} or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N golden questions")
    parser.add_argument("--no-ragas", action="store_true", help="Skip LLM-judged metrics")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only retrieval metrics: no answers, no API calls")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel RAGAS judge calls (keep low to stay under API rate limits)")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated glob patterns of files to leave out of retrieval, e.g. 'CHANGES.rst'")
    args = parser.parse_args()

    names = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    exclude = tuple(p for p in args.exclude.split(",") if p)
    golden = load_golden(args.golden, args.limit)
    run_tag = os.path.splitext(os.path.basename(args.golden))[0]
    if exclude:
        run_tag += "__excl-" + re.sub(r"[^\w.-]", "_", "+".join(exclude))
    judge = None if args.retrieval_only or args.no_ragas else make_chat_model(JUDGE_MODEL)
    print(f"{len(golden)} questions | generator={GEN_MODEL} | judge={JUDGE_MODEL}")

    all_records = []
    for name in names:
        config = CONFIGS[name]
        gen_usage, judge_usage = Counter(), Counter()
        print(f"\n== {name} ==")
        if args.retrieval_only:
            records = retrieve_records(config, golden, args.repo, args.ref, exclude)
        else:
            records = generate_records(config, golden, args.repo, args.ref, exclude, run_tag, gen_usage)
        for record in records:
            if record["gold_files"]:
                record.update(retrieval_metrics(record["sources"], record["gold_files"]))
                record.update(kind_metrics(record["sources"], record["gold_files"]))
        if not (args.no_ragas or args.retrieval_only):
            judge_refusal(records, judge, judge_usage)
            score_with_ragas(records, judge, args.workers, judge_usage)
            # judged scores live in the same cache so reruns don't pay for them again
            cached = read_cache(cache_path(name, run_tag))
            for record in records:
                cached[record["id"]].update({k: record[k] for k in JUDGED_FIELDS if k in record})
            write_cache(cache_path(name, run_tag), cached)
        all_records += [dict(record, config=name) for record in records]
        if gen_usage or judge_usage:
            print(f"  [{name}] new API usage: {describe_usage(gen_usage, judge_usage)}")

    df = pd.DataFrame(all_records)
    summary = summarize(df)

    out_dir = os.path.join(EVAL_DIR, "results", f"{run_tag}__{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir)
    df.drop(columns=["contexts"], errors="ignore").to_csv(os.path.join(out_dir, "per_question.csv"), index=False)
    meta = {"repo": args.repo, "ref": args.ref, "questions": len(golden), "exclude": list(exclude),
            "embeddings": EMBED_MODEL, "reranker": RERANK_MODEL,
            "generator": None if args.retrieval_only else GEN_MODEL,
            "judge": None if args.retrieval_only or args.no_ragas else JUDGE_MODEL}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"meta": meta, "results": summary.to_dict(orient="records")}, f, indent=2)
    table = to_markdown(summary.drop(columns=KIND_COLUMNS))
    by_kind = to_markdown(summary[["config", "hit@1", *KIND_COLUMNS]])
    intervals = to_markdown(uncertainty(df))
    ragas_metrics = [c for c in ("factual_correctness", "faithfulness", "context_recall") if c in df]
    if ragas_metrics:
        intervals += "\n\n" + to_markdown(uncertainty(df, metrics=ragas_metrics))
    with open(os.path.join(out_dir, "ablation.md"), "w") as f:
        f.write(table + "\n\n## By chunk type\n\n" + by_kind
                + "\n\n## 95% bootstrap intervals (paired differences vs headers)\n\n" + intervals + "\n")

    print(f"\n{table}\n\nBy chunk type:\n{by_kind}\n\n95% CIs:\n{intervals}\n\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
