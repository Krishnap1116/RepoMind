"""Draft golden-set questions for a new repo from its Python source files (tests and docs are skipped).

Drafts are written to eval/draft_<repo>.jsonl. Every row MUST be checked against the source by hand
before it goes into a golden_*.jsonl file; unverified drafts make the metrics meaningless.

    python eval/make_golden_draft.py --repo https://github.com/psf/requests --ref v2.32.3 --n 30
"""
import os
import sys
import json
import random
import argparse

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))

from repomind import (  # noqa: E402
    GEN_MODEL, cleanup_repo, clone_repo, get_repo_name, is_github_url, load_codebase, make_chat_model,
)

PROMPT = """You are writing an evaluation set for a code question-answering system.
Below is a snippet from the file `{source}`. Write ONE question a developer might ask that this
snippet answers, and a concise reference answer that names the file and function involved.
Ask about behavior ("how/what happens/where"), not trivia like line numbers.

Snippet:
{snippet}

Reply ONLY with JSON: {{"question": "...", "reference_answer": "..."}}"""


def main():
    parser = argparse.ArgumentParser(description="Draft golden-set Q/A pairs for manual review")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cloned = is_github_url(args.repo)
    repo_path = clone_repo(args.repo, ref=args.ref) if cloned else args.repo
    # only source code: questions about docs/tests are not what we want to measure
    chunks = [doc for doc in load_codebase(repo_path)
              if doc.metadata["language"] == ".py" and not doc.metadata["source"].startswith(("tests", "docs"))
              and len(doc.page_content) > 300]
    if cloned:
        cleanup_repo(repo_path)

    random.seed(args.seed)
    llm = make_chat_model(GEN_MODEL, temperature=0.3)
    # not golden_*: drafts must never be picked up as a verified golden set
    out_path = os.path.join(EVAL_DIR, f"draft_{get_repo_name(args.repo)}.jsonl")
    with open(out_path, "w") as f:
        for n, chunk in enumerate(random.sample(chunks, min(args.n, len(chunks))), 1):
            reply = llm.invoke(PROMPT.format(source=chunk.metadata["source"], snippet=chunk.page_content))
            content = reply.content.strip().strip("```json").strip("```").strip()
            try:
                draft = json.loads(content)
            except json.JSONDecodeError:
                print(f"  skipped unparseable draft for {chunk.metadata['source']}")
                continue
            row = {"id": f"d{n:02d}", "type": "explain", **draft, "gold_files": [chunk.metadata["source"]]}
            f.write(json.dumps(row) + "\n")
    print(f"Drafts written to {out_path}. Review every row before using it.")


if __name__ == "__main__":
    main()
