"""Chunk statistics behind the README's Chunking section, for the character splitter and syntax-aware chunks.

For the Python files under --prefix it reports the number of chunks, their median size in lines and in
embedder tokens (including the contextual header, as indexed), how many functions/methods are split
across more than one chunk, and how many chunks exceed the embedder's 512-token input. A function counts
as split when no single chunk contains all of it (decorators included), using Python's own parser.

    python eval/chunk_stats.py --repo https://github.com/pallets/flask --ref 3.1.0 --prefix src/flask/

Needs the same environment as indexing: tree-sitter-language-pack and the embedder's tokenizer
(downloaded from Hugging Face on first use). No API calls.
"""
import os
import re
import ast
import sys
import argparse
import statistics

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))

from repomind import (  # noqa: E402
    EMBED_MODEL, MAX_CHUNK_TOKENS, cleanup_repo, clone_repo, count_tokens, is_github_url, load_codebase,
)


def function_spans(text):
    """(start, end) line spans of every function and method, decorators included."""
    return [
        (min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno)
        for node in ast.walk(ast.parse(text))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def stats(repo_path, prefix, chunking):
    docs = [doc for doc in load_codebase(repo_path, chunk_headers=True, chunking=chunking)
            if doc.metadata["source"].startswith(prefix) and doc.metadata["source"].endswith(".py")]
    spans_by_file = {}
    for doc in docs:
        start, end = map(int, re.findall(r"\d+", doc.metadata["lines"]))
        spans_by_file.setdefault(doc.metadata["source"], []).append((start, end))

    functions = split = 0
    for source, spans in spans_by_file.items():
        with open(os.path.join(repo_path, source), encoding="utf-8") as f:
            for start, end in function_spans(f.read()):
                functions += 1
                split += not any(s <= start and end <= e for s, e in spans)

    tokens = [count_tokens(doc.page_content) for doc in docs]
    # page_content starts with the one-line header; count the code lines only
    lines = [doc.page_content.count("\n") for doc in docs]
    return {
        "chunks": len(docs),
        "median lines": statistics.median(lines),
        "median tokens (with header)": statistics.median(tokens),
        "functions split": f"{split}/{functions} ({split / functions:.0%})",
        "chunks over 512 tokens": sum(t + 2 > 512 for t in tokens),   # +2: [CLS] and [SEP]
    }


def main():
    parser = argparse.ArgumentParser(description="Chunk statistics for the splitter vs syntax-aware chunking")
    parser.add_argument("--repo", default="https://github.com/pallets/flask")
    parser.add_argument("--ref", default="3.1.0")
    parser.add_argument("--prefix", default="src/flask/", help="Only Python files whose path starts with this")
    args = parser.parse_args()

    cloned = is_github_url(args.repo)
    repo_path = clone_repo(args.repo, ref=args.ref) if cloned else args.repo
    try:
        results = {mode: stats(repo_path, args.prefix, mode) for mode in ("splitter", "ast")}
    finally:
        if cloned:
            cleanup_repo(repo_path)

    print(f"\n{args.repo} @ {args.ref}, Python files under {args.prefix!r}, embedder {EMBED_MODEL}, "
          f"syntax-aware budget {MAX_CHUNK_TOKENS} tokens\n")
    print(f"| metric | splitter | ast |\n|---|---:|---:|")
    for metric in results["splitter"]:
        print(f"| {metric} | {results['splitter'][metric]} | {results['ast'][metric]} |")


if __name__ == "__main__":
    main()
