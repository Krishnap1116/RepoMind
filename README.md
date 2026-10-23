# RepoMind — Understand any codebase with AI

Ask natural-language questions about any codebase (local or GitHub) and get answers with file and line citations. RepoMind ships with an evaluation harness: a hand-verified golden set, deterministic retrieval metrics, RAGAS answer scoring and paired bootstrap intervals.

## Results

**Scope:** two Python repos (Flask 3.1.0, requests v2.32.3); 70 hand-verified questions used for development plus 20 held-out questions; local `bge-base-en-v1.5` embeddings; `gpt-4o-mini` as the RAGAS judge. Δ is the paired difference from *headers* with a 95% bootstrap interval. An interval that excludes 0 is treated as a real difference, otherwise as no measurable difference.

**Answer quality** (70 questions, `gpt-4.1-mini` generating answers):

| Condition | Factual correctness | Answer completeness (recall) | Context recall | Faithfulness |
|---|---:|---:|---:|---:|
| baseline (dense search, 512-char chunks)¹ | 0.34 | 0.32 | 0.41 | 0.89 |
| **headers** (contextual chunk headers) | 0.55 | 0.56 | 0.57 | 0.90 |
| **small_to_big** (headers + whole-function context) | 0.57 (+0.02 [−0.03, +0.07]) | 0.62 (+0.06 [+0.01, +0.11]) | 0.78 (+0.21 [+0.12, +0.30]) | 0.93 (+0.03 [−0.02, +0.08]) |

¹ Baseline was scored with `gpt-4o-mini` generating answers. Against headers with the same generator it is −0.17 [−0.24, −0.11] on factual correctness and −0.19 [−0.29, −0.08] on context recall.

What the evidence supports:
1. **Contextual chunk headers are the main win.** Prefixing every chunk with its file, lines and enclosing function improved retrieval (Hit@1 +0.13 [+0.01, +0.24]) and the answers built on it (factual correctness +0.17 [+0.11, +0.24]). This held on the held-out questions (MRR +0.18 [+0.01, +0.35]).
2. **Small-to-big gets the needed facts to the LLM far more often.** Search runs over small fragments, then the LLM reads the whole function each fragment belongs to. Context recall rose by +0.21 on the development set and by +0.18 [+0.04, +0.33] on held-out questions. It roughly doubles the context sent to the LLM (~650 → ~1,500 tokens per question).
3. **Better context did not clearly give better answers.** With `gpt-4.1-mini`, answers covered more of the reference facts (+0.06 [+0.01, +0.11]), but that gain did not replicate on the held-out set (+0.03 [−0.06, +0.13]). Factual correctness (F1) showed no difference, and with `gpt-4o-mini` neither did completeness.
4. **The generator model mattered more than any retrieval change after headers.** Switching from `gpt-4o-mini` to `gpt-4.1-mini` raised completeness by +0.08 [+0.04, +0.13] with headers and +0.11 [+0.07, +0.16] with small_to_big.
5. **Three techniques did not beat headers on these questions:** syntax-aware chunking as the *search* unit (Hit@1 −0.13 [−0.21, −0.04]), a general-purpose cross-encoder reranker (MRR −0.08 [−0.15, −0.01]), and hybrid BM25 search (no measurable difference). On the held-out locate and multi-file questions, hybrid trended better (MRR +0.10 [−0.04, +0.25]), a lead that is not yet confirmed.
6. **The system says "not in the code"** on 7 of 7 unanswerable questions with `gpt-4.1-mini`, and on 6 of 7 with `gpt-4o-mini`.

Method, per-repo tables and limits: [Evaluation](#evaluation).

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/Krishnap1116/RepoMind.git
cd RepoMind

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key (OpenAI, or OpenRouter: see the comments in the file)
cp .env.example .env
```

Tested with Python 3.12. The embedding model and reranker are downloaded from Hugging Face on first use (~1.5 GB) and run locally.

## Usage

```bash
# Ask about a GitHub repo (optionally pinned to a tag or branch)
python repomind.py --repo https://github.com/pallets/flask --ref 3.1.0 \
  --query "How does Flask handle errors?"

# Ask about a local codebase
python repomind.py --repo ./my_project --query "How is the database connection managed?"

# Pick a retrieval pipeline and grade the answer with RAGAS faithfulness + answer relevancy
python repomind.py --repo ./my_project --query "..." --config headers --score
```

The first query on a repo builds its index; later queries reuse it. Every answer lists the files and line ranges it was built from.

## How it works

```
GitHub URL or local path → clone (if URL)
        ↓
Load supported files (code, front-end, docs, config)
        ↓
Chunk: 512-character splitter (default) or tree-sitter syntax-aware chunks
        ↓
Contextual header per chunk:  "# File: src/flask/app.py | lines 1479 - 1530 | Flask.wsgi_app"
        ↓
Embed locally (bge-base-en-v1.5) → ChromaDB, cached per repo, ref, chunking mode and embedding model
        ↓
Query → dense search (optionally + BM25 via Reciprocal Rank Fusion, optionally + cross-encoder rerank)
        ↓
small_to_big: replace each retrieved fragment by the whole function/class it belongs to (deduplicated)
        ↓
Top 5 contexts → LLM answer with file citations
```

## Retrieval pipelines

Pick one with `--config`. The default is `small_to_big`. Every config differs from `headers` by one component, so together they form an ablation:

| Config | What it is |
|--------|------------|
| `baseline` | Dense search over 512-character chunks, top 3 chunks to the LLM (the original RepoMind) |
| `dense_k5` | Baseline with top 5 chunks to the LLM |
| `headers` | Contextual header (file, lines, enclosing class/function) prepended to each chunk before embedding |
| `small_to_big` | `headers` search; the LLM reads the whole function/class (up to 1,500 tokens) each fragment belongs to |
| `ast` | `headers` with syntax-aware chunks as the search unit |
| `headers_rerank` | `headers` + `BAAI/bge-reranker-base` cross-encoder reranking the top 20 |
| `hybrid` | `headers` + BM25 keyword search fused with dense search (Reciprocal Rank Fusion) |
| `hybrid_rerank` | `hybrid` + cross-encoder reranking |

## Chunking

- **Character splitter** (default search unit, and always used for docs and config files): LangChain's `RecursiveCharacterTextSplitter` with 512 **characters** per chunk (median ≈ 9 lines of code). Language separators only bias where the cut lands, so functions are often split. On Flask, 54% of functions span more than one chunk.
- **Syntax-aware chunks** (`ast`, and the whole-function contexts in `small_to_big`): code and front-end files are parsed with [tree-sitter](https://tree-sitter.github.io/). Each function, method, class or component becomes one chunk, together with the comments above it. A definition is split only if it exceeds the token budget (450 tokens for search chunks, measured with the embedder's tokenizer; 1,500 for small-to-big contexts), and then at its children (a class into methods) before any line-level cut. On Flask, only 7% of functions are split.

All chunk counts, sizes and split rates above are for `src/flask/*.py` and come from `python eval/chunk_stats.py` (no API calls).

**Why both?** Small fragments are better *search* units, while whole functions are better *reading* units. Using whole functions as the search unit (`ast`) retrieved worse, most likely because code became fewer, larger vectors that focused docs passages out-ranked (9 of its 10 losses had a docs chunk at #1). `small_to_big` uses each chunking for the job it measured best at.

| Category | Extensions | Syntax-aware |
|----------|-----------|:---:|
| Python | `.py` | ✅ |
| JavaScript / TypeScript / React | `.js`, `.mjs`, `.jsx`, `.ts`, `.tsx` | ✅ |
| Front-end | `.html`, `.css`, `.scss`, `.vue`, `.svelte` | ✅ |
| Systems | `.c`, `.h`, `.cpp`, `.hpp`, `.rs`, `.go` | ✅ |
| JVM / .NET / other | `.java`, `.kt`, `.cs`, `.swift`, `.rb`, `.php` | ✅ |
| Docs | `.md`, `.rst`, `.txt` | splitter |
| Config / data / scripts | `.yaml`, `.yml`, `.toml`, `.json`, `.ini`, `.cfg`, `.sql`, `.sh`, `.csv`, `.xml` | splitter |

For front-end files, React components (`function App()`, `const Cart = () => …`, `export default …`) and classes are chunks of their own, like functions. CSS rule blocks and HTML elements have no names, so neighbouring ones are grouped up to the budget, and a large element is split at its child elements. Syntax-aware chunking has been **evaluated only on Python**; other languages use the same algorithm without measured results.

---

## Evaluation

### Question sets
Every reference answer and gold file was checked against the pinned source when written.
- **Development set** (used for every design decision): [eval/golden_flask.jsonl](eval/golden_flask.jsonl) has Flask `3.1.0`, 40 answerable + 4 unanswerable questions. [eval/golden_requests.jsonl](eval/golden_requests.jsonl) has requests `v2.32.3`, 30 answerable + 3 unanswerable questions. 60 of the 70 answerable questions are "how does X work" explanations.
- **Held-out set** (written after the design was fixed, used once): [eval/heldout_flask.jsonl](eval/heldout_flask.jsonl) and [eval/heldout_requests.jsonl](eval/heldout_requests.jsonl), with 10 questions each, 5 multi-file and 5 locate per repo. These add the question types the development set lacked.

### Metrics
- **Retrieval (deterministic, no LLM):** Hit@1, Hit@5, MRR@10 and Recall@5, where a hit is a chunk from a gold *implementation* file. Each retrieved chunk is also labelled by file type (code, docs, tests, changelog, other) to see what fills the top ranks.
- **Answers (RAGAS, `gpt-4o-mini` judge):** factual correctness (F1 against the reference answer), answer completeness (the recall half of factual correctness), faithfulness, answer relevancy, context precision and context recall. Context precision and recall are source-neutral: a docs passage that contains the needed facts counts.
- **Unanswerable questions:** an LLM check of whether the answer says the code doesn't cover it.
- **Uncertainty:** paired percentile bootstrap (10,000 resamples) over per-question scores. Every condition answers the same questions, so differences are bootstrapped per question.

### Answer quality, development set (70 questions)

| Generator | Condition | Factual correctness | Completeness | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | baseline | 0.34 | 0.32 | 0.89 | 0.74 | 0.75 | 0.41 |
| gpt-4o-mini | headers | 0.51 | 0.48 | 0.93 | 0.82 | 0.79 | 0.60 |
| gpt-4o-mini | small_to_big | 0.50 | 0.50 | 0.95 | 0.84 | 0.80 | 0.80 |
| gpt-4.1-mini | headers | 0.55 | 0.56 | 0.90 | 0.84 | 0.78 | 0.57 |
| gpt-4.1-mini | small_to_big | 0.57 | 0.62 | 0.93 | 0.84 | 0.79 | 0.78 |

Paired differences (95% intervals):
- **baseline → headers** (gpt-4o-mini): factual correctness +0.17 [+0.11, +0.24], completeness +0.15 [+0.10, +0.21], context recall +0.19 [+0.08, +0.29], faithfulness +0.05 [0.00, +0.09], answer relevancy +0.08 [+0.02, +0.15].
- **headers → small_to_big**:
  - context recall: +0.20 [+0.12, +0.29] (4o-mini), +0.21 [+0.12, +0.30] (4.1-mini)
  - completeness: +0.03 [−0.02, +0.08] (4o-mini), +0.06 [+0.01, +0.11] (4.1-mini)
  - factual correctness: −0.01 [−0.05, +0.04] (4o-mini), +0.02 [−0.03, +0.07] (4.1-mini)
- **gpt-4o-mini → gpt-4.1-mini**, completeness: +0.08 [+0.04, +0.13] with headers, +0.11 [+0.07, +0.16] with small_to_big.

The small-to-big completeness gain is significant with `gpt-4.1-mini` and not with `gpt-4o-mini`, but the two intervals overlap, so this does **not** show that small-to-big needs a stronger model.

### Held-out set (20 questions, gpt-4.1-mini)

| Condition | Hit@1 | MRR@10 | Factual correctness | Completeness | Faithfulness | Context recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.15 | 0.40 | | | | |
| headers | 0.40 | 0.58 | 0.48 | 0.43 | 0.86 | 0.54 |
| small_to_big | 0.40 | 0.58 | 0.46 | 0.47 | 0.95 | 0.72 |
| ast | 0.35 | 0.53 | | | | |
| headers_rerank | 0.35 | 0.60 | | | | |
| hybrid | 0.50 | 0.68 | | | | |
| hybrid_rerank | 0.40 | 0.62 | | | | |

The answer metrics were scored for `headers` and `small_to_big` only, the two candidates for the final default.

- **Replicated:** headers beat baseline (MRR +0.18 [+0.01, +0.35]), and small-to-big raised context recall (+0.18 [+0.04, +0.33]).
- **Not replicated:** the small-to-big completeness gain (+0.03 [−0.06, +0.13]).
- **Hybrid** trends better than headers on these locate and multi-file questions (MRR +0.10 [−0.04, +0.25]), unlike on the explanation-heavy development set. It is unconfirmed at n = 20.

### Retrieval, development set

| Condition | Flask Hit@1 | Flask MRR@10 | Flask Hit@5 | requests Hit@1 | requests MRR@10 | requests Hit@5 | Pooled Δ Hit@1 vs headers |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.350 | 0.518 | 0.725 | 0.367 | 0.527 | 0.733 | −0.13 [−0.24, −0.01] |
| headers (= small_to_big search) | **0.500** | **0.648** | 0.775 | **0.467** | **0.641** | **0.933** | — |
| ast | 0.350 | 0.532 | 0.775 | 0.367 | 0.555 | 0.800 | −0.13 [−0.21, −0.04] |
| headers_rerank | 0.350 | 0.547 | 0.775 | 0.400 | 0.596 | 0.800 | −0.11 [−0.23, +0.00] |
| hybrid | 0.400 | 0.580 | 0.775 | 0.433 | 0.604 | 0.867 | −0.07 [−0.17, +0.03] |
| hybrid_rerank | 0.375 | 0.561 | 0.775 | 0.433 | 0.608 | 0.800 | −0.09 [−0.20, +0.03] |

**What fills the top rank.** On both repos the top chunk is docs, tests, a changelog or config for 45–58% of questions, in every condition. Re-scoring the same rankings with non-code chunks removed lifts Hit@1 to 0.73–0.88. That is not necessarily a failure: context recall counts a docs passage that contains the answer, and answers are what matter.

Per-technique diagnosis:
- **`ast`:** a docs chunk took #1 in 9 of its 10 losses. Code became fewer, larger vectors (`src/flask/*.py` chunks: 924 → 502, median 130 → 167 tokens including the header), each covering more ideas.
- **Reranker:** on Flask it ordered *code* worse (code-only Hit@1 0.725 vs 0.875), ranking sibling modules such as `src/flask/sansio/app.py` above `src/flask/app.py`. On requests it ordered code slightly better but put docs first more often.
- **Changelog hypothesis, tested and rejected:** `CHANGES.rst` ranked #1 on several Flask questions, but excluding it changed no condition's Flask Hit@1 by more than one question.

### Limits
- Two Python repos, both well documented. Syntax-aware chunking for other languages is implemented but not evaluated.
- One general-purpose embedder (512-token input) and one general-purpose reranker. Code-trained or long-context models could behave differently.
- The judge (`gpt-4o-mini`) is weaker than the strongest generator tested (`gpt-4.1-mini`), and an LLM judge is noisy on single answers. Claims rest on paired differences over 70 (or 20) questions.
- Factual correctness compares against short reference answers, so true extra detail in an answer lowers its F1. Completeness (recall) is reported alongside for that reason.
- The development set is skewed to explanation questions. The held-out set covers locate and multi-file questions but is small (n = 20).
- Small-to-big sends about 2.3× more context tokens and adds ~0.1–0.4 s of latency per question.

### Raw results
Each evaluation run folder in `eval/results/` holds `per_question.csv`, `summary.json` and `ablation.md`; the completeness re-score folders (`factual_recall__…`) hold `per_question.csv` and `intervals.md`.
- **Retrieval, all conditions:** [Flask](eval/results/golden_flask__20261022-175414/ablation.md) · [requests](eval/results/golden_requests__20261022-175653/ablation.md) · [Flask without changelog](eval/results/golden_flask__excl-CHANGES.rst__20261022-175802/ablation.md)
- **Answers with gpt-4o-mini:** [Flask](eval/results/golden_flask__20261022-183522/ablation.md) · [requests](eval/results/golden_requests__20261022-185213/ablation.md) · completeness: `eval/results/factual_recall__20261022-190722`
- **Answers with gpt-4.1-mini:** [Flask](eval/results/golden_flask__20261022-194532/ablation.md) · [requests](eval/results/golden_requests__20261022-195753/ablation.md) · completeness (development + held-out): `eval/results/factual_recall__gpt-4.1-mini__20261023-114520`
- **Held-out:** retrieval [Flask](eval/results/heldout_flask__20261023-113511/ablation.md) · [requests](eval/results/heldout_requests__20261023-113604/ablation.md); answers [Flask](eval/results/heldout_flask__20261023-113946/ablation.md) · [requests](eval/results/heldout_requests__20261023-114328/ablation.md)

### Reproduce

```bash
# Chunking statistics (chunk counts and sizes, functions split): fully local, no API calls
python eval/chunk_stats.py --repo https://github.com/pallets/flask --ref 3.1.0 --prefix src/flask/

# Retrieval only: fully local, no API calls
python eval/run_eval.py --retrieval-only
python eval/run_eval.py --retrieval-only --golden eval/golden_requests.jsonl \
  --repo https://github.com/psf/requests --ref v2.32.3

# Answers + RAGAS (uses GEN_MODEL / JUDGE_MODEL from .env), then the completeness re-score
python eval/run_eval.py --configs baseline,headers,small_to_big
python eval/rescore_factual.py --mode recall --configs headers,small_to_big

# Pooled paired intervals across repos; ":label" tags runs, e.g. by generator
python eval/pool_results.py eval/results/golden_flask__20261022-194532 eval/results/golden_requests__20261022-195753 \
  --metrics factual_correctness,context_recall
python eval/pool_results.py eval/results/golden_flask__20261022-183522:4o-mini eval/results/golden_requests__20261022-185213:4o-mini \
  eval/results/golden_flask__20261022-194532:4.1-mini eval/results/golden_requests__20261022-195753:4.1-mini \
  --reference headers@4o-mini --metrics factual_correctness
python eval/pool_results.py eval/results/factual_recall__gpt-4.1-mini__20261023-114520 --only heldout_ --metrics factual_recall
```

Answers and judge scores are cached in `eval/.cache/` per golden set, condition and generator, so an interrupted run resumes without paying twice. To build a golden set for another repo, run `eval/make_golden_draft.py`. It drafts questions from the repo's Python source into `eval/draft_<repo>.jsonl`; verify every row by hand before saving it as a `golden_*.jsonl` file.

## Tests

```bash
pytest tests
```

The tests cover syntax-aware chunking (definitions kept whole, class split into methods, React components, line numbers), the fragment-to-whole-function mapping, Reciprocal Rank Fusion, code-aware tokenization and the evaluation metrics. They need no API key.

## Tech stack

| Component | Tool |
|-----------|------|
| LLM | OpenAI `gpt-4.1-mini` by default; any OpenAI-compatible API (e.g. OpenRouter) via `.env` |
| Embeddings | `BAAI/bge-base-en-v1.5` via sentence-transformers (local) |
| Vector store | ChromaDB (persistent, local) |
| Chunking | tree-sitter (`tree-sitter-language-pack`) and LangChain `RecursiveCharacterTextSplitter` |
| Keyword search | BM25 (`rank_bm25`) with code-aware tokenization (snake_case / CamelCase splitting) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (local) |
| Evaluation | RAGAS + deterministic retrieval metrics, paired bootstrap intervals, pytest |
