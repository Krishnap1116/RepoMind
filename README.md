# RepoMind — Understand any codebase with AI

Ask natural language questions about any codebase — local or GitHub — and get answers with source attribution and hallucination scoring.

# How to run

```bash
python repomind.py --repo https://github.com/pallets/flask --query "How does routing work?"
```

## How it works

```
GitHub URL or Local Path
        ↓
  Clone (if URL)
        ↓
  Load all files (.py, .js, .md, .yaml, .sql, ...)
        ↓
  Language-aware chunking (splits at function/class boundaries)
        ↓
  Embed chunks → store in ChromaDB (cached to disk)
        ↓
  Delete cloned repo (ChromaDB has everything)
        ↓
  Query → retrieve top 3 chunks → generate answer
        ↓
  Evaluate with LLM-as-judge (3 metrics)
```

---

## Features

- **GitHub URL or local path** — paste any public repo URL directly
- **Language-aware chunking** — splits Python at `def`/`class`, JS at `function`, Markdown at `#` headers — never breaks mid-function
- **15+ file types supported** — `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.cpp`, `.c`, `.cs`, `.rb`, `.md`, `.yaml`, `.sql`, `.toml`, and more
- **Per-repo caching** — each repo gets its own ChromaDB folder; second query on same repo is instant, no re-embedding
- **Source attribution** — every answer shows exactly which file and line numbers it came from
- **Hallucination scoring** — LLM-as-judge evaluates Context Relevance, Faithfulness, and Answer Relevance automatically

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/Krishnap1116/RepoMind.git

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
cp .env.example .env
# Edit .env and add your API key
```

---

## Usage

### Query a GitHub repo
```bash
python repomind.py \
  --repo https://github.com/pallets/flask \
  --query "How does error handling work?"
```

### Query a local codebase
```bash
python repomind.py \
  --repo ./my_project \
  --query "How is the database connection managed?"
```

### Second query on same repo (instant — no re-embedding)
```bash
python repomind.py \
  --repo https://github.com/pallets/flask \
  --query "How does Flask handle middleware?"
```

---

## Supported file types

| Category | Extensions |
|----------|-----------|
| Python | `.py` |
| JavaScript / TypeScript | `.js`, `.ts`, `.jsx`, `.tsx` |
| Systems languages | `.c`, `.cpp`, `.rs`, `.go` |
| JVM / .NET | `.java`, `.cs` |
| Web | `.html`, `.css` |
| Docs | `.md`, `.rst`, `.txt` |
| Config | `.yaml`, `.yml`, `.toml`, `.json`, `.ini`, `.cfg` |
| Data / Scripts | `.sql`, `.sh`, `.csv`, `.xml` |

---

## Evaluation metrics

Every answer is automatically scored by a judge LLM on 3 dimensions:

| Metric | Question it answers |
|--------|-------------------|
| **Context Relevance** | Was the retrieved chunk actually relevant to the query? |
| **Faithfulness** | Does the answer stay within the retrieved context (no hallucination)? |
| **Answer Relevance** | Does the answer actually address what was asked? |

Scores range from 0.0 to 1.0. Higher is better.

---

## Tech stack

| Component | Tool |
|-----------|------|
| Embeddings | OpenAI `text-embedding-3-large` |
| LLM | OpenAI `gpt-3.5-turbo` |
| Vector store | ChromaDB (persistent, local) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` with Language-aware splitting |
| Evaluation | LLM-as-judge pattern |

