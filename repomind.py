import os
import re
import json
import argparse
import subprocess
import shutil
import hashlib
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import cached_property, lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rank_bm25 import BM25Okapi
from tree_sitter_language_pack import get_parser
from dotenv import load_dotenv

load_dotenv()

# OpenAI model names by default; for OpenRouter set OPENAI_API_BASE and use ids like "openai/gpt-4.1-mini".
# These are the models the README results were measured with.
GEN_MODEL    = os.environ.get("GEN_MODEL", "gpt-4.1-mini")
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
# "local:" embeds on this machine with sentence-transformers (free, no rate limits); anything else goes to the API
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "local:BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXTENSION_TO_LANGUAGE = {
    ".py"  : Language.PYTHON,
    ".js"  : Language.JS,
    ".ts"  : Language.JS,
    ".jsx" : Language.JS,
    ".tsx" : Language.JS,
    ".java": Language.JAVA,
    ".go"  : Language.GO,
    ".rs"  : Language.RUST,
    ".cpp" : Language.CPP,
    ".c"   : Language.C,
    ".cs"  : Language.CSHARP,
    ".rb"  : Language.RUBY,
    ".md"  : Language.MARKDOWN,
    ".rst" : Language.RST,
    ".html": Language.HTML,
}

TEXT_EXTENSIONS = {
    ".txt", ".json", ".yaml", ".yml",
    ".toml", ".sql", ".sh", ".csv",
    ".xml", ".cfg", ".ini"
}

IGNORE_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".env",
               # build output and tool caches: generated copies of the source, not the source
               ".next", ".nuxt", ".svelte-kit", "dist", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
               "site-packages"}

# Syntax-aware ("ast") chunking: code and front-end files are parsed with tree-sitter
TREE_SITTER_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".kt": "kotlin", ".swift": "swift",
    ".html": "html", ".css": "css", ".scss": "scss", ".vue": "vue", ".svelte": "svelte",
}
# node types that name a unit worth labelling (function, class, method, React component via const, ...)
DEFINITION_TYPES = re.compile(r"function|method|class|interface|struct|enum|impl|trait|module|namespace"
                              r"|definition|declaration|declarator|constructor|export")
# syntax-aware chunks above this many embedder tokens are split: bge-base reads at most 512 tokens
# and silently drops the rest, and the header line needs room too
MAX_CHUNK_TOKENS = 450
# small-to-big: the whole function/class handed to the LLM can be larger than what was searched
PARENT_MAX_TOKENS = 1500

# every chunking mode reads the same files, so conditions differ only in how they are cut
SUPPORTED_EXTENSIONS = set(EXTENSION_TO_LANGUAGE) | TEXT_EXTENSIONS | set(TREE_SITTER_LANGUAGES)

SYMBOL_PATTERN = re.compile(r"^([ \t]*)(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE)


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    retriever: str = "dense"      # dense | hybrid (BM25 + dense fused with RRF)
    rerank: bool = False          # cross-encoder rerank of the candidate pool
    chunk_headers: bool = False   # prepend file/lines/symbol to each chunk before indexing
    chunking: str = "splitter"    # splitter (separator-based) | ast (tree-sitter, one chunk per definition)
    context: str = "chunk"        # chunk (LLM sees the retrieved chunks) | parent (their whole function/class)
    k_candidates: int = 20        # pool size fed to fusion / reranker
    k_final: int = 5              # chunks given to the LLM


# Ablation: each config differs from "headers" by one component, except baseline, which differs from dense_k5 by one
CONFIGS = {
    "baseline"     : RetrievalConfig("baseline", k_final=3),
    "dense_k5"     : RetrievalConfig("dense_k5"),
    "headers"      : RetrievalConfig("headers", chunk_headers=True),
    "ast"          : RetrievalConfig("ast", chunk_headers=True, chunking="ast"),
    # searches the headers index, then gives the LLM the whole definition each fragment belongs to
    "small_to_big" : RetrievalConfig("small_to_big", chunk_headers=True, context="parent"),
    # reranker without BM25, so its effect can be separated from hybrid's
    "headers_rerank": RetrievalConfig("headers_rerank", chunk_headers=True, rerank=True),
    "hybrid"       : RetrievalConfig("hybrid", retriever="hybrid", chunk_headers=True),
    "hybrid_rerank": RetrievalConfig("hybrid_rerank", retriever="hybrid", chunk_headers=True, rerank=True),
}


def require_api_key():
    # checked only where an API is called, so indexing, retrieval-only evals and chunk stats run without a key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Check your .env file.")


def make_chat_model(model, temperature=0):
    require_api_key()
    # retry rate-limited or failed calls with backoff instead of failing a long eval run
    return ChatOpenAI(model=model, temperature=temperature, max_retries=6,
                      base_url=os.environ.get("OPENAI_API_BASE"))


class LocalEmbeddings(Embeddings):

    # bge models retrieve better when queries carry this instruction
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = model_name   # RAGAS reads .model as the model's name
        self.encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=len(texts) > 100).tolist()

    def embed_query(self, text):
        return self.encoder.encode(self.QUERY_PREFIX + text, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def make_embeddings():
    if EMBED_MODEL.startswith("local:"):
        return LocalEmbeddings(EMBED_MODEL.removeprefix("local:"))
    require_api_key()
    # send raw strings: non-OpenAI embedding models can't take tiktoken ids
    return OpenAIEmbeddings(model=EMBED_MODEL, base_url=os.environ.get("OPENAI_API_BASE"),
                            check_embedding_ctx_length=False)


def is_github_url(repo):

    if repo.startswith("https://github.com") or repo.startswith("git@github.com"):
        return True
    return False


def get_repo_name(repo):
    if is_github_url(repo):
        return repo.rstrip("/").split("/")[-1].removesuffix(".git")
    return os.path.basename(os.path.abspath(repo))


def clone_repo(github_url, target_dir=os.path.join(BASE_DIR, "cloned_repo"), ref=None):

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    print(f"Cloning the Git Repo from {github_url}" + (f" @ {ref}" if ref else ""))

    # ref is a tag or branch, so the golden set always sees the same code
    command = ["git", "clone", "--depth=1"]
    if ref:
        command += ["--branch", ref]
    result= subprocess.run(
        command + [github_url, target_dir],
        capture_output=True,
        text=True
    )
    if result.returncode!=0:
        raise RuntimeError(f"Clone failed: {result.stderr}")

    print(f"Cloned to {target_dir}")
    return target_dir


def cleanup_repo(target_dir):
    print(f" Deleting the repo {target_dir}")
    shutil.rmtree(target_dir)

def get_splitter(extension):

    if extension in EXTENSION_TO_LANGUAGE:
        splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=512,
            chunk_overlap=64,
            language=EXTENSION_TO_LANGUAGE[extension]
        )
        return splitter
    else:
        # fallback for yaml, json, sql, toml, txt etc
        return RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
        )

CHANGELOG_NAMES = ("changes", "changelog", "history", "news")
CODE_EXTENSIONS = (set(EXTENSION_TO_LANGUAGE) | set(TREE_SITTER_LANGUAGES)) - {".md", ".rst", ".html"}


def chunk_kind(source):
    """Coarse file type of a chunk: code, tests, docs, changelog or other (config/data)."""
    parts = source.lower().split("/")
    name, extension = os.path.splitext(parts[-1])
    if name in CHANGELOG_NAMES:
        return "changelog"
    if any(part in ("tests", "test", "testing") for part in parts[:-1]) or name.startswith("test_"):
        return "tests"
    if extension in CODE_EXTENSIONS:
        return "code"
    if parts[0] in ("docs", "doc") or extension in (".md", ".rst", ".txt", ".html"):
        return "docs"
    return "other"


def get_line_numbers(full_text, chunk, search_from=0):

    # find the character index where the chunk starts, searching after the previous chunk
    # so repeated snippets get their own position
    index=full_text.find(chunk, search_from)
    if index==-1:
        index=full_text.find(chunk)
    if index==-1:
        return (0,0,-1)
    # we are counting number of lines from start to the index which gives starting lines of the chunk
    start_pos= full_text[:index].count("\n")+1
    # to get the end line of the chunk we count number of lines present in the chunk and add it to the starting line number which we found
    end_pos= start_pos+chunk.count("\n")

    return (start_pos,end_pos,index)


def enclosing_symbol(full_text, index):
    """Dotted name of the class/def a Python chunk lives in, e.g. 'Flask.wsgi_app'."""
    line_start = full_text.rfind("\n", 0, index) + 1
    line_end = full_text.find("\n", line_start)
    first_line = full_text[line_start:line_end if line_end != -1 else len(full_text)]
    chunk_indent = len(first_line) - len(first_line.lstrip())

    # stack of (indent, name) for def/class lines above the chunk
    stack = []
    for match in SYMBOL_PATTERN.finditer(full_text, 0, line_start):
        indent = len(match.group(1).expandtabs())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, match.group(2)))
    # drop scopes that already closed before the chunk's first line
    while stack and stack[-1][0] >= chunk_indent:
        stack.pop()
    own = SYMBOL_PATTERN.match(first_line)
    if own:
        stack.append((chunk_indent, own.group(2)))
    return ".".join(name for _, name in stack)


def node_name(node):
    """Name of a definition node (function, class, component, ...), looking through wrappers like
    `export`, decorators and `const App = () => ...`."""
    if not DEFINITION_TYPES.search(node.type):
        return None
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", "replace")
    for child in node.named_children:
        found = node_name(child)
        if found:
            return found
    return None


@lru_cache(maxsize=1)
def _token_counter():
    """Counts tokens with the embedding model's own tokenizer (tiktoken for API models)."""
    if EMBED_MODEL.startswith("local:"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL.removeprefix("local:"))
        return lambda text: len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text))


def count_tokens(text):
    return _token_counter()(text)


def split_code(text, language, max_tokens=MAX_CHUNK_TOKENS):
    """Syntax-aware chunks for any tree-sitter language.

    - Each definition (function, method, class, component) that fits in max_tokens is its own
      chunk, together with the comments directly above it. It is never cut and never merged with others.
    - A definition that is too big is split into its children, recursively (e.g. a class into methods).
    - Everything else (imports, constants, CSS rules, HTML elements) is merged with adjacent siblings
      up to the limit.
    - Only a single leaf longer than the limit (e.g. a huge string literal) is cut at line boundaries.
    """
    source = text.encode("utf-8")
    root = get_parser(language).parse(source).root_node
    lines = text.splitlines()
    chunks = []

    def tokens(start_byte, end_byte):
        return count_tokens(source[start_byte:end_byte].decode("utf-8", "replace"))

    def emit(start_row, end_row, symbol):
        chunk = "\n".join(lines[start_row:end_row + 1])
        if re.search(r"\w", chunk):   # skip chunks that are only braces or punctuation
            chunks.append((chunk, start_row + 1, end_row + 1, symbol))

    def emit_lines(start_row, end_row, symbol):
        piece_start, piece_size = start_row, 0
        for row in range(start_row, end_row + 1):
            line_size = count_tokens(lines[row]) + 1
            piece_size += line_size
            if piece_size > max_tokens and row > piece_start:
                emit(piece_start, row - 1, symbol)
                piece_start, piece_size = row, line_size
        emit(piece_start, end_row, symbol)

    def walk(node, scope):
        group = []   # adjacent non-definition siblings waiting to be emitted together

        def flush(nodes):
            if nodes:
                emit(nodes[0].start_point[0], nodes[-1].end_point[0], scope)

        for child in node.children:
            name = node_name(child)
            too_big = tokens(child.start_byte, child.end_byte) > max_tokens
            if name and not too_big:
                # comments directly above a definition travel with it
                lead = len(group)
                while lead and "comment" in group[lead - 1].type:
                    lead -= 1
                flush(group[:lead])
                start = group[lead].start_point[0] if lead < len(group) else child.start_point[0]
                emit(start, child.end_point[0], ".".join(filter(None, [scope, name])))
                group = []
            elif too_big:
                flush(group)
                group = []
                if child.child_count:
                    walk(child, ".".join(filter(None, [scope, name])))
                else:
                    emit_lines(child.start_point[0], child.end_point[0], scope)
            else:
                if group and tokens(group[0].start_byte, child.end_byte) > max_tokens:
                    flush(group)
                    group = []
                group.append(child)
        flush(group)

    walk(root, "")
    return chunks


def split_with_splitter(text, extension):
    """Separator-based chunks: every file in splitter mode, and files tree-sitter doesn't cover in ast mode."""
    chunks = []
    search_from = 0
    for chunk in get_splitter(extension).split_text(text):
        start_line, end_line, index = get_line_numbers(text, chunk, search_from)
        if index != -1:
            search_from = index + 1
        symbol = enclosing_symbol(text, index) if extension == ".py" and index != -1 else ""
        chunks.append((chunk, start_line, end_line, symbol))
    return chunks


def is_minified(text):
    """Minified/bundled files put many statements on one line; line-based syntax chunks would copy that
    whole line into every chunk, so these files use the character splitter instead."""
    return max(map(len, text.splitlines()), default=0) > 1000


def split_file(text, extension, chunking):
    if chunking == "ast" and extension in TREE_SITTER_LANGUAGES and not is_minified(text):
        return split_code(text, TREE_SITTER_LANGUAGES[extension])
    return split_with_splitter(text, extension)


def iter_repo_files(repo_path):
    """(source, extension, text) for every supported, readable file in the repo."""
    for root,dirs,files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            _,extension= os.path.splitext(file)
            if extension not in SUPPORTED_EXTENSIONS:
                continue
            filepath= os.path.join(root,file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            except (UnicodeDecodeError, PermissionError):
                print(f" Skipping unreadable file: {filepath}")
                continue
            yield os.path.relpath(filepath,repo_path), extension, text


def load_codebase(repo_path, chunk_headers=False, chunking="splitter"):

    documents=[]
    lang_count = Counter()
    total_chunks=0
    file_count = 0

    for source, extension, text in iter_repo_files(repo_path):
        file_count+=1
        lang_count[extension] += 1
        chunks = split_file(text, extension, chunking)
        total_chunks+=len(chunks)
        for chunk, start_line, end_line, symbol in chunks:
            lines = f"lines {start_line} - {end_line}"
            content = chunk
            if chunk_headers:
                # contextual header: gives both BM25 and embeddings the file path and symbol
                header = f"# File: {source} | {lines}"
                if symbol:
                    header += f" | {symbol}"
                content = f"{header}\n{chunk}"
            doc= Document(
                page_content=content,
                metadata={
                    "source":source,
                    "language":extension,
                    "lines":lines,
                    "chunk_id":f"{len(documents)}:{source}:{start_line}-{end_line}"
                }
            )
            documents.append(doc)

    print(f"Files loaded: {file_count}")
    print(f"Total chunks: {total_chunks}")
    print(f"languages: {dict(lang_count)}")
    return documents


def cache_dir(kind, repo, ref, suffix=""):
    """Cache folder for one version of a repo. A GitHub repo is named by its ref. A local folder is named by
    its full path, and its cache is cleared when a file that gets indexed was added, removed or edited."""
    if is_github_url(repo):
        tag = re.sub(r"[^\w.-]", "_", ref) if ref else "head"
        return os.path.join(BASE_DIR, f"{kind}_{get_repo_name(repo)}_{tag}{suffix}")
    path = os.path.abspath(repo)
    folder = os.path.join(BASE_DIR, f"{kind}_{get_repo_name(repo)}_{hashlib.sha1(path.encode()).hexdigest()[:8]}{suffix}")
    # only the files iter_repo_files reads count, so logs, .DS_Store or our own caches never force a rebuild
    fingerprint = hashlib.sha1()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for file in sorted(files):
            if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(root, file)
                fingerprint.update(f"{filepath}:{os.path.getmtime(filepath)}\n".encode())
    stamp_path = os.path.join(folder, "source_fingerprint")
    stamp = os.path.exists(stamp_path) and open(stamp_path).read()
    if os.path.exists(folder) and stamp != fingerprint.hexdigest():
        print(f"{path} changed since it was indexed. Rebuilding.")
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    with open(stamp_path, "w") as f:
        f.write(fingerprint.hexdigest())
    return folder


def build_or_load_index(repo, ref=None, chunk_headers=False, chunking="splitter"):
    """Return (vectorstore, chunks). Each (repo, ref, chunking, header mode, embedding model) gets its own cache folder."""
    mode = ("headers" if chunk_headers else "plain") + ("_ast" if chunking == "ast" else "")
    embed = re.sub(r"[^\w.-]", "_", EMBED_MODEL.split("/")[-1])
    persist_dir = cache_dir("repomind_db", repo, ref, f"_{mode}_{embed}")
    chunks_path = os.path.join(persist_dir, "chunks.jsonl")
    embeddings = make_embeddings()

    if os.path.exists(chunks_path):
        print(f"Cache found at {persist_dir}. Skipping clone + embed.")
        with open(chunks_path) as f:
            docs = [Document(**json.loads(line)) for line in f]
        vector_data = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        return vector_data, docs

    cloned = is_github_url(repo)
    repo_path = clone_repo(repo, ref=ref) if cloned else repo

    print("Loading codebase..")
    docs = load_codebase(repo_path, chunk_headers, chunking)
    if cloned:
        cleanup_repo(repo_path)

    print(f"Embedding {len(docs)} chunks")
    vector_data=Chroma.from_documents(
        docs,
        embeddings,
        ids=[doc.metadata["chunk_id"] for doc in docs],
        persist_directory= persist_dir
    )
    # BM25 needs the raw chunks at query time, so keep them next to the vector DB.
    # Written last and renamed into place: its existence means the whole index is complete.
    with open(chunks_path + ".tmp", "w") as f:
        for doc in docs:
            f.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}) + "\n")
    os.replace(chunks_path + ".tmp", chunks_path)
    print("Stored")
    return vector_data, docs


def build_or_load_parents(repo, ref=None):
    """Whole definitions (function/method/class/component) per code file, for small-to-big context.

    Same tree-sitter units as syntax-aware chunking, with a larger PARENT_MAX_TOKENS budget. Nothing is
    embedded, so this only needs a clone. Returns {source: [(start_line, end_line, symbol, text), ...]}.
    """
    path = os.path.join(cache_dir("repomind_parents", repo, ref), "parents.jsonl")
    if not os.path.exists(path):
        cloned = is_github_url(repo)
        repo_path = clone_repo(repo, ref=ref) if cloned else repo
        # renamed into place when complete, so an interrupted build is redone instead of reused
        with open(path + ".tmp", "w") as f:
            for source, extension, text in iter_repo_files(repo_path):
                if extension not in TREE_SITTER_LANGUAGES or is_minified(text):
                    continue
                for chunk, start, end, symbol in split_code(text, TREE_SITTER_LANGUAGES[extension], PARENT_MAX_TOKENS):
                    f.write(json.dumps({"source": source, "start": start, "end": end,
                                        "symbol": symbol, "text": chunk}) + "\n")
        os.replace(path + ".tmp", path)
        if cloned:
            cleanup_repo(repo_path)
    parents = {}
    with open(path) as f:
        for row in map(json.loads, f):
            parents.setdefault(row["source"], []).append((row["start"], row["end"], row["symbol"], row["text"]))
    return parents


def tokenize(text):
    """BM25 tokens for code: keep identifiers whole and also split snake_case / CamelCase."""
    tokens = []
    for word in re.findall(r"\w+", text):
        tokens.append(word.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.metadata["chunk_id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


class RAGPipeline:

    def __init__(self, repo, config, ref=None, exclude=()):
        self.config = config
        self.vectorstore, self.docs = build_or_load_index(repo, ref, config.chunk_headers, config.chunking)
        # exclude: glob patterns of files to leave out of retrieval, e.g. "CHANGES.rst" or "tests/*"
        excluded = sorted({doc.metadata["source"] for doc in self.docs
                           if any(fnmatch(doc.metadata["source"], pattern) for pattern in exclude)})
        self.dense_filter = {"source": {"$nin": excluded}} if excluded else None
        self.docs = [doc for doc in self.docs if doc.metadata["source"] not in set(excluded)]
        self.bm25 = None
        if config.retriever == "hybrid":
            self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.docs])
        self.reranker = None
        if config.rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)
        self.parents = build_or_load_parents(repo, ref) if config.context == "parent" else {}

    @cached_property
    def llm(self):
        # created on first answer, so retrieval-only runs need no API key
        return make_chat_model(GEN_MODEL)

    def retrieve(self, query, k=None):
        """Top k ranked chunks, best first (k defaults to config.k_final). contexts() turns them into what the LLM reads."""
        k = k or self.config.k_final
        pool = max(k, self.config.k_candidates)

        if self.bm25 is not None:
            dense = self.vectorstore.similarity_search(query, k=pool, filter=self.dense_filter)
            bm25_scores = self.bm25.get_scores(tokenize(query))
            top = sorted(range(len(self.docs)), key=lambda i: bm25_scores[i], reverse=True)[:pool]
            candidates = reciprocal_rank_fusion([dense, [self.docs[i] for i in top]])[:pool]
        else:
            candidates = self.vectorstore.similarity_search(query, k=pool if self.reranker else k,
                                                            filter=self.dense_filter)

        if self.reranker is not None:
            scores = self.reranker.predict([(query, doc.page_content) for doc in candidates])
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            candidates = [candidates[i] for i in order]
        return candidates[:k]

    def parent_of(self, chunk):
        """The whole definition a retrieved fragment starts in; the fragment itself if there is none."""
        source = chunk.metadata["source"]
        start_line = int(re.findall(r"\d+", chunk.metadata["lines"])[0])
        for start, end, symbol, text in self.parents.get(source, []):
            if start <= start_line <= end:
                lines = f"lines {start} - {end}"
                header = f"# File: {source} | {lines}" + (f" | {symbol}" if symbol else "")
                return Document(page_content=f"{header}\n{text}",
                                metadata=dict(chunk.metadata, lines=lines, chunk_id=f"parent:{source}:{start}-{end}"))
        return chunk

    def contexts(self, ranked):
        """What the LLM reads for a ranking: the chunks themselves, or their deduplicated parents."""
        if self.config.context != "parent":
            return ranked
        seen, parents = set(), []
        for chunk in ranked:
            parent = self.parent_of(chunk)
            if parent.metadata["chunk_id"] not in seen:
                seen.add(parent.metadata["chunk_id"])
                parents.append(parent)
        return parents

    def answer(self, query):
        # retrieve a full candidate pool so deduplicated parents still fill k_final slots
        chunks = self.contexts(self.retrieve(query, k=self.config.k_candidates))[:self.config.k_final]
        return generate_answer(query, chunks, self.llm), chunks


def format_chunk(chunk):
    # header-mode chunks already carry their file label
    if chunk.page_content.startswith("# File:"):
        return chunk.page_content
    return f"# File: {chunk.metadata['source']} | {chunk.metadata['lines']}\n{chunk.page_content}"


def generate_answer(query, chunks, llm, usage=None):
    """Answer from the chunks. If a usage Counter is given, adds the call's input/output tokens to it."""

    context="\n\n".join(format_chunk(chunk) for chunk in chunks)
    prompt=f"""
    You are an expert code assistant helping a developer understand a codebase.
    Given the following code snippets, answer the question clearly.
    - Mention the specific file and function where relevant
    - Be concise — developers want direct answers
    - If the code doesn't contain the answer, say so clearly

    Code context:
    {context}

    Question: {query}
    """
    response= llm.invoke(prompt)
    if usage is not None and response.usage_metadata:
        usage["input_tokens"] += response.usage_metadata["input_tokens"]
        usage["output_tokens"] += response.usage_metadata["output_tokens"]
    return response.content


def score_with_ragas(query, answer, chunks):
    """Reference-free RAGAS scores for a single answer (no golden answer needed)."""
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    judge = LangchainLLMWrapper(make_chat_model(JUDGE_MODEL))
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=[format_chunk(chunk) for chunk in chunks],
    )
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=LangchainEmbeddingsWrapper(make_embeddings())),
    ]
    return {metric.name: metric.single_turn_score(sample) for metric in metrics}


def main():

    parser=argparse.ArgumentParser(description="Codebase Q&A using RAG")
    parser.add_argument("--repo",  type=str, required=True,
                        help="Local path OR GitHub URL")
    parser.add_argument("--query", type=str, required=True,
                        help="Question about the codebase")
    parser.add_argument("--ref", type=str, default=None,
                        help="Git tag or branch to clone (GitHub URLs only)")
    # headers retrieval + whole-function context: most context recall in evaluation (see README "Results")
    parser.add_argument("--config", choices=CONFIGS, default="small_to_big",
                        help="Retrieval pipeline to use")
    parser.add_argument("--score", action="store_true",
                        help="Score the answer with RAGAS faithfulness + answer relevancy")
    args=parser.parse_args()

    pipeline = RAGPipeline(args.repo, CONFIGS[args.config], ref=args.ref)
    answer, chunks = pipeline.answer(args.query)

    print(f"The answer is :{answer}")
    print("Sources:")
    for chunk in chunks:
        print(f"  - {chunk.metadata['source']} ({chunk.metadata['lines']})")
    if args.score:
        for name, score in score_with_ragas(args.query, answer, chunks).items():
            print(f"  {name:<18}: {score:.3f}")


if __name__ == "__main__":
    main()
