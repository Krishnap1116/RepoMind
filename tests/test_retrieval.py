from langchain_core.documents import Document

from repomind import CONFIGS, RAGPipeline, reciprocal_rank_fusion, tokenize


def doc(chunk_id, source="src/app.py", lines="lines 1 - 2"):
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id, "source": source, "lines": lines})


def test_tokenize_keeps_identifiers_and_splits_them():
    tokens = tokenize("add_url_rule SecureCookieSession")
    assert "add_url_rule" in tokens and {"add", "url", "rule"} <= set(tokens)
    assert {"securecookiesession", "secure", "cookie", "session"} <= set(tokens)


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [d.metadata["chunk_id"] for d in fused] == ["b", "a", "c"]


def pipeline_with_parents(parents):
    pipeline = RAGPipeline.__new__(RAGPipeline)   # no index or API needed for context mapping
    pipeline.config = CONFIGS["small_to_big"]
    pipeline.parents = parents
    return pipeline


def test_fragments_map_to_their_whole_function_and_are_deduplicated():
    pipeline = pipeline_with_parents({"src/app.py": [(10, 30, "App.run", "def run(self):\n    ...")]})
    ranked = [doc("f1", lines="lines 12 - 18"), doc("f2", lines="lines 19 - 25"),
              doc("d1", source="docs/run.rst", lines="lines 1 - 5")]
    contexts = pipeline.contexts(ranked)
    assert len(contexts) == 2
    assert contexts[0].page_content.startswith("# File: src/app.py | lines 10 - 30 | App.run")
    assert contexts[1].metadata["chunk_id"] == "d1"   # docs pass through unchanged


def test_chunk_mode_returns_ranking_unchanged():
    pipeline = pipeline_with_parents({})
    pipeline.config = CONFIGS["headers"]
    ranked = [doc("a"), doc("b")]
    assert pipeline.contexts(ranked) == ranked
