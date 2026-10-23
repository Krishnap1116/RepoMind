import pandas as pd
import pytest

from run_eval import kind_metrics, retrieval_metrics, uncertainty


def test_retrieval_metrics_are_file_level():
    sources = ["docs/a.rst", "src/x.py", "src/gold.py", "src/gold.py"]
    metrics = retrieval_metrics(sources, ["src/gold.py"])
    assert metrics["hit@1"] == 0.0 and metrics["hit@5"] == 1.0
    assert metrics["mrr@10"] == pytest.approx(1 / 3)
    assert metrics["recall@5"] == 1.0


def test_code_only_ranking_drops_non_code_chunks():
    metrics = kind_metrics(["docs/a.rst", "CHANGES.rst", "src/gold.py"], ["src/gold.py"])
    assert metrics["noncode@1"] == 1.0
    assert metrics["code_hit@1"] == 1.0


def test_paired_difference_is_exact_when_every_question_moves_equally():
    df = pd.DataFrame({
        "id": ["q1", "q2", "q3"] * 2,
        "type": ["explain"] * 6,
        "config": ["headers"] * 3 + ["other"] * 3,
        "hit@1": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        "mrr@10": [0.5, 0.5, 0.5, 0.75, 0.75, 0.75],
    })
    row = uncertainty(df).set_index("config").loc["other"]
    assert row["Δmrr@10 vs headers"] == "+0.250 [+0.25, +0.25]"
