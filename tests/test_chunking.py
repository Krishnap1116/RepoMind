import repomind
from repomind import chunk_kind, split_code, split_file

PYTHON = '''import os

# loads settings
def load(path):
    return open(path).read()


class Store:
    """Keeps items."""

    def add(self, item):
        self.items.append(item)

    def clear(self):
        self.items = []
'''

TSX = '''import React from "react";

// Shows the cart total
export const Cart = ({ items }) => {
  return <div>{items.length}</div>;
};

export default function App() {
  return <Cart items={[]} />;
}
'''


def symbols(chunks):
    return [symbol for _, _, _, symbol in chunks]


def test_each_python_definition_is_its_own_chunk_with_its_comment():
    chunks = split_code(PYTHON, "python")
    assert symbols(chunks) == ["", "load", "Store"]
    load_chunk = chunks[1][0]
    assert load_chunk.startswith("# loads settings") and "return open(path).read()" in load_chunk


def test_class_over_budget_is_split_into_methods_not_mid_method():
    chunks = split_code(PYTHON, "python", max_tokens=25)
    by_symbol = {symbol: chunk for chunk, _, _, symbol in chunks}
    assert "Store.add" in by_symbol and "Store.clear" in by_symbol
    assert "self.items.append(item)" in by_symbol["Store.add"]
    assert "self.items = []" in by_symbol["Store.clear"]


def test_react_components_are_separate_chunks():
    chunks = split_code(TSX, "tsx")
    assert symbols(chunks) == ["", "Cart", "App"]
    assert chunks[1][0].startswith("// Shows the cart total")


def test_line_numbers_point_at_the_definition():
    chunks = split_code(PYTHON, "python")
    _, start, end, symbol = chunks[1]
    lines = PYTHON.splitlines()
    assert symbol == "load"
    assert lines[start - 1] == "# loads settings" and lines[end - 1].strip() == "return open(path).read()"


def test_splitter_mode_and_unsupported_files_fall_back_to_character_chunks():
    assert all(len(chunk) <= 512 for chunk, *_ in split_file(PYTHON * 5, ".py", "splitter"))
    assert split_file("# Title\n\nSome docs.", ".md", "ast")[0][0].startswith("# Title")


def test_splitter_chunks_get_correct_lines_and_enclosing_symbol():
    def body(indent):
        return "\n".join(f"{indent}x{i} = compute_something_long({i})" for i in range(30))
    text = f"class Store:\n    def add(self, item):\n{body(' ' * 8)}\n\n\ndef top_level():\n{body(' ' * 4)}\n"
    lines = text.splitlines()
    chunks = split_file(text, ".py", "splitter")
    for chunk, start, end, _ in chunks:
        assert lines[start - 1].strip() == chunk.splitlines()[0].strip()
        assert lines[end - 1].strip() == chunk.splitlines()[-1].strip()
    symbols = [symbol for *_, symbol in chunks]
    assert symbols[0] == "Store" and symbols[-1] == "top_level"
    assert symbols == sorted(symbols, key=["Store", "Store.add", "top_level"].index)
    assert "Store.add" in symbols


def test_minified_file_is_not_copied_into_every_chunk():
    minified = ";".join(f"var a{i}=function(){{return {i}}}" for i in range(200))
    chunks = split_file(minified, ".js", "ast")
    assert sum(len(chunk) for chunk, *_ in chunks) < 2 * len(minified)


def test_chunk_kind_labels():
    assert chunk_kind("src/flask/app.py") == "code"
    assert chunk_kind("tests/test_basic.py") == "tests"
    assert chunk_kind("docs/quickstart.rst") == "docs"
    assert chunk_kind("CHANGES.rst") == "changelog"
    assert chunk_kind("HISTORY.md") == "changelog"
    assert chunk_kind(".github/workflows/tests.yaml") == "other"


def test_budget_is_counted_in_tokens():
    assert repomind.count_tokens("def add(a, b):\n    return a + b") < 20
