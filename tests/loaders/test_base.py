"""Tests for loaders/_base.py helpers."""

from __future__ import annotations


import pandas as pd

from lerobot_isaac_dashboard.loaders._base import (
    LoaderResult,
    _align_to_schema,
    empty_df,
    glob_runs,
    safe_read_json,
    safe_read_jsonl,
    safe_read_parquet,
)


# ---------------------------------------------------------------------------
# empty_df
# ---------------------------------------------------------------------------


def test_empty_df_has_correct_columns():
    cols = ["a", "b", "c"]
    dtypes = {"a": "string", "b": "Int64", "c": "Float64"}
    df = empty_df(cols, dtypes)
    assert list(df.columns) == cols
    assert len(df) == 0


def test_empty_df_extra_columns_in_dtypes_ignored():
    """Columns in dtypes but not in columns list are silently ignored."""
    cols = ["a"]
    dtypes = {"a": "string", "extra": "Int64"}
    df = empty_df(cols, dtypes)
    assert list(df.columns) == ["a"]


# ---------------------------------------------------------------------------
# _align_to_schema
# ---------------------------------------------------------------------------


def test_align_adds_missing_columns():
    schema = {"a": "string", "b": "Int64"}
    df = pd.DataFrame({"a": ["x"]})
    aligned, warnings = _align_to_schema(df, schema)
    assert "b" in aligned.columns
    assert any("b" in w for w in warnings)


def test_align_preserves_extra_columns():
    schema = {"a": "string"}
    df = pd.DataFrame({"a": ["x"], "extra": [1]})
    aligned, _ = _align_to_schema(df, schema)
    assert "extra" in aligned.columns


def test_align_schema_cols_come_first():
    schema = {"a": "string", "b": "Int64"}
    df = pd.DataFrame({"z": [1], "a": ["x"], "b": [2]})
    aligned, _ = _align_to_schema(df, schema)
    assert list(aligned.columns[:2]) == ["a", "b"]


# ---------------------------------------------------------------------------
# safe_read_parquet
# ---------------------------------------------------------------------------


def test_safe_read_parquet_missing(tmp_path):
    result = safe_read_parquet(tmp_path / "nonexistent.parquet")
    assert result is None


def test_safe_read_parquet_valid(tmp_path):
    pq_path = tmp_path / "test.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(pq_path)
    df = safe_read_parquet(pq_path)
    assert df is not None
    assert list(df["x"]) == [1, 2]


def test_safe_read_parquet_corrupt(tmp_path):
    corrupt = tmp_path / "bad.parquet"
    corrupt.write_bytes(b"not a parquet file at all")
    result = safe_read_parquet(corrupt)
    assert result is None


# ---------------------------------------------------------------------------
# safe_read_jsonl
# ---------------------------------------------------------------------------


def test_safe_read_jsonl_missing(tmp_path):
    result = safe_read_jsonl(tmp_path / "nonexistent.jsonl")
    assert result is None


def test_safe_read_jsonl_valid(tmp_path):
    jl = tmp_path / "data.jsonl"
    jl.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    df = safe_read_jsonl(jl)
    assert df is not None
    assert len(df) == 2
    assert list(df["a"]) == [1, 2]


def test_safe_read_jsonl_skips_bad_lines(tmp_path):
    jl = tmp_path / "mixed.jsonl"
    jl.write_text('{"a": 1}\nNOT JSON\n{"a": 3}\n', encoding="utf-8")
    df = safe_read_jsonl(jl)
    assert df is not None
    assert len(df) == 2


# ---------------------------------------------------------------------------
# safe_read_json
# ---------------------------------------------------------------------------


def test_safe_read_json_missing(tmp_path):
    assert safe_read_json(tmp_path / "nope.json") is None


def test_safe_read_json_valid(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text('{"key": "val"}', encoding="utf-8")
    data = safe_read_json(p)
    assert data == {"key": "val"}


def test_safe_read_json_corrupt(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json", encoding="utf-8")
    assert safe_read_json(p) is None


# ---------------------------------------------------------------------------
# glob_runs
# ---------------------------------------------------------------------------


def test_glob_runs_missing_root(tmp_path):
    result = glob_runs(tmp_path / "nonexistent", "*.txt")
    assert result == []


def test_glob_runs_finds_files(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    result = glob_runs(tmp_path, "*.txt")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# LoaderResult
# ---------------------------------------------------------------------------


def test_loader_result_defaults():
    df = pd.DataFrame()
    result = LoaderResult(df=df, is_empty=True)
    assert result.source_paths == []
    assert result.warnings == []
