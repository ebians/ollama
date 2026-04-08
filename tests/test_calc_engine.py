"""計算エンジンのテスト"""

import pytest
from app.calc_engine import (
    extract_tables_from_text,
    parse_number,
    compute_difference,
    compute_summary,
    format_structured_tables,
)


class TestParseNumber:
    def test_integer(self):
        assert parse_number("100") == 100.0

    def test_float(self):
        assert parse_number("3.14") == 3.14

    def test_comma_separated(self):
        assert parse_number("1,234,567") == 1_234_567.0

    def test_man_unit(self):
        assert parse_number("100万") == 1_000_000.0

    def test_oku_unit(self):
        assert parse_number("1.5億") == 150_000_000.0

    def test_negative_triangle(self):
        assert parse_number("▲30") == -30.0

    def test_negative_delta(self):
        assert parse_number("△20万") == -200_000.0

    def test_negative_hyphen(self):
        assert parse_number("-50") == -50.0

    def test_yen_suffix(self):
        assert parse_number("500円") == 500.0

    def test_percent_suffix(self):
        assert parse_number("12.5%") == 12.5

    def test_empty(self):
        assert parse_number("") is None

    def test_text_only(self):
        assert parse_number("テスト") is None

    def test_whitespace(self):
        assert parse_number("  100  ") == 100.0


class TestExtractTables:
    def test_single_table(self):
        text = (
            "説明文\n\n"
            "| 部門 | 売上 | 利益 |\n"
            "| --- | --- | --- |\n"
            "| A部門 | 1000万 | 200万 |\n"
            "| B部門 | 800万 | 150万 |\n"
            "\n後続テキスト"
        )
        tables = extract_tables_from_text(text)
        assert len(tables) == 1
        assert len(tables[0]) == 2
        assert tables[0][0]["部門"] == "A部門"
        assert tables[0][0]["売上"] == "1000万"
        assert tables[0][1]["利益"] == "150万"

    def test_multiple_tables(self):
        text = (
            "| 月 | 値 |\n| --- | --- |\n| 1月 | 100 |\n\n"
            "中間テキスト\n\n"
            "| 年 | 売上 |\n| --- | --- |\n| 2024 | 500 |\n| 2025 | 600 |\n"
        )
        tables = extract_tables_from_text(text)
        assert len(tables) == 2

    def test_no_table(self):
        text = "これはテーブルのないテキストです。"
        tables = extract_tables_from_text(text)
        assert len(tables) == 0


class TestComputeDifference:
    def test_basic(self):
        result = compute_difference(150.0, 100.0)
        assert result["difference"] == 50.0
        assert result["ratio_percent"] == 150.0
        assert result["change_rate_percent"] == 50.0

    def test_negative_diff(self):
        result = compute_difference(80.0, 100.0)
        assert result["difference"] == -20.0
        assert result["change_rate_percent"] == -20.0


class TestComputeSummary:
    def test_basic(self):
        result = compute_summary([10.0, 20.0, 30.0])
        assert result["count"] == 3
        assert result["sum"] == 60.0
        assert result["average"] == 20.0
        assert result["max"] == 30.0
        assert result["min"] == 10.0

    def test_empty(self):
        result = compute_summary([])
        assert result == {}


class TestFormatStructuredTables:
    def test_format(self):
        tables = [
            [
                {"部門": "A", "売上": "100万"},
                {"部門": "B", "売上": "200万"},
            ]
        ]
        result = format_structured_tables(tables)
        assert "表1" in result
        assert "A" in result
        assert "合計" in result  # 数値列の統計が含まれる
