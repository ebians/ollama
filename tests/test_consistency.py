"""整合性チェッカーのテスト"""

import pytest
from app.consistency_checker import (
    run_consistency_checks,
    check_unit_consistency,
    check_number_consistency,
    check_term_definitions,
    CheckResult,
)


class TestUnitConsistency:
    def test_same_value_different_unit(self):
        chunks = [
            {"text": "ボルトの直径は10mm", "source": "spec_a.pdf"},
            {"text": "ボルトの直径は1cm", "source": "spec_b.pdf"},
        ]
        issues = check_unit_consistency(chunks)
        assert len(issues) >= 1
        assert issues[0].category == "unit"

    def test_no_inconsistency(self):
        chunks = [
            {"text": "重量は5kg", "source": "doc_a.pdf"},
            {"text": "長さは100mm", "source": "doc_b.pdf"},
        ]
        issues = check_unit_consistency(chunks)
        assert len(issues) == 0

    def test_currency_mixed(self):
        chunks = [
            {"text": "予算は100万円", "source": "budget.xlsx"},
            {"text": "予算は1000000円", "source": "report.pdf"},
        ]
        issues = check_unit_consistency(chunks)
        # 100万 = 1,000,000 → 同じ基準値、異なる単位
        assert len(issues) >= 1


class TestNumberConsistency:
    def test_different_values_same_key(self):
        chunks = [
            {"text": "納期は30日", "source": "contract.pdf"},
            {"text": "納期は45日", "source": "spec.pdf"},
        ]
        issues = check_number_consistency(chunks)
        assert len(issues) >= 1
        assert issues[0].category == "number"
        assert "納期" in issues[0].description

    def test_same_values(self):
        chunks = [
            {"text": "定員は100人", "source": "rule_a.pdf"},
            {"text": "定員は100人", "source": "rule_b.pdf"},
        ]
        issues = check_number_consistency(chunks)
        assert len(issues) == 0


class TestTermDefinitions:
    def test_different_definitions(self):
        chunks = [
            {"text": "「稼働率」とは月間稼働時間の割合を指す", "source": "glossary_a.pdf"},
            {"text": "「稼働率」とは年間稼働日数の比率である", "source": "glossary_b.pdf"},
        ]
        issues = check_term_definitions(chunks)
        assert len(issues) >= 1
        assert issues[0].category == "term"


class TestRunAllChecks:
    def test_combined(self):
        chunks = [
            {"text": "ボルト直径は10mm、納期は30日", "source": "spec_a.pdf"},
            {"text": "ボルト直径は1cm、納期は45日", "source": "spec_b.pdf"},
        ]
        result = run_consistency_checks(chunks)
        assert isinstance(result, CheckResult)
        assert len(result.inconsistencies) >= 1
        assert result.summary

    def test_empty_chunks(self):
        result = run_consistency_checks([])
        assert len(result.inconsistencies) == 0

    def test_to_context(self):
        chunks = [
            {"text": "納期は30日", "source": "a.pdf"},
            {"text": "納期は45日", "source": "b.pdf"},
        ]
        result = run_consistency_checks(chunks)
        ctx = result.to_context()
        assert "ルールベースチェック" in ctx
