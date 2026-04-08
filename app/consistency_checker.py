"""文書間の整合性チェックモジュール: ルールベース検証 + LLMクロスリファレンス"""

import re
from dataclasses import dataclass, field


@dataclass
class Inconsistency:
    """検出された不一致"""
    category: str           # "unit" | "number" | "term" | "date" | "requirement"
    severity: str           # "error" | "warning" | "info"
    description: str        # 日本語の説明
    source_a: str = ""      # 文書名A
    source_b: str = ""      # 文書名B
    text_a: str = ""        # 該当テキストA
    text_b: str = ""        # 該当テキストB


@dataclass
class CheckResult:
    """整合性チェック結果"""
    inconsistencies: list[Inconsistency] = field(default_factory=list)
    summary: str = ""

    def to_context(self) -> str:
        """LLMに渡す前のコンテキスト文字列を生成"""
        if not self.inconsistencies:
            return "ルールベースチェックでは不一致は検出されませんでした。"
        lines = ["## ルールベースチェック結果\n"]
        for i, inc in enumerate(self.inconsistencies, 1):
            icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(inc.severity, "⚪")
            lines.append(f"{i}. {icon} [{inc.category}] {inc.description}")
            if inc.source_a:
                lines.append(f"   文書A: {inc.source_a}")
            if inc.source_b:
                lines.append(f"   文書B: {inc.source_b}")
            if inc.text_a:
                lines.append(f"   抜粋A: {inc.text_a[:100]}")
            if inc.text_b:
                lines.append(f"   抜粋B: {inc.text_b[:100]}")
            lines.append("")
        return "\n".join(lines)


# ---- 単位パターン ----
_UNIT_PATTERNS = {
    "length": [
        (re.compile(r"(\d[\d,.]*)\s*mm\b", re.I), "mm", 1.0),
        (re.compile(r"(\d[\d,.]*)\s*cm\b", re.I), "cm", 10.0),
        (re.compile(r"(\d[\d,.]*)\s*m(?!m)\b", re.I), "m", 1000.0),
        (re.compile(r"(\d[\d,.]*)\s*μm\b", re.I), "μm", 0.001),
    ],
    "weight": [
        (re.compile(r"(\d[\d,.]*)\s*mg\b", re.I), "mg", 0.001),
        (re.compile(r"(\d[\d,.]*)\s*g(?!b)\b", re.I), "g", 1.0),
        (re.compile(r"(\d[\d,.]*)\s*kg\b", re.I), "kg", 1000.0),
        (re.compile(r"(\d[\d,.]*)\s*t(?:on)?\b", re.I), "t", 1_000_000.0),
    ],
    "currency": [
        (re.compile(r"(\d[\d,.]*)\s*円"), "円", 1.0),
        (re.compile(r"(\d[\d,.]*)\s*万円"), "万円", 10_000.0),
        (re.compile(r"(\d[\d,.]*)\s*億円"), "億円", 100_000_000.0),
    ],
}

# ---- 日付パターン ----
_DATE_PATTERNS = [
    re.compile(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})[日]?"),
    re.compile(r"令和(\d{1,2})年(\d{1,2})月(\d{1,2})日"),
    re.compile(r"R(\d{1,2})\.(\d{1,2})\.(\d{1,2})"),
]

# ---- 用語定義パターン ----
_TERM_DEF_PATTERN = re.compile(
    r"[「『]([^」』]{2,20})[」』]\s*(?:とは|は|を|：|:)\s*(.{5,100})",
)


def check_unit_consistency(chunks: list[dict]) -> list[Inconsistency]:
    """異なる文書間で同じ物理量が異なる単位系で記述されているかチェック"""
    results: list[Inconsistency] = []
    # 各文書から単位付き数値を抽出
    source_values: dict[str, list[tuple[str, str, float, float, str]]] = {}
    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        for category, patterns in _UNIT_PATTERNS.items():
            for pat, unit, to_base in patterns:
                for match in pat.finditer(text):
                    raw = match.group(1).replace(",", "")
                    try:
                        value = float(raw) * to_base
                    except ValueError:
                        continue
                    source_values.setdefault(source, []).append(
                        (category, unit, float(raw), value, match.group(0))
                    )

    # 異なるソース間で同じカテゴリの値を比較
    sources = list(source_values.keys())
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            vals_a = source_values[sources[i]]
            vals_b = source_values[sources[j]]
            for cat_a, unit_a, raw_a, base_a, text_a in vals_a:
                for cat_b, unit_b, raw_b, base_b, text_b in vals_b:
                    if cat_a != cat_b:
                        continue
                    # 同じ基準値なのに異なる単位表記
                    if unit_a != unit_b and abs(base_a - base_b) < base_a * 0.01:
                        results.append(Inconsistency(
                            category="unit",
                            severity="warning",
                            description=f"同一値が異なる単位で表記: {text_a} vs {text_b}",
                            source_a=sources[i],
                            source_b=sources[j],
                            text_a=text_a,
                            text_b=text_b,
                        ))
    return results


def check_number_consistency(chunks: list[dict]) -> list[Inconsistency]:
    """同じキーワードに紐づく数値が文書間で異なるかチェック"""
    results: list[Inconsistency] = []
    # キーワード+数値のペアを抽出
    keyword_values: dict[str, list[tuple[str, str, str]]] = {}
    pattern = re.compile(r"([^\d\s]{2,10})\s*[はが:：]\s*(\d[\d,.万億兆]*)")
    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        for m in pattern.finditer(text):
            key = m.group(1).strip()
            val = m.group(2).strip()
            keyword_values.setdefault(key, []).append((val, source, m.group(0)))

    for key, entries in keyword_values.items():
        if len(entries) < 2:
            continue
        # 同じキーワードで異なる値を検出
        seen_from_sources: dict[str, list[tuple[str, str]]] = {}
        for val, source, text in entries:
            seen_from_sources.setdefault(source, []).append((val, text))
        sources = list(seen_from_sources.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                for val_a, text_a in seen_from_sources[sources[i]]:
                    for val_b, text_b in seen_from_sources[sources[j]]:
                        if val_a != val_b:
                            results.append(Inconsistency(
                                category="number",
                                severity="error",
                                description=f"「{key}」の値が文書間で不一致: {val_a} vs {val_b}",
                                source_a=sources[i],
                                source_b=sources[j],
                                text_a=text_a,
                                text_b=text_b,
                            ))
    return results


def check_term_definitions(chunks: list[dict]) -> list[Inconsistency]:
    """用語定義の矛盾をチェック"""
    results: list[Inconsistency] = []
    term_defs: dict[str, list[tuple[str, str]]] = {}
    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        for m in _TERM_DEF_PATTERN.finditer(text):
            term = m.group(1)
            defn = m.group(2).strip()
            term_defs.setdefault(term, []).append((defn, source))

    for term, defs in term_defs.items():
        if len(defs) < 2:
            continue
        # 異なるソースでの定義を比較
        seen: dict[str, str] = {}
        for defn, source in defs:
            if source in seen:
                continue
            for prev_source, prev_defn in seen.items():
                if defn != prev_defn:
                    results.append(Inconsistency(
                        category="term",
                        severity="warning",
                        description=f"「{term}」の定義が文書間で異なる可能性",
                        source_a=prev_source,
                        source_b=source,
                        text_a=prev_defn[:100],
                        text_b=defn[:100],
                    ))
            seen[source] = defn
    return results


def run_consistency_checks(chunks: list[dict]) -> CheckResult:
    """全ルールベースチェックを実行する"""
    all_issues: list[Inconsistency] = []
    all_issues.extend(check_unit_consistency(chunks))
    all_issues.extend(check_number_consistency(chunks))
    all_issues.extend(check_term_definitions(chunks))

    # 重複排除（descriptionベース）
    seen_descs: set[str] = set()
    unique_issues: list[Inconsistency] = []
    for issue in all_issues:
        if issue.description not in seen_descs:
            seen_descs.add(issue.description)
            unique_issues.append(issue)

    errors = sum(1 for i in unique_issues if i.severity == "error")
    warnings = sum(1 for i in unique_issues if i.severity == "warning")
    summary = f"検出: {errors}件のエラー, {warnings}件の警告"

    return CheckResult(inconsistencies=unique_issues, summary=summary)
