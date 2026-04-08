"""Markdownテーブルから構造化データを抽出し、数値計算を支援するモジュール"""

import re
from typing import Any


def extract_tables_from_text(text: str) -> list[list[dict[str, str]]]:
    """テキスト内のMarkdownテーブルを検出し、構造化データとして返す。

    Returns:
        [
            [{"col1": "val", "col2": "val"}, ...],  # テーブル1の行リスト
            ...
        ]
    """
    tables: list[list[dict[str, str]]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Markdownテーブルのヘッダー行を検出
        if line.startswith("|") and line.endswith("|"):
            header_cells = _parse_row(line)
            if not header_cells:
                i += 1
                continue
            # 次の行がセパレーター行か確認
            if i + 1 < len(lines) and _is_separator(lines[i + 1].strip()):
                i += 2  # ヘッダー + セパレーターをスキップ
                rows: list[dict[str, str]] = []
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not row_line.startswith("|"):
                        break
                    cells = _parse_row(row_line)
                    if cells:
                        row = {}
                        for j, header in enumerate(header_cells):
                            row[header] = cells[j] if j < len(cells) else ""
                        rows.append(row)
                    i += 1
                if rows:
                    tables.append(rows)
                continue
        i += 1
    return tables


def _parse_row(line: str) -> list[str]:
    """Markdownテーブル行をパースしてセルのリストを返す"""
    if not line.startswith("|") or not line.endswith("|"):
        return []
    cells = line[1:-1].split("|")
    return [c.strip() for c in cells]


def _is_separator(line: str) -> bool:
    """セパレーター行（| --- | --- |）かどうかを判定"""
    if not line.startswith("|") or not line.endswith("|"):
        return False
    return bool(re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|$", line))


def parse_number(text: str) -> float | None:
    """日本語を含む数値文字列をfloatに変換する。

    対応形式: "100", "1,234", "1.5", "100万", "1.2億", "-50", "▲30", "△20"
    """
    if not text:
        return None
    text = text.strip()

    # 符号の処理 (▲/△ = マイナス)
    negative = False
    if text.startswith(("▲", "△", "−", "ー")):
        negative = True
        text = text[1:].strip()
    elif text.startswith("-"):
        negative = True
        text = text[1:].strip()

    # カンマ除去
    text = text.replace(",", "")

    # 日本語単位の処理
    multiplier = 1.0
    jp_units = {"万": 10_000, "億": 100_000_000, "兆": 1_000_000_000_000}
    for unit, mult in jp_units.items():
        if unit in text:
            text = text.replace(unit, "")
            multiplier = mult
            break

    # 単位除去（円、%、個、人 など）
    text = re.sub(r"[円%％個人件台本枚回年月日時分秒kKmMgGtTbB].*$", "", text).strip()

    if not text:
        return None

    try:
        value = float(text) * multiplier
        return -value if negative else value
    except ValueError:
        return None


def compute_difference(a: float, b: float) -> dict[str, float]:
    """二値の差分・比率を計算する"""
    diff = a - b
    ratio = (a / b * 100) if b != 0 else float("inf")
    change_rate = ((a - b) / abs(b) * 100) if b != 0 else float("inf")
    return {
        "a": a,
        "b": b,
        "difference": round(diff, 4),
        "ratio_percent": round(ratio, 2),
        "change_rate_percent": round(change_rate, 2),
    }


def compute_summary(values: list[float]) -> dict[str, float]:
    """数値リストの集計統計を計算する"""
    if not values:
        return {}
    return {
        "count": len(values),
        "sum": round(sum(values), 4),
        "average": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
        "min": round(min(values), 4),
    }


def format_structured_tables(tables: list[list[dict[str, str]]]) -> str:
    """構造化テーブルデータをLLMが処理しやすいテキスト形式に変換する"""
    parts: list[str] = []
    for idx, table in enumerate(tables, 1):
        if not table:
            continue
        headers = list(table[0].keys())
        lines = [f"### 表{idx}"]
        lines.append("列: " + ", ".join(headers))
        lines.append("")

        # 数値検出・型情報を追加
        for row_idx, row in enumerate(table, 1):
            row_parts = []
            for h in headers:
                val = row.get(h, "")
                num = parse_number(val)
                if num is not None:
                    row_parts.append(f"{h}={val} (数値: {num})")
                else:
                    row_parts.append(f"{h}={val}")
            lines.append(f"行{row_idx}: " + " | ".join(row_parts))

        # 列ごとの数値集計を追加
        for h in headers:
            nums = []
            for row in table:
                n = parse_number(row.get(h, ""))
                if n is not None:
                    nums.append(n)
            if nums and len(nums) >= 2:
                stats = compute_summary(nums)
                lines.append(f"\n【{h}の統計】合計={stats['sum']}, 平均={stats['average']}, 最大={stats['max']}, 最小={stats['min']}")

        parts.append("\n".join(lines))
    return "\n\n".join(parts)
