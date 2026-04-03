"""vectorstore.py のチャンク分割テスト（Ollama不要）"""
import pytest
from app.vectorstore import _split_text


class TestSplitText:
    def test_short_text(self):
        """CHUNK_SIZE以下のテキストは1チャンク"""
        text = "短いテキスト"
        chunks = _split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "短いテキスト"

    def test_paragraph_split(self):
        """空行で段落分割される"""
        text = "段落1の内容です。\n\n段落2の内容です。"
        chunks = _split_text(text)
        assert len(chunks) >= 1
        full = "".join(chunks)
        assert "段落1" in full
        assert "段落2" in full

    def test_markdown_heading_split(self):
        """Markdown見出しで分割される"""
        text = "# 見出し1\n内容1\n\n# 見出し2\n内容2"
        chunks = _split_text(text)
        full = " ".join(chunks)
        assert "見出し1" in full
        assert "見出し2" in full

    def test_long_text_chunked(self):
        """長いテキストが複数チャンクに分割される"""
        text = "あ" * 2000
        chunks = _split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 600  # CHUNK_SIZE(500) + 余裕

    def test_numbered_heading_split(self):
        """番号付き見出しで分割される"""
        text = "1. 第一章\n内容A\n\n2. 第二章\n内容B"
        chunks = _split_text(text)
        full = " ".join(chunks)
        assert "第一章" in full
        assert "第二章" in full

    def test_empty_text(self):
        """空文字列は空リスト"""
        assert _split_text("") == []

    def test_whitespace_only(self):
        """空白のみは空リスト"""
        assert _split_text("   \n\n  ") == []
