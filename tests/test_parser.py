"""parser.py のテスト（Ollama / Tesseract 不要）"""
import pytest
from app.parser import extract_text, _get_ext, _extract_docx, _extract_xlsx, SUPPORTED_EXTENSIONS


class TestGetExt:
    def test_txt(self):
        assert _get_ext("readme.txt") == ".txt"

    def test_pdf(self):
        assert _get_ext("document.PDF") == ".pdf"

    def test_docx(self):
        assert _get_ext("report.docx") == ".docx"

    def test_xlsx(self):
        assert _get_ext("data.xlsx") == ".xlsx"

    def test_no_extension(self):
        assert _get_ext("noext") == ""

    def test_multiple_dots(self):
        assert _get_ext("my.file.name.pdf") == ".pdf"


class TestSupportedExtensions:
    def test_contains_all(self):
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS

    def test_unsupported(self):
        assert ".exe" not in SUPPORTED_EXTENSIONS
        assert ".jpg" not in SUPPORTED_EXTENSIONS


class TestExtractText:
    def test_plain_text_utf8(self):
        content = "これはテストです\nHello World"
        result = extract_text("test.txt", content.encode("utf-8"))
        assert "これはテストです" in result
        assert "Hello World" in result

    def test_plain_text_ascii(self):
        content = "Simple ASCII text"
        result = extract_text("test.txt", content.encode("utf-8"))
        assert result == "Simple ASCII text"

    def test_empty_text(self):
        result = extract_text("empty.txt", b"")
        assert result == ""


class TestExtractPdf:
    def test_text_pdf(self):
        """テキストレイヤーのあるPDFからテキスト抽出"""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "PDF test content", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        result = extract_text("test.pdf", pdf_bytes)
        assert "PDF test content" in result


class TestExtractDocx:
    def test_docx(self):
        """Word文書からテキスト抽出"""
        from docx import Document
        import io
        doc = Document()
        doc.add_paragraph("段落1のテキスト")
        doc.add_paragraph("段落2のテキスト")
        buf = io.BytesIO()
        doc.save(buf)

        result = extract_text("test.docx", buf.getvalue())
        assert "段落1のテキスト" in result
        assert "段落2のテキスト" in result


class TestExtractXlsx:
    def test_xlsx(self):
        """Excelファイルからテキスト抽出"""
        from openpyxl import Workbook
        import io
        wb = Workbook()
        ws = wb.active
        ws.title = "テストシート"
        ws["A1"] = "名前"
        ws["B1"] = "部署"
        ws["A2"] = "田中"
        ws["B2"] = "開発部"
        buf = io.BytesIO()
        wb.save(buf)

        result = extract_text("test.xlsx", buf.getvalue())
        assert "テストシート" in result
        assert "田中" in result
        assert "開発部" in result
