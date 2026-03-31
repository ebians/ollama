import io

import fitz  # PyMuPDF
from docx import Document


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text(filename: str, data: bytes) -> str:
    """ファイル形式に応じてテキストを抽出する"""
    ext = _get_ext(filename)
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    # デフォルト: プレーンテキスト
    return data.decode("utf-8")


def _get_ext(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""


def _extract_pdf(data: bytes) -> str:
    """PDFからテキストを抽出する"""
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _extract_docx(data: bytes) -> str:
    """Word(.docx)からテキストを抽出する"""
    doc = Document(io.BytesIO(data))
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n".join(paragraphs)
