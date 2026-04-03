import io
import os

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
from openpyxl import load_workbook

# Tesseractのパス設定
_tesseract_cmd = os.getenv("TESSERACT_CMD", r"D:\tools\Tesseract\tesseract.exe")
if os.path.exists(_tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".xlsx"}


def extract_text(filename: str, data: bytes) -> str:
    """ファイル形式に応じてテキストを抽出する"""
    ext = _get_ext(filename)
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    if ext == ".xlsx":
        return _extract_xlsx(data)
    # デフォルト: プレーンテキスト
    return data.decode("utf-8")


def _get_ext(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""


def _extract_pdf(data: bytes) -> str:
    """PDFからテキストを抽出する（テキストが少ない場合OCRにフォールバック）"""
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        text = page.get_text().strip()
        if len(text) < 20:
            # テキストが少ない → 画像として OCR
            text = _ocr_page(page)
        pages.append(text)
    doc.close()
    return "\n".join(pages)


def _ocr_page(page) -> str:
    """PDFページを画像化してOCRでテキスト抽出する"""
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = pytesseract.image_to_string(img, lang="jpn+eng")
        return text.strip()
    except Exception:
        return ""


def _extract_docx(data: bytes) -> str:
    """Word(.docx)からテキストを抽出する"""
    doc = Document(io.BytesIO(data))
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n".join(paragraphs)


def _extract_xlsx(data: bytes) -> str:
    """Excel(.xlsx)からテキストを抽出する（シートごとに見出し付き）"""
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet in wb.worksheets:
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            line = "\t".join(cells).strip()
            if line:
                rows.append(line)
        if rows:
            parts.append(f"## シート: {sheet.title}\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts)
