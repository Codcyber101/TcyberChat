import pytest
import asyncio
from pathlib import Path

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

from src.services.document_service import DocumentService


@pytest.mark.asyncio
async def test_extract_pdf_content_with_pymupdf(tmp_path):
    if not HAS_FITZ:
        pytest.skip("PyMuPDF not available")

    # Create a simple PDF using PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    text = "Hello PyMuPDF test. The capital is Paris."
    page.insert_text((72, 72), text)
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()

    svc = DocumentService(db=None, upload_dir=str(tmp_path))
    content = await svc._extract_pdf_content(str(pdf_path))
    assert isinstance(content, str)
    assert "Hello PyMuPDF test" in content
