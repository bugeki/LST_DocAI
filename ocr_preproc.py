# OCR and preprocessing utilities for LST-DocAI
import re
import sys
from pathlib import Path

DATE_RE = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")

def txt_to_text(path):
    text = Path(path).read_text(encoding='utf-8')
    return text

def clean_text(text):
    t = re.sub(r"\s+", " ", text).strip()
    t = t.replace('\u2019', "'")
    return t

def process_path(path):
    """
    Minimal demo: if a text file is provided, return cleaned text.
    For PDFs/images, you should integrate pdf2image + pytesseract.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in ['.txt']:
        raw = txt_to_text(p)
    else:
        # In POC keep simple: read as text for now
        raw = txt_to_text(p)
    return clean_text(raw)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ocr_preproc.py sample_data/sample_doc.txt')
    else:
        out = process_path(sys.argv[1])
        print(out)
