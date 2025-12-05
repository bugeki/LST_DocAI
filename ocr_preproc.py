"""
OCR and preprocessing utilities for LST-DocAI

Supports:
- Text files (.txt)
- PDF files (.pdf) - converts to images and runs OCR
- Image files (.png, .jpg, .jpeg) - runs OCR directly

Requirements:
- pytesseract (pip install pytesseract)
- pdf2image (pip install pdf2image)
- Tesseract OCR installed on system
  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-tur
  - macOS: brew install tesseract tesseract-lang
"""

import re
import sys
from pathlib import Path
from typing import Optional

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("WARNING: pytesseract or PIL not installed. OCR features will be disabled.")
    print("Install with: pip install pytesseract pillow")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("WARNING: pdf2image not installed. PDF OCR will be disabled.")
    print("Install with: pip install pdf2image")
    print("Also install poppler: https://github.com/oschwartz10612/poppler-windows/releases (Windows)")

# Date pattern for Turkish documents
DATE_RE = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")


def txt_to_text(path: str) -> str:
    """Read text from a .txt file."""
    text = Path(path).read_text(encoding='utf-8')
    return text


def image_to_text(image_path: str, lang: str = 'tur+eng') -> str:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image_path: Path to image file
        lang: Tesseract language code (default: 'tur+eng' for Turkish + English)
    
    Returns:
        Extracted text as string
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract is required for image OCR. Install with: pip install pytesseract pillow")
    
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text
    except Exception as e:
        raise RuntimeError(f"OCR failed for {image_path}: {str(e)}")


def pdf_to_text(pdf_path: str, lang: str = 'tur+eng', dpi: int = 300) -> str:
    """
    Extract text from a PDF file by converting pages to images and running OCR.
    
    Args:
        pdf_path: Path to PDF file
        lang: Tesseract language code
        dpi: Resolution for PDF to image conversion (higher = better quality, slower)
    
    Returns:
        Extracted text from all pages concatenated
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image is required for PDF OCR. Install with: pip install pdf2image")
    
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract is required for PDF OCR. Install with: pip install pytesseract pillow")
    
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Extract text from each page
        texts = []
        for i, image in enumerate(images):
            print(f"Processing PDF page {i+1}/{len(images)}...", file=sys.stderr)
            page_text = pytesseract.image_to_string(image, lang=lang)
            texts.append(page_text)
        
        # Join pages with double newline
        return '\n\n'.join(texts)
    except Exception as e:
        raise RuntimeError(f"PDF OCR failed for {pdf_path}: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    - Remove excessive line breaks (before normalizing whitespace)
    - Normalize whitespace
    - Fix common encoding issues
    """
    # Remove excessive line breaks first (before normalizing whitespace)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize whitespace (but preserve single newlines)
    # Replace multiple spaces/tabs with single space, but keep single newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
    text = re.sub(r' \n', '\n', text)  # Remove space before newline
    text = re.sub(r'\n ', '\n', text)  # Remove space after newline
    
    # Fix common Turkish encoding issues
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u201c', '"')   # Left double quotation mark
    text = text.replace('\u201d', '"')   # Right double quotation mark
    text = text.replace('\u2013', '-')   # En dash
    text = text.replace('\u2014', '--')  # Em dash
    
    return text.strip()


def process_path(path: str, ocr_lang: str = 'tur+eng', pdf_dpi: int = 300) -> str:
    """
    Process a file (text, PDF, or image) and return cleaned text.
    
    Args:
        path: Path to the file
        ocr_lang: Tesseract language code for OCR (default: 'tur+eng')
        pdf_dpi: DPI for PDF to image conversion (default: 300)
    
    Returns:
        Cleaned text content
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If required OCR libraries are missing
        RuntimeError: If OCR processing fails
    """
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = p.suffix.lower()
    
    # Handle different file types
    if suffix == '.txt':
        raw = txt_to_text(p)
    elif suffix == '.pdf':
        if not PDF2IMAGE_AVAILABLE or not TESSERACT_AVAILABLE:
            raise ImportError(
                "PDF OCR requires pdf2image and pytesseract. "
                "Install with: pip install pdf2image pytesseract pillow"
            )
        raw = pdf_to_text(str(p), lang=ocr_lang, dpi=pdf_dpi)
    elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "Image OCR requires pytesseract. Install with: pip install pytesseract pillow"
            )
        raw = image_to_text(str(p), lang=ocr_lang)
    else:
        # Try to read as text file for unknown extensions
        print(f"Warning: Unknown file type {suffix}, attempting to read as text...", file=sys.stderr)
        try:
            raw = txt_to_text(p)
        except Exception as e:
            raise ValueError(f"Cannot process file type {suffix}: {str(e)}")
    
    return clean_text(raw)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ocr_preproc.py <file_path> [--lang LANG] [--dpi DPI]')
        print('  file_path: Path to text, PDF, or image file')
        print('  --lang: Tesseract language code (default: tur+eng)')
        print('  --dpi: PDF conversion DPI (default: 300)')
        sys.exit(1)
    
    file_path = sys.argv[1]
    lang = 'tur+eng'
    dpi = 300
    
    # Parse optional arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--lang' and i + 1 < len(sys.argv):
            lang = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--dpi' and i + 1 < len(sys.argv):
            dpi = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    try:
        output = process_path(file_path, ocr_lang=lang, pdf_dpi=dpi)
        print(output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)