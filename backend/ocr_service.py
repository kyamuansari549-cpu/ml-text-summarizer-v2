"""OCR service: extract text from an uploaded image with Tesseract.

Needs the Tesseract binary installed on the machine:
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki  (then either add
    it to PATH or set TESSERACT_CMD to tesseract.exe)
  - Linux:   sudo apt install tesseract-ocr tesseract-ocr-hin

Env:
    TESSERACT_CMD  full path to the tesseract binary (optional)
"""

import io
import os

from PIL import Image

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
except ImportError:  # pragma: no cover - handled at call time
    pytesseract = None
    TesseractNotFoundError = RuntimeError

_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if _TESSERACT_CMD and pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def extract_text(image_bytes, lang="eng"):
    """Return the text Tesseract reads from *image_bytes*.

    Raises RuntimeError if Tesseract isn't installed.
    """
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed (pip install pytesseract) "
            "and the Tesseract binary is required."
        )
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Tesseract expects dark text on a light background. Dark screenshots
    # (e.g. terminal/VS Code) get inverted so they read correctly.
    from PIL import ImageOps, ImageStat

    if ImageStat.Stat(img.convert("L")).mean[0] < 128:
        img = ImageOps.invert(img)
    try:
        text = pytesseract.image_to_string(img, lang=lang)
    except TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract binary not found. Install it and/or set TESSERACT_CMD."
        ) from exc
    return (text or "").strip()
