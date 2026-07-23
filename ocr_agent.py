"""
ocr_agent.py -- Serial number verification agent.

Checks:
  1. Extracts the serial number text using Tesseract OCR
  2. Validates it matches RBI's format (letter-prefix + digits)
  3. Checks that digit height increases left-to-right (RBI's ascending font
     size security feature -- genuine notes print numerals in increasing size)

Requires the tesseract-ocr system binary to be installed:
  Ubuntu/Debian: sudo apt-get install tesseract-ocr
  Mac:           brew install tesseract
  Windows:       https://github.com/UB-Mannheim/tesseract/wiki
"""

import os
import re
import shutil
import cv2
import numpy as np
import pytesseract

# Ensure pytesseract can find the native Tesseract binary when available.
# This helps on Windows when the PATH or launcher environment differs from
# the Python process used by Streamlit or FastAPI.
if pytesseract.pytesseract.tesseract_cmd == "tesseract":
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                tesseract_path = candidate
                break
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# RBI serial numbers look like: 2-3 letters, then a numeral (0-9), then 6 digits
# e.g. "AAR000000" or "0AA000000". This regex is deliberately permissive since
# OCR often misreads a character or two.
SERIAL_PATTERN = re.compile(r"^[0-9A-Z]{2,4}[A-Z][0-9]{5,6}$")


def _crop_number_panel(image_bgr, corner="bottom_right"):
    """
    Crops the region where the number panel typically sits.
    These are rough normalized coordinates -- works reasonably across
    denominations since the layout is consistent in the Mahatma Gandhi
    New Series. Tune the fractions if your crops are off.
    """
    h, w = image_bgr.shape[:2]
    if corner == "bottom_right":
        return image_bgr[int(h * 0.78):int(h * 0.95), int(w * 0.55):int(w * 0.95)]
    else:  # top_left
        return image_bgr[int(h * 0.05):int(h * 0.22), int(w * 0.03):int(w * 0.35)]


def extract_serial_number(image_bgr):
    """Runs OCR on the number panel and returns the cleaned text + confidence."""
    crop = _crop_number_panel(image_bgr, "bottom_right")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = pytesseract.image_to_string(thresh, config=config)
    cleaned = re.sub(r"[^0-9A-Z]", "", text.upper())

    return cleaned


def validate_serial_format(serial_text):
    """Returns True if the serial number matches RBI's expected pattern."""
    if not serial_text:
        return False
    return bool(SERIAL_PATTERN.match(serial_text))


def check_ascending_font(image_bgr):
    """
    Checks whether digit height increases left-to-right in the number panel,
    which is a genuine RBI security feature. Returns a score in [0, 1].
    """
    crop = _crop_number_panel(image_bgr, "bottom_right")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3] > 5]
    if len(boxes) < 4:
        return {"score": 0.5, "detail": "Not enough digits detected to check font progression"}

    # sort left-to-right, look at heights
    boxes.sort(key=lambda b: b[0])
    heights = [b[3] for b in boxes]

    increasing_pairs = sum(1 for i in range(len(heights) - 1) if heights[i + 1] >= heights[i] - 1)
    ratio = increasing_pairs / (len(heights) - 1)

    return {"score": round(ratio, 3), "detail": f"{increasing_pairs}/{len(heights) - 1} digit-pairs show increasing height"}


def run_ocr_agent(image_bgr):
    """Combines all OCR-based checks into a single agent score + explanation."""
    serial = extract_serial_number(image_bgr)
    format_valid = validate_serial_format(serial)
    font_result = check_ascending_font(image_bgr)

    format_score = 1.0 if format_valid else 0.2  # not zero -- OCR misreads happen even on genuine notes
    combined_score = (format_score + font_result["score"]) / 2

    return {
        "agent": "ocr",
        "score": round(combined_score, 3),
        "checks": {
            "serial_number_detected": serial or "(none detected)",
            "format_valid": format_valid,
            "ascending_font": font_result,
        },
    }


if __name__ == "__main__":
    # Quick self-test with a blank synthetic image (for sanity checking only,
    # won't find a real serial number -- just confirms nothing crashes)
    test_img = np.full((400, 900, 3), 240, dtype=np.uint8)
    result = run_ocr_agent(test_img)
    print("Self-test result (synthetic image, not a real note):")
    print(result)