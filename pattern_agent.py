"""
pattern_agent.py -- Geometric verification agent.

Checks that don't need any ML training, just classical image processing:
  1. Note size/aspect ratio vs the known RBI spec for that denomination
  2. Bleed line count on the left/right edges vs the known spec

These are the most reliable checks because they work from a normal photo
(no special lighting/tilt needed) and don't depend on your training data size.
"""

import cv2
import numpy as np
from config import DENOMINATIONS


def _find_note_contour(image_bgr):
    """Finds the largest rectangular contour in the image (the note itself)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest


def check_size_ratio(image_bgr, denomination):
    """
    Compares the detected note's width:height ratio against the RBI spec.
    Returns a score in [0, 1] -- 1.0 means ratio matches almost exactly.
    """
    spec = DENOMINATIONS[denomination]
    expected_w, expected_h = spec["size_mm"]
    expected_ratio = expected_w / expected_h

    contour = _find_note_contour(image_bgr)
    if contour is None:
        return {"score": 0.0, "detail": "Could not detect note boundary in image"}

    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return {"score": 0.0, "detail": "Invalid note boundary detected"}

    detected_ratio = w / h
    # Ratio is orientation-independent -- notes can be photographed either way
    diff = min(abs(detected_ratio - expected_ratio),
               abs((1 / detected_ratio) - expected_ratio))
    score = max(0.0, 1.0 - (diff / expected_ratio) * 4)  # 25% off ratio -> score 0

    return {
        "score": round(score, 3),
        "detail": f"Expected ratio {expected_ratio:.3f}, detected {detected_ratio:.3f}",
    }


def count_bleed_lines(image_bgr, denomination, edge_margin_frac=0.08):
    """
    Counts the short thick vertical bars (bleed lines / identification marks)
    near the left and right edges of the note, and compares against the
    expected count for that denomination.

    NOTE: This is a heuristic geometric check, not pixel-perfect template
    matching. It works best on a straight-on, well-lit, cropped note photo.
    """
    spec = DENOMINATIONS[denomination]
    expected_count = spec["expected_bleed_lines"]

    if expected_count is None:
        return {"score": 0.5, "detail": f"No documented bleed-line spec for Rs {denomination} note; skipping"}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin = int(w * edge_margin_frac)

    left_strip = gray[:, :margin]
    right_strip = gray[:, w - margin:]

    def count_bars_in_strip(strip):
        _, thresh = cv2.threshold(strip, 90, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = 0
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # bleed lines are short, thick horizontal-ish bars
            if cw > strip.shape[1] * 0.3 and 3 < ch < strip.shape[0] * 0.15:
                bars += 1
        return bars

    left_count = count_bars_in_strip(left_strip)
    right_count = count_bars_in_strip(right_strip)
    detected = max(left_count, right_count)   # take the clearer side

    diff = abs(detected - expected_count)
    score = max(0.0, 1.0 - diff / max(expected_count, 1))

    return {
        "score": round(score, 3),
        "detail": f"Expected ~{expected_count} bleed lines, detected {detected} "
                  f"(left={left_count}, right={right_count})",
    }


def classify_shape(contour):
    """
    Classifies a contour's shape as triangle / circle / complex (H-like).
    Used to verify the identification mark shape (₹100=triangle, ₹500=circle,
    ₹200=H-with-circles which falls under 'complex').
    """
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "unknown"
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri)

    if len(approx) == 3:
        return "triangle"
    elif circularity > 0.75:
        return "circle"
    else:
        return "complex"


def check_identification_mark(image_bgr, denomination, search_margin_frac=0.15):
    """
    Looks for the identification mark near the right edge of the note
    (where RBI places it) and checks whether its shape matches what's
    expected for this denomination.

    NOTE: The crop region here is approximate. If your real photos have
    the mark in a slightly different spot (depends on how tightly you
    cropped the note), adjust search_margin_frac or the crop logic below.
    """
    spec = DENOMINATIONS[denomination]
    expected_shape = spec["id_mark_shape"]

    # ₹50 has no documented ID mark in RBI's materials -- skip gracefully
    if expected_shape is None:
        return {"score": 0.5, "detail": f"No documented ID mark for Rs {denomination}; skipping"}

    # map our RBI shape labels to the 3 shape categories classify_shape() returns
    shape_category_map = {
        "triangle": "triangle",
        "circle": "circle",
        "H_with_circles": "complex",
    }
    expected_category = shape_category_map.get(expected_shape, "complex")

    h, w = image_bgr.shape[:2]
    margin = int(w * search_margin_frac)
    right_strip = image_bgr[:, w - margin:]

    gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"score": 0.3, "detail": "Could not locate identification mark region"}

    # Look at a few of the largest candidate contours in that strip
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    best_match = False
    detected_categories = []
    for c in contours:
        if cv2.contourArea(c) < 30:  # skip tiny noise
            continue
        category = classify_shape(c)
        detected_categories.append(category)
        if category == expected_category:
            best_match = True

    score = 1.0 if best_match else 0.3
    return {
        "score": score,
        "detail": f"Expected '{expected_category}' shape, detected candidates: {detected_categories or 'none'}",
    }



def run_pattern_agent(image_bgr, denomination):
    """Combines all geometric checks into a single agent score + explanation."""
    size_result = check_size_ratio(image_bgr, denomination)
    bleed_result = count_bleed_lines(image_bgr, denomination)
    id_mark_result = check_identification_mark(image_bgr, denomination)

    combined_score = (size_result["score"] + bleed_result["score"] + id_mark_result["score"]) / 3

    return {
        "agent": "pattern",
        "score": round(combined_score, 3),
        "checks": {
            "size_ratio": size_result,
            "bleed_lines": bleed_result,
            "identification_mark": id_mark_result,
        },
    }


if __name__ == "__main__":
    # Quick self-test with a synthetic note-like image (for sanity checking only)
    import sys
    test_img = np.full((400, 900, 3), 230, dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (850, 350), (200, 190, 170), -1)
    result = run_pattern_agent(test_img, "500")
    print("Self-test result (synthetic image, not a real note):")
    print(result)