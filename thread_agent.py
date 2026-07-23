"""
thread_agent.py -- Uses the color-shift security thread and see-through
register checks. Unlike pattern/OCR/texture agents, this one needs TWO
photos of the same note:

  1. A straight-on photo (normal, flat)
  2. Either:
       a) A tilted photo of the same note (to check the thread's
          green-to-blue color shift), and/or
       b) A backlit photo -- note held up against a light source (to
          check the see-through register / watermark alignment)

If you only have one of the two paired photos, that specific check is
skipped gracefully (scored neutral) rather than failing.

NOTE ON CROP REGIONS: The coordinates below are approximate, based on
RBI's published note layout (thread runs roughly through the middle of
the note, watermark/register sits in the left third). You will very
likely need to tune these fractions once you test on your own photos --
run tune_crop_regions() (bottom of this file) to visually check where
your crop lands.
"""

import cv2
import numpy as np


def _thread_strip(image_bgr, x_frac=(0.42, 0.52)):
    """Crops the vertical strip where the windowed security thread runs."""
    h, w = image_bgr.shape[:2]
    x1, x2 = int(w * x_frac[0]), int(w * x_frac[1])
    return image_bgr[:, x1:x2]


def _average_hue(strip_bgr):
    """Returns the average hue of a strip in HSV space (0-179 in OpenCV)."""
    hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0]))


def check_color_shift_thread(straight_img, tilt_img):
    """
    Genuine notes: the security thread visibly shifts from green to blue
    when the note is tilted. We approximate this by comparing average hue
    of the thread region between a straight-on and a tilted photo.

    Green hue (OpenCV) ~ 35-85, Blue hue ~ 90-130. A genuine note should
    show a noticeable hue increase between straight and tilted photos.
    """
    if straight_img is None or tilt_img is None:
        return {"score": 0.5, "detail": "Tilt photo not provided -- skipping thread color-shift check"}

    straight_hue = _average_hue(_thread_strip(straight_img))
    tilt_hue = _average_hue(_thread_strip(tilt_img))
    hue_shift = tilt_hue - straight_hue

    # A genuine shift should move noticeably toward blue (positive hue shift,
    # roughly 15-60 depending on lighting). Tune this range against your data.
    if hue_shift > 10:
        score = min(1.0, hue_shift / 40)
    else:
        score = max(0.0, 0.3 + hue_shift / 50)  # small/negative shift -> low score

    return {
        "score": round(score, 3),
        "detail": f"Hue shift straight->tilt: {hue_shift:.1f} (straight={straight_hue:.1f}, tilt={tilt_hue:.1f})",
    }


def check_see_through_register(front_img, backlit_img, alignment_threshold=15):
    """
    Genuine notes: the see-through register (partial Gandhi portrait
    printed on both front, aligning through the paper) lines up almost
    perfectly when held against light. We approximate this by checking
    how well a specific patch on the front image aligns (via template
    matching) with the equivalent patch when backlit.
    """
    if front_img is None or backlit_img is None:
        return {"score": 0.5, "detail": "Backlit photo not provided -- skipping see-through register check"}

    h, w = front_img.shape[:2]
    # Approximate register location: left third of note, vertically centered
    patch = front_img[int(h * 0.35):int(h * 0.65), int(w * 0.03):int(w * 0.18)]
    if patch.size == 0:
        return {"score": 0.3, "detail": "Could not extract register patch from front image"}

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    backlit_gray = cv2.cvtColor(backlit_img, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(backlit_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # max_val is a similarity score from -1 to 1; higher = better alignment
    score = max(0.0, min(1.0, (max_val + 1) / 2))

    return {
        "score": round(score, 3),
        "detail": f"Template match confidence: {max_val:.3f}",
    }


def run_thread_agent(straight_img, tilt_img=None, backlit_img=None):
    """Combines both checks. Either tilt_img or backlit_img can be None
    if you don't have that photo -- the check is skipped gracefully."""
    thread_result = check_color_shift_thread(straight_img, tilt_img)
    register_result = check_see_through_register(straight_img, backlit_img)

    combined_score = (thread_result["score"] + register_result["score"]) / 2

    return {
        "agent": "thread",
        "score": round(combined_score, 3),
        "checks": {
            "color_shift_thread": thread_result,
            "see_through_register": register_result,
        },
    }


def tune_crop_regions(image_path, out_path="crop_debug.jpg"):
    """
    Helper to visually check where the thread strip and register patch
    land on YOUR photos. Run this once on a real note image and open
    crop_debug.jpg to see the boxes -- adjust the x_frac/patch fractions
    above if they don't line up with the actual features.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return
    h, w = img.shape[:2]
    debug = img.copy()

    # thread strip box (yellow)
    x1, x2 = int(w * 0.42), int(w * 0.52)
    cv2.rectangle(debug, (x1, 0), (x2, h), (0, 255, 255), 2)

    # register patch box (magenta)
    rx1, ry1 = int(w * 0.03), int(h * 0.35)
    rx2, ry2 = int(w * 0.18), int(h * 0.65)
    cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)

    cv2.imwrite(out_path, debug)
    print(f"Saved debug crop visualization to {out_path} -- "
          f"yellow box = thread region, magenta box = register region")


if __name__ == "__main__":
    import numpy as np
    straight = np.full((400, 900, 3), (150, 130, 100), dtype=np.uint8)
    tilt = np.full((400, 900, 3), (170, 100, 90), dtype=np.uint8)
    result = run_thread_agent(straight, tilt_img=tilt, backlit_img=None)
    print("Self-test result (synthetic images, not real notes):")
    print(result)