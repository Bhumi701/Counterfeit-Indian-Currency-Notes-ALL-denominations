"""
main.py -- Full pipeline entry point.

BASIC USAGE (pattern + OCR + texture agents only):
    python main.py path/to/note_image.jpg 500

WITH THREAD AGENT (if you have tilt and/or backlit photos of the same note):
    python main.py path/to/note_image.jpg 500 --tilt path/to/tilt_photo.jpg
    python main.py path/to/note_image.jpg 500 --backlit path/to/backlit_photo.jpg

WITH FEATURE DETECTION AGENT (auto-enabled if you've trained the YOLO26
detector via train_detector.py -- runs/note_feature_detector/weights/best.pt exists):
    No extra flags needed, it's used automatically when available.
"""

import sys
import os
import argparse
import cv2

from pattern_agent import run_pattern_agent
from ocr_agent import run_ocr_agent
from texture_agent import TextureAgent
from thread_agent import run_thread_agent
from fusion_agent import fuse

DETECTOR_WEIGHTS_PATH = "runs/note_feature_detector/weights/best.pt"


def analyze_note(image_path, denomination, tilt_path=None, backlit_path=None):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")

    if denomination not in ("10", "20", "50", "100", "200", "500"):
        raise ValueError("Denomination must be one of: 10, 20, 50, 100, 200, 500")

    print(f"\nAnalyzing Rs {denomination} note: {image_path}\n" + "-" * 50)

    pattern_result = run_pattern_agent(image_bgr, denomination)
    print("Pattern agent:", pattern_result)

    ocr_result = run_ocr_agent(image_bgr)
    print("OCR agent:", ocr_result)

    texture_agent = TextureAgent()
    texture_result = texture_agent.run(image_bgr)
    print("Texture agent:", texture_result)

    thread_result = None
    if tilt_path or backlit_path:
        tilt_img = cv2.imread(tilt_path) if tilt_path else None
        backlit_img = cv2.imread(backlit_path) if backlit_path else None
        thread_result = run_thread_agent(image_bgr, tilt_img=tilt_img, backlit_img=backlit_img)
        print("Thread agent:", thread_result)

    feature_detection_result = None
    if os.path.exists(DETECTOR_WEIGHTS_PATH):
        from feature_detection_agent import FeatureDetector, run_feature_detection_agent
        detector = FeatureDetector(DETECTOR_WEIGHTS_PATH)
        if detector.loaded:
            feature_detection_result = run_feature_detection_agent(image_bgr, denomination, detector)
            print("Feature detection agent:", feature_detection_result)

    final = fuse(pattern_result, ocr_result, texture_result, thread_result, feature_detection_result)

    print("\n" + "=" * 50)
    print(f"VERDICT: {final['verdict']}")
    print(f"Confidence score: {final['final_score']}")
    print(f"Weights used: {final['weights_used']}")
    print("Reasons:")
    for r in final["reasons"]:
        print(f"  - {r}")
    print("=" * 50 + "\n")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfeit note detection pipeline")
    parser.add_argument("image_path", help="Path to the straight-on note photo")
    parser.add_argument("denomination", choices=["10", "20", "50", "100", "200", "500"])
    parser.add_argument("--tilt", default=None, help="Path to a tilted photo of the same note (optional)")
    parser.add_argument("--backlit", default=None, help="Path to a backlit photo of the same note (optional)")
    args = parser.parse_args()

    analyze_note(args.image_path, args.denomination, args.tilt, args.backlit)