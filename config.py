"""
config.py -- Known physical specs of each denomination, taken from RBI's
official "Know Your Banknotes" documentation.

These numbers are the ground truth we compare each scanned note against.
"""

# Physical note size in mm (width x height), from RBI docs
DENOMINATIONS = {
    "10": {
        "size_mm": (63, 123),
        "id_mark_shape": None,       # RBI doc does not list one for Rs 10
        "expected_bleed_lines": None,
        "base_color_name": "chocolate brown",
    },
    "20": {
        "size_mm": (63, 129),
        "id_mark_shape": None,       # RBI doc does not list one for Rs 20
        "expected_bleed_lines": None,
        "base_color_name": "greenish yellow",
    },
    "50": {
        "size_mm": (66, 135),
        "id_mark_shape": None,       # RBI doc does not list one for Rs 50
        "expected_bleed_lines": None,
        "base_color_name": "fluorescent blue",
    },
    "100": {
        "size_mm": (66, 142),
        "id_mark_shape": "triangle",
        "expected_bleed_lines": 4,
        "base_color_name": "lavender",
    },
    "200": {
        "size_mm": (66, 146),
        "id_mark_shape": "H_with_circles",
        "expected_bleed_lines": 4,
        "base_color_name": "bright yellow",
    },
    "500": {
        "size_mm": (66, 150),
        "id_mark_shape": "circle",
        "expected_bleed_lines": 5,
        "base_color_name": "stone grey",
    },
}

# Fusion weights -- how much each agent's opinion counts toward the final verdict.
# Pattern (geometric) checks are most reliable since they don't depend on
# capture conditions, so they get the highest weight. Thread agent is also
# very reliable (physical property, hard to fake) but only usable when you
# have tilt/backlit photos -- when unavailable, its weight is automatically
# redistributed across the other agents (see fusion_agent.py).
FUSION_WEIGHTS = {
    "pattern": 0.30,
    "ocr": 0.15,
    "texture": 0.15,
    "thread": 0.15,
    "feature_detection": 0.25,
}

# Verdict thresholds on the final fused score (0 = certainly fake, 1 = certainly genuine)
GENUINE_THRESHOLD = 0.65
SUSPECT_THRESHOLD = 0.40   # below this -> "fake", between the two -> "suspect, verify manually"