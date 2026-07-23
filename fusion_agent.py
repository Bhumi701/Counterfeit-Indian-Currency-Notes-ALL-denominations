"""
fusion_agent.py -- Combines agent scores into one final verdict, with an
explanation of which checks passed/failed.

The thread agent is optional (it needs tilt/backlit photos you may not
always have for every note). When it's not provided, its weight is
redistributed proportionally across the agents that ARE available, so
the fusion always uses weights that sum to 1.0.
"""

from config import FUSION_WEIGHTS, GENUINE_THRESHOLD, SUSPECT_THRESHOLD


def _effective_weights(available_agents):
    """Redistributes weight of any missing agent proportionally across
    the agents that are present."""
    base = {k: FUSION_WEIGHTS[k] for k in available_agents}
    total = sum(base.values())
    if total == 0:
        # fallback: equal weights if something went very wrong
        return {k: 1.0 / len(available_agents) for k in available_agents}
    return {k: v / total for k, v in base.items()}


def fuse(pattern_result, ocr_result, texture_result, thread_result=None, feature_detection_result=None):
    results = {
        "pattern": pattern_result,
        "ocr": ocr_result,
        "texture": texture_result,
    }
    if thread_result is not None:
        results["thread"] = thread_result
    if feature_detection_result is not None:
        results["feature_detection"] = feature_detection_result

    weights = _effective_weights(list(results.keys()))

    weighted_sum = sum(results[agent]["score"] * weights[agent] for agent in results)

    if weighted_sum >= GENUINE_THRESHOLD:
        verdict = "GENUINE"
    elif weighted_sum >= SUSPECT_THRESHOLD:
        verdict = "SUSPECT -- please verify manually"
    else:
        verdict = "LIKELY FAKE"

    # Build a human-readable explanation of what drove the verdict
    reasons = []
    if pattern_result["score"] < 0.6:
        checks = pattern_result["checks"]
        reasons.append(
            f"Geometric pattern check scored low ({pattern_result['score']}): "
            f"{checks['size_ratio']['detail']}; {checks['bleed_lines']['detail']}; "
            f"{checks['identification_mark']['detail']}"
        )
    if ocr_result["score"] < 0.6:
        reasons.append(f"Serial number check scored low ({ocr_result['score']}): "
                        f"detected '{ocr_result['checks']['serial_number_detected']}', "
                        f"format valid: {ocr_result['checks']['format_valid']}")
    if texture_result["score"] < 0.6:
        reasons.append(f"Print texture check scored low ({texture_result['score']})")
    if thread_result is not None and thread_result["score"] < 0.6:
        checks = thread_result["checks"]
        reasons.append(
            f"Thread/register check scored low ({thread_result['score']}): "
            f"{checks['color_shift_thread']['detail']}; {checks['see_through_register']['detail']}"
        )
    if feature_detection_result is not None and feature_detection_result["score"] < 0.6:
        checks = feature_detection_result["checks"]
        reasons.append(
            f"Feature detection scored low ({feature_detection_result['score']}): "
            f"{checks['identification_mark']['detail']}; {checks['security_thread']['detail']}; "
            f"{checks['serial_panel']['detail']}"
        )

    if not reasons:
        reasons.append("All checks passed within expected range.")

    agent_scores = {agent: results[agent]["score"] for agent in results}

    return {
        "final_score": round(weighted_sum, 3),
        "verdict": verdict,
        "reasons": reasons,
        "agent_scores": agent_scores,
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
    }