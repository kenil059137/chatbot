def calculate_confidence(scores):
    if not scores:
        return 0.0, "none"

    # Use top 3 scores only — best matches matter most
    top_scores = sorted(scores)[:3]
    avg_distance = sum(top_scores) / len(top_scores)
    confidence = round(1 / (1 + avg_distance), 2)

    if confidence >= 0.70:
        level = "high"
    elif confidence >= 0.45:
        level = "medium"
    else:
        level = "low"

    return confidence, level