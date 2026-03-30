def calculate_confidence(scores):
    if not scores:
        return 0.0, "none"

    avg_distance = sum(scores) / len(scores)
    confidence = round(1 / (1 + avg_distance), 2)

    if confidence >= 0.7:
        level = "high"
    elif confidence >= 0.45:
        level = "medium"
    else:
        level = "low"

    return confidence, level