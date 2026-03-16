def calculate_confidence(scores):

    if not scores:
        return 0

    avg_distance = sum(scores) / len(scores)

    confidence = 1 / (1 + avg_distance)

    return round(confidence, 2)