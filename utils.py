def interpret_emotion(results):
    emotions = list(results.items())

    top_emotion, top_score = emotions[0]
    second_emotion, second_score = emotions[1]

    # Convert to percentage
    top_score_percent = round(top_score * 100, 2)

    # Mixed emotion logic
    if abs(top_score - second_score) < 0.15:
        label = f"Mixed Emotion: {top_emotion} + {second_emotion}"
    else:
        label = top_emotion

    # Neutral override
    if top_score < 0.40:
        label = "neutral"

    return label, top_score_percent