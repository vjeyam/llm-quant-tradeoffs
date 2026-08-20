from statistics import mean, median

from src.schemas import Prediction


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def summarize_predictions(
    predictions: list[Prediction],
) -> dict:
    if not predictions:
        raise ValueError("No predictions provided")

    latencies = [
        prediction.latency_seconds
        for prediction in predictions
    ]

    throughputs = [
        prediction.tokens_per_second
        for prediction in predictions
        if prediction.tokens_per_second is not None
    ]

    parsed_predictions = [
        prediction
        for prediction in predictions
        if prediction.prediction is not None
    ]

    strict_predictions = [
        prediction
        for prediction in predictions
        if prediction.strict_format
    ]

    correct = sum(
        prediction.correct
        for prediction in predictions
    )

    total = len(predictions)

    return {
        "dataset": predictions[0].dataset,
        "model": predictions[0].model,
        "num_examples": total,

        "correct": correct,
        "accuracy": correct / total,

        "unparseable_count": total - len(parsed_predictions),
        "unparseable_rate": (
            total - len(parsed_predictions)
        ) / total,

        "strict_format_count": len(strict_predictions),
        "strict_format_rate": (
            len(strict_predictions) / total
        ),

        "mean_latency_seconds": mean(latencies),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": percentile(
            latencies,
            0.95,
        ),

        "mean_tokens_per_second": (
            mean(throughputs)
            if throughputs
            else None
        ),
    }