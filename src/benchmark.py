import json
from pathlib import Path

from src.data import load_medqa, load_pubmedqa
from src.inference import run_dataset
from src.metrics import summarize_predictions
from src.schemas import ModelConfig, Prediction


RESULTS_DIR = Path("results")


def save_predictions(
    predictions: list[Prediction],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            [p.model_dump() for p in predictions],
            file,
            indent=2,
        )


def save_summary(
    summary: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def benchmark_dataset(
    dataset_name: str,
    examples,
    config: ModelConfig,
) -> dict:
    print(
        f"\nRunning {config.name} on "
        f"{dataset_name} ({len(examples)} examples)..."
    )

    predictions = run_dataset(examples, config)
    summary = summarize_predictions(predictions)

    model_dir = RESULTS_DIR / config.name.replace(":", "_")

    save_predictions(
        predictions,
        model_dir / f"{dataset_name}_predictions.json",
    )

    save_summary(
        summary,
        model_dir / f"{dataset_name}_summary.json",
    )

    print(
        f"Accuracy: {summary['accuracy']:.3f} | "
        f"Unparseable: {summary['unparseable_rate']:.3f} | "
        f"Strict format: {summary['strict_format_rate']:.3f} | "
        f"Median latency: {summary['median_latency_seconds']:.3f}s"
    )

    return summary


def run_benchmark(
    config: ModelConfig,
    medqa_limit: int | None = None,
    pubmedqa_limit: int | None = None,
) -> dict:
    medqa = load_medqa(limit=medqa_limit)
    pubmedqa = load_pubmedqa(limit=pubmedqa_limit)

    medqa_summary = benchmark_dataset(
        "medqa",
        medqa,
        config,
    )

    pubmedqa_summary = benchmark_dataset(
        "pubmedqa",
        pubmedqa,
        config,
    )

    results = {
        "model": config.name,
        "medqa": medqa_summary,
        "pubmedqa": pubmedqa_summary,
    }

    model_dir = RESULTS_DIR / config.name.replace(":", "_")

    save_summary(
        results,
        model_dir / "benchmark_summary.json",
    )

    return results