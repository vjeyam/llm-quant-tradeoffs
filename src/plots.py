import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/quantization_summary.json")
PLOTS_DIR = Path("plots")


def load_results() -> dict:
    with RESULTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)
    

def ordered_results(results: dict) -> list[dict]:
    order = [
        "qwen2.5-7b-f16",
        "qwen2.5-7b-q8",
        "qwen2.5-7b-q4",
        "qwen2.5-7b-q2",
    ]

    return [
        {
            "model": model,
            **results[model],
        }
        for model in order
    ]
    
    
def plot_accuracy_vs_size(data: list[dict]) -> None:
    sizes = [row["size_gb"] for row in data]
    medqa = [row["medqa"]["accuracy"] for row in data]
    pubmedqa = [row["pubmedqa"]["accuracy"] for row in data]
    labels = [row["quantization"] for row in data]

    plt.figure(figsize=(8, 5))

    plt.plot(sizes, medqa, marker="o", label="MedQA")
    plt.plot(sizes, pubmedqa, marker="o", label="PubMedQA")

    for x, y, label in zip(sizes, medqa, labels):
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")

    plt.xlabel("Model Size (GB)")
    plt.ylabel("Accuracy")
    plt.title("Medical QA Accuracy vs Model Size")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "accuracy_vs_size.png", dpi=200)
    plt.close()


def plot_accuracy_vs_latency(data: list[dict]) -> None:
    medqa_latency = [
        row["medqa"]["median_latency_seconds"]
        for row in data
    ]

    medqa_accuracy = [
        row["medqa"]["accuracy"]
        for row in data
    ]

    pubmedqa_latency = [
        row["pubmedqa"]["median_latency_seconds"]
        for row in data
    ]

    pubmedqa_accuracy = [
        row["pubmedqa"]["accuracy"]
        for row in data
    ]

    labels = [row["quantization"] for row in data]

    plt.figure(figsize=(8, 5))

    plt.scatter(
        medqa_latency,
        medqa_accuracy,
        label="MedQA",
        s=70,
    )

    plt.scatter(
        pubmedqa_latency,
        pubmedqa_accuracy,
        label="PubMedQA",
        s=70,
    )

    for x, y, label in zip(
        medqa_latency,
        medqa_accuracy,
        labels,
    ):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Median Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Inference Latency")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "accuracy_vs_latency.png",
        dpi=200,
    )

    plt.close()


def plot_compression_vs_accuracy(data: list[dict]) -> None:
    baseline_size = data[0]["size_gb"]

    compression = [
        1 - (row["size_gb"] / baseline_size)
        for row in data
    ]

    medqa = [
        row["medqa"]["accuracy"]
        for row in data
    ]

    pubmedqa = [
        row["pubmedqa"]["accuracy"]
        for row in data
    ]

    labels = [
        row["quantization"]
        for row in data
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        compression,
        medqa,
        marker="o",
        label="MedQA",
    )

    plt.plot(
        compression,
        pubmedqa,
        marker="o",
        label="PubMedQA",
    )

    for x, y, label in zip(
        compression,
        medqa,
        labels,
    ):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Size Reduction vs F16")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Retention vs Compression")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "compression_vs_accuracy.png",
        dpi=200,
    )

    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    results = load_results()
    data = ordered_results(results)

    plot_accuracy_vs_size(data)
    plot_accuracy_vs_latency(data)
    plot_compression_vs_accuracy(data)


if __name__ == "__main__":
    main()