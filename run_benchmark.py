from src.benchmark import run_benchmark
from src.schemas import ModelConfig


MODELS = [
    "qwen2.5-7b-f16",
    "qwen2.5-7b-q8",
    "qwen2.5-7b-q4",
    "qwen2.5-7b-q2",
]


def main():
    results = {}

    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {model}")
        print(f"{'=' * 60}")

        config = ModelConfig(
            name=model,
            temperature=0.0,
            seed=42,
        )

        results[model] = run_benchmark(
            config=config,
            medqa_limit=None,
            pubmedqa_limit=None,
        )

    return results


if __name__ == "__main__":
    main()