import json
from pathlib import Path


RESULTS_DIR = Path("results")

MODEL_METADATA = {
    "qwen2.5-7b-f16": {
        "quantization": "F16",
        "size_gb": 14.19,
    },
    "qwen2.5-7b-q8": {
        "quantization": "Q8_0",
        "size_gb": 7.50,
    },
    "qwen2.5-7b-q4": {
        "quantization": "Q4_K_M",
        "size_gb": 4.36,
    },
    "qwen2.5-7b-q2": {
        "quantization": "Q2_K",
        "size_gb": 2.81,
    },
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)
    

def build_quantization_summary() -> dict:
    summary = {}
    
    for model, metadata in MODEL_METADATA.items():
        model_dir = RESULTS_DIR / model
        
        medqa = load_json(
            model_dir / "medqa_summary.json"
        )
        
        pubmedqa = load_json(
            model_dir / "pubmedqa_summary.json"
        )
        
        summary[model] = {
            **metadata,
            
            "medqa": {
                "accuracy": medqa["accuracy"],
                "strict_format_rate": medqa["strict_format_rate"],
                "unparseable_rate": medqa["unparseable_rate"],
                "median_latency_seconds": medqa["median_latency_seconds"],
                "p95_latency_seconds": medqa["p95_latency_seconds"],
                "mean_tokens_per_second": medqa["mean_tokens_per_second"],
            },

            "pubmedqa": {
                "accuracy": pubmedqa["accuracy"],
                "strict_format_rate": pubmedqa["strict_format_rate"],
                "unparseable_rate": pubmedqa["unparseable_rate"],
                "median_latency_seconds": pubmedqa["median_latency_seconds"],
                "p95_latency_seconds": pubmedqa["p95_latency_seconds"],
                "mean_tokens_per_second": pubmedqa["mean_tokens_per_second"],
            },
        }
        
    output_path = RESULTS_DIR / "quantization_summary.json"
    
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
            
    return summary
    
    
if __name__ == "__main__":
    build_quantization_summary()