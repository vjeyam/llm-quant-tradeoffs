import re
from time import perf_counter

from ollama import Client

from src.schemas import ModelConfig, Prediction, QuestionExample


client = Client(host="http://localhost:11434")


SYSTEM_PROMPT = """
You are answering medical benchmark questions.

Select the single best answer using only the choices provided.

Return only the answer label.
Do not provide explanations, reasoning, punctuation, or additional text.
""".strip()


def build_prompt(example: QuestionExample) -> str:
    parts = []
    
    if example.context:
        parts.append(f"Context:\n{example.context}")
        
    parts.append(f"Questions:\n{example.question}")
    
    choices = "\n".join(
        f"{label}. {text}"
        for label, text in example.choices.items()
    )
    
    parts.append(f"Choices:\n{choices}")
    return "\n\n".join(parts)


def normalize_prediction(
    output: str,
    example: QuestionExample,
) -> tuple[str | None, bool]:
    text = output.strip()

    if example.dataset == "medqa":
        strict = re.fullmatch(r"[AaBbCcDd][\.\)]?", text)

        if strict:
            return text[0].upper(), True

        match = re.search(
            r"\b(?:answer\s*(?:is|:)?\s*)?([ABCD])\b",
            text,
            flags=re.IGNORECASE,
        )

        return (
            match.group(1).upper() if match else None,
            False,
        )

    if example.dataset == "pubmedqa":
        normalized = text.lower().rstrip(".")

        if normalized in {"yes", "no", "maybe"}:
            return normalized, True

        match = re.search(
            r"\b(yes|no|maybe)\b",
            text,
            flags=re.IGNORECASE,
        )

        return (
            match.group(1).lower() if match else None,
            False,
        )

    return None, False


def run_inference(
    example: QuestionExample,
    config: ModelConfig,
) -> Prediction:
    start = perf_counter()

    response = client.chat(
        model=config.name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_prompt(example),
            },
        ],
        options={
            "temperature": config.temperature,
            "seed": config.seed,
            "num_predict": config.num_predict,
        },
    )

    latency = perf_counter() - start
    output = response.message.content
    prediction, strict_format = normalize_prediction(output, example)

    return Prediction(
        example_id=example.id,
        dataset=example.dataset,
        model=config.name,
        prediction=prediction,
        answer=example.answer,
        raw_output=output,
        strict_format=strict_format,
        latency_seconds=latency,
        prompt_tokens=response.prompt_eval_count,
        completion_tokens=response.eval_count,
        eval_duration_ns=response.eval_duration,
    )
    
    
def run_dataset(
    examples: list[QuestionExample],
    config: ModelConfig,
) -> list[Prediction]:
    if not examples:
        return []

    # Warm up the model so startup/loading time doesn't pollute latency metrics.
    run_inference(examples[0], config)

    return [
        run_inference(example, config)
        for example in examples
    ]