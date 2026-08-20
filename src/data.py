from datasets import load_dataset

from src.schemas import QuestionExample


MEDQA_DATASET = "GBaker/MedQA-USMLE-4-options-hf"
PUBMEDQA_DATASET = "qiaojin/PubMedQA"


def load_medqa(
    split: str = "test",
    limit: int | None = None,
) -> list[QuestionExample]:
    dataset = load_dataset(MEDQA_DATASET, split=split)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [
        _normalize_medqa(row, i)
        for i, row in enumerate(dataset)
    ]


def load_pubmedqa(
    split: str = "train",
    limit: int | None = None,
) -> list[QuestionExample]:
    dataset = load_dataset(
        PUBMEDQA_DATASET,
        "pqa_labeled",
        split=split,
    )

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [
        _normalize_pubmedqa(row, i)
        for i, row in enumerate(dataset)
    ]


def _normalize_medqa(row: dict, index: int) -> QuestionExample:
    choices = {
        "A": row["ending0"],
        "B": row["ending1"],
        "C": row["ending2"],
        "D": row["ending3"],
    }

    answer = ("A", "B", "C", "D")[int(row["label"])]

    return QuestionExample(
        id=row.get("id", f"medqa_{index}"),
        dataset="medqa",
        question=row["sent1"],
        choices=choices,
        answer=answer,
    )


def _normalize_pubmedqa(row: dict, index: int) -> QuestionExample:
    context = row["context"]["contexts"]

    if isinstance(context, list):
        context = "\n\n".join(context)

    return QuestionExample(
        id=f"pubmedqa_{row.get('pubid', index)}",
        dataset="pubmedqa",
        question=row["question"],
        context=context,
        choices={
            "yes": "yes",
            "no": "no",
            "maybe": "maybe",
        },
        answer=row["final_decision"].lower(),
    )