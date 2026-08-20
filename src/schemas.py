from typing import Literal

from pydantic import BaseModel


DatasetName = Literal["medqa", "pubmedqa"]


class QuestionExample(BaseModel):
    id: str
    dataset: DatasetName
    question: str
    choices: dict[str, str]
    answer: str
    context: str | None = None
    

class ModelConfig(BaseModel):
    name: str
    temperature: float = 0.0
    seed: int = 42
    num_predict: int = 32
    
    
class Prediction(BaseModel):
    example_id: str
    dataset: DatasetName
    model: str
    prediction: str | None
    answer: str

    raw_output: str
    strict_format: bool

    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    eval_duration_ns: int | None = None

    @property
    def correct(self) -> bool:
        return self.prediction == self.answer

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or not self.eval_duration_ns:
            return None

        seconds = self.eval_duration_ns / 1e9
        return self.completion_tokens / seconds if seconds else None