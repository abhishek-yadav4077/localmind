"""Load, validate and upload the evaluation dataset (``eval/dataset.json``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from localmind.config import get_logger

logger = get_logger(__name__)

Category = Literal["factual", "reasoning", "coding", "summarization", "refusal"]


class EvalItem(BaseModel):
    id: str
    category: Category
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    must_refuse: bool = False

    @model_validator(mode="after")
    def _refusal_only_in_refusal_category(self) -> EvalItem:
        if self.must_refuse and self.category != "refusal":
            raise ValueError(f"{self.id}: must_refuse items belong in the 'refusal' category")
        return self


class EvalDataset(BaseModel):
    name: str
    description: str = ""
    items: list[EvalItem]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalDataset:
        ids = [i.id for i in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate item ids in dataset")
        return self

    @property
    def content_hash(self) -> str:
        """Hash of the items: changes whenever a question or reference changes."""
        raw = json.dumps([i.model_dump() for i in self.items], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    @property
    def versioned_name(self) -> str:
        """``localmind-eval-<hash>``: editing the JSON creates a *new* LangSmith
        dataset, so experiments are only ever compared on identical data."""
        return f"{self.name}-{self.content_hash}"

    def to_examples(self) -> list[dict[str, Any]]:
        """LangSmith example format: inputs / outputs (reference) / metadata."""
        return [
            {
                "inputs": {"question": i.question},
                "outputs": {"reference": i.reference, "must_refuse": i.must_refuse},
                "metadata": {"id": i.id, "category": i.category},
            }
            for i in self.items
        ]


def load_dataset(path: str | Path) -> EvalDataset:
    return EvalDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def ensure_langsmith_dataset(client: Any, dataset: EvalDataset) -> str:
    """Create the versioned dataset in LangSmith if it doesn't exist. Returns its name."""
    name = dataset.versioned_name
    if client.has_dataset(dataset_name=name):
        logger.info("langsmith dataset exists", extra={"dataset": name})
        return name
    created = client.create_dataset(
        name,
        description=dataset.description,
        metadata={"content_hash": dataset.content_hash, "source": "eval/dataset.json"},
    )
    client.create_examples(dataset_id=created.id, examples=dataset.to_examples())
    logger.info(
        "langsmith dataset created", extra={"dataset": name, "examples": len(dataset.items)}
    )
    return name
