"""Manifest parsing, response parsing, and controlled multimodal prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("circle", "square", "triangle", "star")
Task = Literal["color", "shape"]
Condition = Literal["neutral", "supportive", "misleading"]


def load_manifest(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    """Load the dataset, resolving image paths relative to the manifest's parent."""
    manifest = Path(path)
    records: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record["split"] != split:
                continue
            image_path = manifest.parent / record["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Line {line_number}: missing image {image_path}")
            record["absolute_image"] = str(image_path.resolve())
            records.append(record)
    return records


def build_prompt(record: dict[str, Any], task: Task, condition: Condition) -> str:
    """Keep task wording fixed across contexts and change only the context sentence."""
    if task not in ("color", "shape"):
        raise ValueError(f"Unsupported task: {task}")
    if condition not in ("neutral", "supportive", "misleading"):
        raise ValueError(f"Unsupported condition: {condition}")

    actual = record[task]
    claimed = record[f"wrong_{task}"] if condition == "misleading" else actual
    if condition == "neutral":
        context = "Context: This image shows one large central figure."
    elif task == "color":
        context = f"Context: The large central figure is {claimed}."
    else:
        context = f"Context: The large central figure is a {claimed}."

    if task == "color":
        question = "What COLOR is the large central figure? Reply with one word: red, blue, green, or yellow."
    else:
        question = "What SHAPE is the large central figure? Reply with one word: circle, square, triangle, or star."
    return f"{context}\nQuestion: {question}"


def parse_answer(text: str, task: Task) -> str | None:
    """Strict scorer: reject verbose or ambiguous responses rather than guess intent."""
    choices = COLORS if task == "color" else SHAPES
    candidate = re.sub(r"[\s.!?,'\"`]+$", "", text.strip().lower())
    candidate = re.sub(r"^[\s'\"`]+", "", candidate)
    return candidate if candidate in choices else None

