from __future__ import annotations

from functools import lru_cache
from typing import Literal

Label = Literal["entailment", "contradiction", "neutral"]
_VALID_LABELS: set[str] = {"entailment", "contradiction", "neutral"}

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model, torch


def classify_pair(premise: str, hypothesis: str) -> tuple[Label, float]:
    """Classify the relationship between two claims as entailment/contradiction/neutral.

    Runs a small pretrained NLI cross-encoder (no fine-tuning) on CPU. Returns the predicted
    label and its confidence (softmax probability).
    """
    tokenizer, model, torch = _load_model()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    raw_label = model.config.id2label[idx].lower()
    confidence = float(probs[idx])
    if raw_label == "entailment":
        return "entailment", confidence
    if raw_label == "contradiction":
        return "contradiction", confidence
    if raw_label == "neutral":
        return "neutral", confidence
    raise ValueError(f"Unexpected NLI label {raw_label!r} from {MODEL_NAME}, expected one of {_VALID_LABELS}")
