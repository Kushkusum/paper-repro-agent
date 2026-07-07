from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, TypeVar

import groq
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Free-tier Groq models each have their own separate rate-limit bucket (tokens/minute and
# tokens/day). If the preferred model is rate-limited or a request is too large for its
# per-minute cap, fall back to the next model here rather than crashing the whole run.
DEFAULT_FALLBACKS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
_fallbacks_env = os.environ.get("GROQ_MODEL_FALLBACKS")
MODEL_FALLBACKS = [m.strip() for m in _fallbacks_env.split(",")] if _fallbacks_env else DEFAULT_FALLBACKS

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key from console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def _model_chain(preferred: str) -> list[str]:
    return [preferred] + [m for m in MODEL_FALLBACKS if m != preferred]


def _create_completion(preferred_model: str, **kwargs: Any):
    last_error: Exception | None = None
    for model in _model_chain(preferred_model):
        try:
            return client().chat.completions.create(model=model, **kwargs), model
        except groq.APIStatusError as e:
            last_error = e
            print(f"[llm] {model} unavailable ({e.__class__.__name__}), trying next fallback model...", file=sys.stderr)
    raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}") from last_error


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def call_text(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    resp, _ = _create_completion(
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system_prompt}\n\n"
        "You must respond with ONLY a single valid JSON object (no prose, no markdown fences) "
        f"conforming to this JSON schema:\n{schema_json}"
    )
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_prompt},
    ]
    last_error = ""
    for attempt in range(max_retries):
        resp, _ = _create_completion(
            model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        try:
            data = _extract_json(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"That response was invalid: {last_error}\nReturn ONLY the corrected JSON object.",
                }
            )
    raise RuntimeError(f"Failed to get valid structured output after {max_retries} attempts: {last_error}")
