"""Gemini API client wrapper for the local agent."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

try:  # pragma: no cover - optional dependency during scaffolding
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
    from google.api_core.exceptions import ResourceExhausted
except Exception:  # pragma: no cover - optional dependency during scaffolding
    genai = None  # type: ignore
    HarmCategory = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore
    ResourceExhausted = None  # type: ignore

from .config import GeminiConfig


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    raw: Any


class GeminiLLM:
    def __init__(self, config: GeminiConfig):
        if genai is None:  # pragma: no cover - optional dependency
            raise ImportError("google-generativeai is required to use the Gemini provider")
        self.config = config
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("Gemini API key not found. Set GOOGLE_API_KEY or pass api_key in GeminiConfig.")
        genai.configure(api_key=api_key)
        self._generation_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_output_tokens,
        }
        if HarmCategory is not None and HarmBlockThreshold is not None:
            self._safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
        else:
            self._safety_settings = None

    def __call__(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        model_kwargs = {
            "model_name": self.config.model,
            "system_instruction": system_prompt,
            "generation_config": self._generation_config,
        }
        if self._safety_settings is not None:
            model_kwargs["safety_settings"] = self._safety_settings
        model = genai.GenerativeModel(**model_kwargs)
        response = None
        last_exc: Optional[Exception] = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = model.generate_content(user_prompt)
                break
            except Exception as exc:  # pragma: no cover - surface API errors clearly
                can_retry = ResourceExhausted is not None and isinstance(exc, ResourceExhausted)
                if can_retry and attempt < max_attempts:
                    delay = getattr(exc, "retry_delay", None)
                    seconds = None
                    if delay is not None:
                        total_seconds = getattr(delay, "total_seconds", None)
                        if callable(total_seconds):
                            seconds = float(total_seconds())
                        elif total_seconds is not None:
                            seconds = float(total_seconds)
                        else:
                            seconds = float(getattr(delay, "seconds", 0) + getattr(delay, "nanos", 0) / 1e9)
                    sleep_for = max(5.0 * attempt, seconds or 0.0)
                    time.sleep(sleep_for)
                    last_exc = exc
                    continue
                last_exc = exc
                break
        if response is None:
            raise LLMError(f"Gemini API call failed: {last_exc}") from last_exc

        parts: List[str] = []
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                value = getattr(part, "text", None)
                if value:
                    parts.append(value)
        text = "".join(parts).strip()
        if not text:
            try:
                raw_text = response.text  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                raw_text = None
            if isinstance(raw_text, str) and raw_text.strip():
                text = raw_text.strip()

        if not text:
            prompt_feedback = getattr(response, "prompt_feedback", None)
            finish_reasons = [getattr(c, "finish_reason", None) for c in getattr(response, "candidates", [])]
            raise LLMError(
                "Gemini returned an empty response."
                + (f" Prompt feedback: {prompt_feedback}" if prompt_feedback else "")
                + (f" Finish reasons: {finish_reasons}" if finish_reasons else "")
            )
        return LLMResponse(content=text, raw=response)


def ensure_json(response: LLMResponse) -> Any:
    content = response.content.strip() if isinstance(response.content, str) else response.content
    if isinstance(content, str) and content.startswith("```"):
        stripped = content.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[len("json"):].lstrip()
        content = stripped
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMError(f"Failed to parse LLM JSON: {exc}\nResponse: {response.content}") from exc
