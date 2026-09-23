import json
import logging
import re
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from ike.core.config import get_settings

logger = logging.getLogger(__name__)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)

    def generate(
        self,
        *,
        system: str,
        user: str,
        strong: bool = False,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
        on_delta: Callable[[str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> LLMResult:
        model = self.settings.llm_strong_model if strong else self.settings.llm_fast_model
        effort = self.settings.llm_strong_reasoning if strong else self.settings.llm_fast_reasoning
        request: dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": user,
            "reasoning": {"effort": effort},
            "max_output_tokens": max_output_tokens or self.settings.llm_max_output_tokens,
            "store": False,
        }
        if json_mode:
            request["text"] = {"format": {"type": "json_object"}}
        if on_delta is None:
            if cancel_check:
                cancel_check()
            response = self.client.responses.create(**request)
            if cancel_check:
                cancel_check()
            usage = getattr(response, "usage", None)
            return LLMResult(
                text=response.output_text.strip(),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            )

        # Stream only final answer prose. Planning/verification JSON calls remain non-streamed.
        parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stream = self.client.responses.create(**request, stream=True)
        try:
            for event in stream:
                if cancel_check:
                    cancel_check()
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = str(getattr(event, "delta", "") or "")
                    if delta:
                        parts.append(delta)
                        on_delta(delta)
                elif event_type == "response.completed":
                    response = getattr(event, "response", None)
                    usage = getattr(response, "usage", None)
                    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        return LLMResult(text="".join(parts).strip(), input_tokens=input_tokens, output_tokens=output_tokens)


    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[StructuredModelT],
        strong: bool = False,
        max_output_tokens: int = 1600,
    ) -> tuple[StructuredModelT, LLMResult]:
        """Generate schema-constrained planner output with SDK/Pydantic validation.

        This is the preferred path for query/evidence/applicability controllers.  It uses
        Responses API Structured Outputs so syntactically valid JSON is not mistaken for a
        valid domain object.  ``generate_json`` remains available during staged migration.
        """

        model = self.settings.llm_strong_model if strong else self.settings.llm_fast_model
        effort = self.settings.llm_strong_reasoning if strong else self.settings.llm_fast_reasoning
        response = self.client.responses.parse(
            model=model,
            instructions=system,
            input=user,
            reasoning={"effort": effort},
            max_output_tokens=max_output_tokens,
            store=False,
            text_format=schema_model,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError(f"Model returned no parsed {schema_model.__name__} payload")
        if not isinstance(parsed, schema_model):
            parsed = schema_model.model_validate(parsed)
        usage = getattr(response, "usage", None)
        result = LLMResult(
            text=str(getattr(response, "output_text", "") or "").strip(),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
        return parsed, result

    def generate_json(self, *, system: str, user: str, strong: bool = False, max_output_tokens: int = 1600) -> tuple[Any, LLMResult]:
        result = self.generate(
            system=system,
            user=user + "\n\nReturn only valid JSON. Do not wrap it in Markdown.",
            strong=strong,
            max_output_tokens=max_output_tokens,
            json_mode=True,
        )
        text = result.text.strip()
        match = _JSON_BLOCK.search(text)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text), result
        except json.JSONDecodeError as exc:
            logger.warning("llm_json_parse_failed", extra={"payload": text[:1000]})
            raise ValueError("Model returned invalid JSON") from exc
