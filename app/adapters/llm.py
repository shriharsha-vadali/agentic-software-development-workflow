"""
app/adapters/llm.py
-------------------
Provider-independent LLM client interface and implementations.
Includes OpenAILLMClient (async httpx) and deterministic MockLLMClient for testing.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class LLMAdapterError(Exception):
    """Base exception for LLM adapter failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMAuthError(LLMAdapterError):
    """Authentication or invalid API key error."""


class LLMTimeoutError(LLMAdapterError):
    """Request timeout communicating with the LLM API."""


class LLMResponseError(LLMAdapterError):
    """Invalid, truncated, or unparseable response from LLM."""


# =============================================================================
# Abstract Interface
# =============================================================================


class BaseLLMClient(ABC):
    """Abstract interface for LLM client providers."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a raw text completion."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Generate and parse structured output adhering to a Pydantic schema."""


# =============================================================================
# OpenAI-Compatible Implementation
# =============================================================================


class OpenAILLMClient(BaseLLMClient):
    """Async client for OpenAI and OpenAI-compatible Chat Completions APIs."""

    def __init__(
        self,
        api_key: str | SecretStr,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key: SecretStr = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._external_client = http_client is not None
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if not self._external_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        client = await self._get_client()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            resp = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM request timed out after {self.timeout}s") from exc
        except httpx.NetworkError as exc:
            raise LLMAdapterError(f"Network error contacting LLM service: {exc}") from exc

        if resp.status_code == 401:
            raise LLMAuthError(
                "LLM API authentication failed. Verify LLM_API_KEY.", status_code=401
            )
        if resp.status_code != 200:
            raise LLMResponseError(
                f"LLM API returned error status {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )

        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            raise LLMResponseError("Malformed completion response from LLM API") from exc

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        guidance = (
            f"You MUST respond ONLY with valid JSON conforming to this schema:\n{schema}\n"
            "Do NOT include markdown code fences or explanatory text outside the JSON."
        )
        combined_sys = f"{system_prompt}\n\n{guidance}" if system_prompt else guidance
        raw = await self.generate_text(prompt, system_prompt=combined_sys, temperature=temperature)

        # Strip accidental code fences
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return response_model.model_validate_json(cleaned)
        except Exception as exc:
            raise LLMResponseError(
                f"Failed to parse LLM structured output into {response_model.__name__}: {exc}\n"
                f"Raw: {raw[:200]}"
            ) from exc


# =============================================================================
# Deterministic Mock Implementation for Testing & Offline Execution
# =============================================================================


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client that returns structured responses without API calls."""

    def __init__(self, predefined_responses: dict[str, Any] | None = None) -> None:
        self.predefined_responses = predefined_responses or {}
        self.call_history: list[dict[str, Any]] = []

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        self.call_history.append({"prompt": prompt, "system": system_prompt})
        for key, resp in self.predefined_responses.items():
            if key in prompt:
                return json.dumps(resp) if isinstance(resp, (dict, list)) else str(resp)
        return "Mock LLM text response."

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        self.call_history.append({"prompt": prompt, "model": response_model.__name__})

        # Check for predefined match
        for key, resp in self.predefined_responses.items():
            if key in prompt:
                if isinstance(resp, dict):
                    return response_model.model_validate(resp)
                if isinstance(resp, str):
                    return response_model.model_validate_json(resp)

        # Fallback to sensible default instances for known models
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        sample: dict[str, Any] = {}

        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")
            if prop_type == "string":
                sample[prop_name] = f"Mock {prop_name}"
            elif prop_type == "integer":
                sample[prop_name] = 1
            elif prop_type == "array":
                # Handle array items
                item_spec = prop_spec.get("items", {})
                if item_spec.get("type") == "string":
                    sample[prop_name] = [f"Mock {prop_name} item"]
                else:
                    sample[prop_name] = []
            elif prop_type == "boolean":
                sample[prop_name] = True
            elif prop_type == "object":
                sample[prop_name] = {}

        try:
            return response_model.model_validate(sample)
        except Exception:
            # If default sample doesn't validate, try constructing empty
            return response_model.model_validate({})


def get_llm_client(settings: Settings = get_settings()) -> BaseLLMClient:
    """Dependency / factory returning configured LLM client."""
    # If API key is placeholder or empty, fall back to MockLLMClient for testing safety
    token_str = settings.llm_api_key.get_secret_value().strip()
    if not token_str or "placeholder" in token_str.lower() or "your_" in token_str.lower():
        return MockLLMClient()

    return OpenAILLMClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=60.0,
    )
