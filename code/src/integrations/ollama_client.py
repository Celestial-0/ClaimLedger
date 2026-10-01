"""Minimal Ollama REST client for local LLM inference.

Uses only the Python standard library (urllib) so the ClaimLedger package
remains dependency-free.  Targets the ``/api/generate`` endpoint exposed by
a locally running Ollama server (default ``http://127.0.0.1:11434``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_BASE = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class OllamaResponse:
    """Wrapper around a single Ollama generation response."""

    model: str
    text: str
    total_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    thinking: str = ""

    @property
    def total_duration_ms(self) -> float:
        return self.total_duration_ns / 1_000_000

    @property
    def tokens_per_second(self) -> float:
        duration_s = self.total_duration_ns / 1_000_000_000
        if duration_s <= 0:
            return 0.0
        return self.eval_count / duration_s


@dataclass
class OllamaClient:
    """Lightweight Ollama API client.

    Parameters
    ----------
    model : str
        Ollama model tag (e.g. ``"phi4-mini"``, ``"gemma3:4b"``).
    base_url : str
        Ollama server URL.
    temperature : float
        Sampling temperature.  Use 0.0 for deterministic evaluation.
    system_prompt : str
        Optional system-level instruction prepended to every request.
    timeout_s : int
        HTTP request timeout in seconds.
    """

    model: str = "phi4-mini"
    base_url: str = _DEFAULT_BASE
    temperature: float = 0.0
    system_prompt: str = ""
    timeout_s: int = 120
    think: bool | None = None
    response_format: dict[str, Any] | str | None = None
    _options: dict[str, Any] = field(default_factory=dict)

    def generate(self, prompt: str, *, max_tokens: int = 512) -> OllamaResponse:
        """Send a single-turn generation request to ``/api/generate``."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                **self._options,
            },
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt
        if self.think is not None:
            payload["think"] = self.think
        if self.response_format is not None:
            payload["format"] = self.response_format

        data = self._post("/api/generate", payload)
        return OllamaResponse(
            model=data.get("model", self.model),
            text=data.get("response", ""),
            total_duration_ns=data.get("total_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            thinking=data.get("thinking", ""),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
    ) -> OllamaResponse:
        """Send a multi-turn chat request to ``/api/chat``."""
        full_messages = list(messages)
        if self.system_prompt:
            full_messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                **self._options,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.response_format is not None:
            payload["format"] = self.response_format

        data = self._post("/api/chat", payload)
        message = data.get("message", {})
        return OllamaResponse(
            model=data.get("model", self.model),
            text=message.get("content", ""),
            total_duration_ns=data.get("total_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            thinking=message.get("thinking", data.get("thinking", "")),
        )

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self) -> list[str]:
        """Return locally available model tags."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except (urllib.error.URLError, OSError, KeyError):
            return []

    def show_model(self) -> dict[str, Any]:
        """Return Ollama's local model metadata for run provenance."""
        try:
            shown = self._post("/api/show", {"name": self.model})
            tagged = next((item for item in self._tag_metadata() if item.get("name") == self.model), {})
            model_info = shown.get("model_info", {})
            keep_info = {
                key: model_info[key]
                for key in (
                    "general.architecture",
                    "general.parameter_count",
                    "general.quantization_version",
                    "general.size_label",
                    "general.file_type",
                    "qwen3.context_length",
                    "gemma3.context_length",
                    "phi4.context_length",
                )
                if key in model_info
            }
            details = shown.get("details", {})
            return {
                "name": self.model,
                "digest": tagged.get("digest"),
                "size": tagged.get("size"),
                "modified_at": tagged.get("modified_at"),
                "capabilities": shown.get("capabilities", []),
                "details": {
                    key: details[key]
                    for key in ("family", "parameter_size", "quantization_level", "format")
                    if key in details
                },
                "model_info": keep_info,
            }
        except (RuntimeError, json.JSONDecodeError):
            return {"name": self.model, "metadata_unavailable": True}

    def _tag_metadata(self) -> list[dict[str, Any]]:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [item for item in data.get("models", []) if isinstance(item, dict)]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama API error {exc.code} for {url}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}. Is the server running? ({exc.reason})"
            ) from exc
