"""LLM backends.

* GroqClient      - real agents powered by the Groq API (default, no extra packages).
* AnthropicClient - real agents powered by the Claude API.
* MockLLM         - offline demo mode (no API key). It returns canned answers for
                    a calculator requirement so you can verify the whole pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Minimal interface every backend must implement."""

    @abstractmethod
    def complete(self, system: str, prompt: str, task: str = "") -> str:
        """Return the model's text reply."""


class AnthropicClient(LLMClient):
    """Claude API backend. Needs `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, max_tokens: int = 4096) -> None:
        try:
            import anthropic  # imported lazily so mock mode needs no packages
        except ImportError as exc:
            raise RuntimeError("The 'anthropic' package is missing. Run: pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it, or run with --mock for the offline demo."
            )
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str, task: str = "") -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


# --------------------------------------------------------------------------- #
# Offline demo backend
# --------------------------------------------------------------------------- #

_MOCK_PLAN = """```json
{
  "summary": "A safe calculator module with basic arithmetic and clear error handling.",
  "assumptions": ["Operands are int or float (bool is rejected).", "Only the standard library is used."],
  "functions": [
    {"name": "add", "signature": "add(a, b)", "description": "Return a + b."},
    {"name": "subtract", "signature": "subtract(a, b)", "description": "Return a - b."},
    {"name": "multiply", "signature": "multiply(a, b)", "description": "Return a * b."},
    {"name": "divide", "signature": "divide(a, b)", "description": "Return a / b; raise on zero divisor."},
    {"name": "power", "signature": "power(a, b)", "description": "Return a ** b; reject complex or undefined results."}
  ],
  "edge_cases": ["division by zero", "non-numeric input", "bool input", "0 to a negative power", "negative base with fractional exponent", "very large numbers"],
  "steps": ["Define CalculatorError", "Add operand validation helper", "Implement the five operations", "Write unit tests", "Review and finalise"]
}
```"""

_MOCK_CODE = '''```python
"""A small, safe calculator module."""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


class CalculatorError(Exception):
    """Raised when an operation cannot be performed."""


def _validate(*values: object) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalculatorError(f"Invalid operand: {value!r}")


def add(a: Number, b: Number) -> Number:
    """Return a + b."""
    _validate(a, b)
    return a + b


def subtract(a: Number, b: Number) -> Number:
    """Return a - b."""
    _validate(a, b)
    return a - b


def multiply(a: Number, b: Number) -> Number:
    """Return a * b."""
    _validate(a, b)
    return a * b


def divide(a: Number, b: Number) -> float:
    """Return a / b. Raises CalculatorError if b is zero."""
    _validate(a, b)
    if b == 0:
        raise CalculatorError("Division by zero is not allowed.")
    return a / b


def power(a: Number, b: Number) -> Number:
    """Return a ** b. Raises CalculatorError for undefined or non-real results."""
    _validate(a, b)
    if a == 0 and b < 0:
        raise CalculatorError("Zero cannot be raised to a negative power.")
    try:
        result = a ** b
    except (OverflowError, ZeroDivisionError) as exc:
        raise CalculatorError(f"Power operation failed: {exc}") from exc
    if isinstance(result, complex):
        raise CalculatorError("Result is not a real number.")
    return result
```'''

_MOCK_TESTS = '''```python
import unittest

from solution import CalculatorError, add, divide, multiply, power, subtract


class TestBasicOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertAlmostEqual(add(0.1, 0.2), 0.3)

    def test_subtract(self):
        self.assertEqual(subtract(5, 8), -3)

    def test_multiply(self):
        self.assertEqual(multiply(4, 2.5), 10.0)

    def test_divide(self):
        self.assertEqual(divide(9, 3), 3)

    def test_power(self):
        self.assertEqual(power(2, 10), 1024)
        self.assertEqual(power(2, -1), 0.5)


class TestEdgeCases(unittest.TestCase):
    def test_divide_by_zero(self):
        with self.assertRaises(CalculatorError):
            divide(1, 0)
        with self.assertRaises(CalculatorError):
            divide(1, 0.0)

    def test_invalid_types(self):
        for bad in ("1", None, [1], True):
            with self.assertRaises(CalculatorError):
                add(bad, 1)

    def test_zero_to_negative_power(self):
        with self.assertRaises(CalculatorError):
            power(0, -1)

    def test_negative_base_fractional_exponent(self):
        with self.assertRaises(CalculatorError):
            power(-8, 0.5)

    def test_large_numbers(self):
        self.assertEqual(multiply(10**12, 10**12), 10**24)

    def test_float_overflow(self):
        with self.assertRaises(CalculatorError):
            power(10.0, 10000)


if __name__ == "__main__":
    unittest.main()
```'''

_MOCK_REVIEW = """```json
{
  "summary": "Clean, well-structured module with proper validation and error handling.",
  "verdict": "approve",
  "issues": [
    {"severity": "low", "description": "NaN and infinity are accepted as valid operands.", "suggestion": "Use math.isfinite() in _validate if only finite numbers should be allowed."},
    {"severity": "low", "description": "No logging of failed operations.", "suggestion": "Add the logging module if this will be used in a service."}
  ]
}
```"""


class MockLLM(LLMClient):
    """Offline stand-in that always answers with the calculator demo."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = max(0.0, delay)  # optional pause so the UI can show live progress

    def complete(self, system: str, prompt: str, task: str = "") -> str:
        if self.delay:
            time.sleep(self.delay)
        if task == "planner":
            return _MOCK_PLAN
        if task in ("developer", "developer_revise"):
            return _MOCK_CODE
        if task in ("qa", "qa_revise"):
            return _MOCK_TESTS
        if task == "reviewer":
            return _MOCK_REVIEW
        raise ValueError(f"MockLLM does not know the task {task!r}")


# --------------------------------------------------------------------------- #
# Groq backend (OpenAI-compatible chat completions API, standard library only)
# --------------------------------------------------------------------------- #
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_MAX_WAIT_SECONDS = 60
_SERVER_ERRORS = (500, 502, 503, 504)
_MAX_ATTEMPTS = 5

# Groq counts (prompt tokens + max_tokens) against the per-minute limit, so every agent only
# asks for the output size it realistically needs. GROQ_MAX_TOKENS overrides all of these.
_TASK_MAX_TOKENS = {
    "planner": 1200,
    "reviewer": 1200,
    "developer": 3000,
    "developer_revise": 3000,
    "qa": 3000,
    "qa_revise": 3000,
}
DEFAULT_TASK_MAX_TOKENS = 3000

_log = logging.getLogger("ai_dev_team.groq")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _retry_delay(retry_after: Optional[str], message: str) -> Optional[float]:
    """Seconds Groq asks us to wait: from the Retry-After header or the 'try again in ...' text."""
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    match = re.search(r"try again in\s+([0-9hms.]+)", message or "", re.IGNORECASE)
    if not match:
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1))
    if not parts:
        return None
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in parts)


class GroqClient(LLMClient):
    """Groq API backend. Reads GROQ_API_KEY (and optional GROQ_MODEL) from the environment."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is empty. Paste your key into the .env file (GROQ_API_KEY=...) "
                "or use demo mode (--mock)."
            )
        self.model = model or os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
        self.base_url = (os.environ.get("GROQ_BASE_URL", "").strip() or DEFAULT_GROQ_URL).rstrip("/")
        self.max_tokens = self._int_setting(max_tokens, "GROQ_MAX_TOKENS")
        self.timeout = self._int_setting(timeout, "GROQ_TIMEOUT") or 120
        self.max_wait = self._int_setting(None, "GROQ_MAX_WAIT") or DEFAULT_MAX_WAIT_SECONDS

    @staticmethod
    def _int_setting(value: Optional[int], env_name: str) -> Optional[int]:
        raw = value if value else os.environ.get(env_name, "")
        try:
            number = int(raw)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def complete(self, system: str, prompt: str, task: str = "") -> str:
        limit = self.max_tokens or _TASK_MAX_TOKENS.get(task, DEFAULT_TASK_MAX_TOKENS)
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": limit,
                "temperature": 0.2,
            }
        ).encode("utf-8")
        data = self._post(payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Groq returned an unexpected response format.") from exc
        if not isinstance(content, str) or not content.strip():
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    "The model ran out of output tokens before answering. "
                    "Set GROQ_MAX_TOKENS to a larger value (for example 6000) or use llama-3.3-70b-versatile."
                )
            raise RuntimeError("Groq returned an empty reply.")
        return content

    def _post(self, payload: bytes) -> dict:
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ai-dev-team/1.0",
            },
        )
        for attempt in range(_MAX_ATTEMPTS):
            last_attempt = attempt == _MAX_ATTEMPTS - 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = self._error_detail(exc)
                if exc.code == 429:
                    delay = _retry_delay(exc.headers.get("Retry-After") if exc.headers else None, detail)
                    if delay is None:
                        delay = 2.0 * (attempt + 1)
                    if delay > self.max_wait or last_attempt:
                        raise RuntimeError(
                            f"Groq rate limit reached: {detail} "
                            f"The free-tier limit resets every minute: wait about {math.ceil(delay)}s and try again, "
                            "set GROQ_MODEL to a model with a higher limit (for example llama-3.3-70b-versatile), "
                            "or upgrade your Groq plan."
                        ) from exc
                    _log.warning("Groq rate limit reached, waiting %.1fs before retrying...", delay)
                    time.sleep(delay + 0.5)
                    continue
                if exc.code in _SERVER_ERRORS and not last_attempt:
                    time.sleep(2 * (attempt + 1))
                    continue
                if exc.code in (401, 403):
                    detail += " (check that GROQ_API_KEY in .env is correct)"
                raise RuntimeError(f"Groq API error {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Cannot reach the Groq API: {exc.reason}") from exc
            except OSError as exc:  # timeouts, connection resets
                raise RuntimeError(f"Network error while calling Groq: {exc}") from exc
            except ValueError as exc:
                raise RuntimeError("Groq returned invalid JSON.") from exc
        raise RuntimeError("Groq request failed after retries.")

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            message = body.get("error", {}).get("message")
            if message:
                return str(message)
        except Exception:
            pass
        return str(exc.reason or "request failed")


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
def configured_provider() -> Optional[str]:
    """Return 'groq' or 'anthropic' based on LLM_PROVIDER or whichever key is set."""
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in ("groq", "anthropic"):
        return explicit
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return None


def has_api_key() -> bool:
    """True if the key needed by the selected provider is actually filled in."""
    provider = configured_provider()
    if provider == "groq":
        return bool(os.environ.get("GROQ_API_KEY", "").strip())
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return False


def create_llm(provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None) -> LLMClient:
    """Build the real LLM client chosen by `provider` or by the environment."""
    provider = (provider or configured_provider() or "").lower()
    if provider == "groq":
        return GroqClient(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    if not provider:
        raise RuntimeError(
            "No API key found. Paste your key into the .env file as GROQ_API_KEY=..., "
            "or use demo mode (--mock)."
        )
    raise RuntimeError(f"Unknown provider {provider!r}. Use 'groq' or 'anthropic'.")