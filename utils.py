"""Helper utilities shared by the agents and the orchestrator."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_JSON_BLOCK = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the largest fenced code block from an LLM reply.

    If the reply has no fenced block, the whole text is treated as code.
    """
    blocks = _CODE_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return (text or "").strip() + "\n"


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from an LLM reply. Returns None if impossible."""
    text = (text or "").strip()
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def syntax_error(code: str) -> Optional[str]:
    """Return a readable message if `code` is not valid Python, else None."""
    if not code or not code.strip():
        return "The code is empty."
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"
    return None


def truncate(text: str, limit: int = 4000) -> str:
    """Shorten long text (e.g. test output) before sending it to an LLM."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


@dataclass
class TestResult:
    """Outcome of running the generated unit tests."""

    passed: bool
    output: str
    returncode: int
    timed_out: bool = False

    __test__ = False  # stop pytest from collecting this class


def run_tests(code: str, tests: str, timeout: int = 60) -> TestResult:
    """Run `tests` against `code` in a temporary directory using unittest.

    The code is saved as solution.py and the tests as test_solution.py.
    NOTE: this executes AI-generated code in a subprocess (not a hardened
    sandbox). Run the tool inside a container/VM for untrusted requirements.
    """
    with tempfile.TemporaryDirectory(prefix="ai_dev_team_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(code, encoding="utf-8")
        (tmp_path / "test_solution.py").write_text(tests, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", "test_solution"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(False, f"Tests timed out after {timeout} seconds.", -1, True)
        output = (proc.stdout or "") + (proc.stderr or "")
        return TestResult(proc.returncode == 0, output, proc.returncode)