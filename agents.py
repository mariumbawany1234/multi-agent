"""The four agents of the virtual development team."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from llm import LLMClient
from utils import extract_code, extract_json, truncate

SEVERITIES = ("critical", "high", "medium", "low")


class Agent:
    """Base class: a named role with a system prompt and an LLM backend."""

    name = "Agent"
    role = "Generic agent"
    system_prompt = "You are a helpful assistant."

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _ask(self, prompt: str, task: str) -> str:
        return self.llm.complete(self.system_prompt, prompt, task)


# --------------------------------------------------------------------------- #
# 1. Project Planner
# --------------------------------------------------------------------------- #
class ProjectPlanner(Agent):
    name = "Project Planner"
    role = "Software Architect"
    system_prompt = (
        "You are a senior software architect. Analyse the user's software requirement and "
        "produce an implementation plan for a single Python module. "
        "Reply with ONLY a JSON object with these keys: "
        '"summary" (string), "assumptions" (list of strings), '
        '"functions" (list of objects with "name", "signature", "description"), '
        '"edge_cases" (list of strings), "steps" (ordered list of strings). '
        "Use only the Python standard library."
    )

    def create_plan(self, requirement: str) -> Dict[str, Any]:
        reply = self._ask(f"Software requirement:\n{requirement}", "planner")
        data = extract_json(reply) or {}
        functions: List[Dict[str, str]] = []
        for item in data.get("functions", []) or []:
            if isinstance(item, dict):
                functions.append(
                    {
                        "name": str(item.get("name", "")),
                        "signature": str(item.get("signature", "")),
                        "description": str(item.get("description", "")),
                    }
                )
            else:
                functions.append({"name": str(item), "signature": "", "description": ""})
        return {
            "summary": str(data.get("summary") or requirement),
            "assumptions": [str(x) for x in data.get("assumptions", []) or []],
            "functions": functions,
            "edge_cases": [str(x) for x in data.get("edge_cases", []) or []],
            "steps": [str(x) for x in data.get("steps", []) or []],
        }


# --------------------------------------------------------------------------- #
# 2. Python Developer
# --------------------------------------------------------------------------- #
class PythonDeveloper(Agent):
    name = "Python Developer"
    role = "Backend Developer"
    system_prompt = (
        "You are a senior Python backend developer. Write ONE complete, modular Python module "
        "that implements the plan. Rules: Python 3.9+ compatible, standard library only, "
        "type hints, docstrings, input validation, custom exceptions and clear error messages. "
        "No input(), no network access and no side effects at import time. "
        "The module will be saved as solution.py. "
        "Reply with a single ```python code block and nothing else."
    )

    def write_code(self, requirement: str, plan: Dict[str, Any]) -> str:
        prompt = f"Requirement:\n{requirement}\n\nImplementation plan:\n{json.dumps(plan, indent=2)}"
        return extract_code(self._ask(prompt, "developer"))

    def revise_code(self, requirement: str, plan: Dict[str, Any], code: str, feedback: str) -> str:
        prompt = (
            f"Requirement:\n{requirement}\n\nImplementation plan:\n{json.dumps(plan, indent=2)}\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Problems to fix:\n{truncate(feedback)}\n\n"
            "Return the complete corrected module in a single ```python block."
        )
        return extract_code(self._ask(prompt, "developer_revise"))


# --------------------------------------------------------------------------- #
# 3. QA Engineer
# --------------------------------------------------------------------------- #
class QAEngineer(Agent):
    name = "QA Engineer"
    role = "Software Tester"
    system_prompt = (
        "You are a QA engineer. Write unit tests using ONLY the standard library unittest module. "
        "The code under test lives in solution.py, so import from `solution`. "
        "Cover normal cases, edge cases, invalid input and error handling. "
        "Tests must be deterministic and must not use the network or files. "
        "End the file with: if __name__ == '__main__': unittest.main(). "
        "Reply with a single ```python code block and nothing else."
    )

    def write_tests(self, requirement: str, plan: Dict[str, Any], code: str) -> str:
        prompt = (
            f"Requirement:\n{requirement}\n\nPlan (functions and edge cases):\n{json.dumps(plan, indent=2)}\n\n"
            f"Code under test (solution.py):\n```python\n{code}\n```"
        )
        return extract_code(self._ask(prompt, "qa"))

    def revise_tests(self, tests: str, problem: str) -> str:
        prompt = (
            f"These tests have a problem:\n{truncate(problem)}\n\n"
            f"Tests:\n```python\n{tests}\n```\n\nReturn the corrected test file in a single ```python block."
        )
        return extract_code(self._ask(prompt, "qa_revise"))


# --------------------------------------------------------------------------- #
# 4. Code Reviewer
# --------------------------------------------------------------------------- #
class CodeReviewer(Agent):
    name = "Code Reviewer"
    role = "Senior Developer"
    system_prompt = (
        "You are a senior developer doing a strict code review. Look for bugs, unhandled errors, "
        "security problems, poor design and missing tests. "
        "Reply with ONLY a JSON object: "
        '{"summary": str, "verdict": "approve" or "needs_changes", '
        '"issues": [{"severity": "critical|high|medium|low", "description": str, "suggestion": str}]}.'
    )

    def review(
        self,
        requirement: str,
        code: str,
        tests: str,
        test_output: str,
        tests_passed: bool,
    ) -> Dict[str, Any]:
        prompt = (
            f"Requirement:\n{requirement}\n\nCode:\n```python\n{code}\n```\n\n"
            f"Tests:\n```python\n{tests}\n```\n\n"
            f"Test run ({'PASSED' if tests_passed else 'FAILED'}):\n{truncate(test_output, 3000)}"
        )
        reply = self._ask(prompt, "reviewer")
        data = extract_json(reply)
        if data is None:
            return {
                "summary": "The reviewer reply could not be parsed: " + truncate(reply, 300),
                "verdict": "needs_changes",
                "issues": [],
            }
        issues: List[Dict[str, str]] = []
        for item in data.get("issues", []) or []:
            if isinstance(item, dict):
                severity = str(item.get("severity", "low")).lower()
                issues.append(
                    {
                        "severity": severity if severity in SEVERITIES else "low",
                        "description": str(item.get("description", "")),
                        "suggestion": str(item.get("suggestion", "")),
                    }
                )
            else:
                issues.append({"severity": "low", "description": str(item), "suggestion": ""})
        return {
            "summary": str(data.get("summary", "")),
            "verdict": str(data.get("verdict", "needs_changes")),
            "issues": issues,
        }
