"""Coordinates the agents like a small software development team."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents import CodeReviewer, ProjectPlanner, PythonDeveloper, QAEngineer
from llm import LLMClient
from utils import TestResult, run_tests, syntax_error, truncate

BLOCKING_SEVERITIES = ("critical", "high", "medium")


@dataclass
class DevResult:
    requirement: str
    plan: Dict[str, Any]
    code: str
    tests: str
    review: Dict[str, Any]
    test_history: List[Dict[str, Any]] = field(default_factory=list)
    final_passed: bool = False
    report: str = ""
    output_dir: Optional[Path] = None


class DevTeam:
    """Planner -> Developer -> QA -> Reviewer -> final code + report."""

    def __init__(
        self,
        llm: LLMClient,
        max_fix_rounds: int = 2,
        test_timeout: int = 60,
        log: Callable[[str], None] = print,
    ) -> None:
        self.planner = ProjectPlanner(llm)
        self.developer = PythonDeveloper(llm)
        self.qa = QAEngineer(llm)
        self.reviewer = CodeReviewer(llm)
        self.max_fix_rounds = max(0, max_fix_rounds)
        self.test_timeout = test_timeout
        self.log = log

    # ------------------------------------------------------------------ #
    def run(self, requirement: str, output_dir: Optional[Path] = None) -> DevResult:
        requirement = (requirement or "").strip()
        if not requirement:
            raise ValueError("The software requirement must not be empty.")

        # Step 1 - plan
        self.log(f"[1/5] {self.planner.name} ({self.planner.role}) is preparing the plan...")
        plan = self.planner.create_plan(requirement)

        # Step 2 - code
        self.log(f"[2/5] {self.developer.name} ({self.developer.role}) is writing code...")
        code = self._valid_code(self.developer.write_code(requirement, plan), requirement, plan)

        # Step 3 - tests
        self.log(f"[3/5] {self.qa.name} ({self.qa.role}) is writing tests...")
        tests = self._valid_tests(self.qa.write_tests(requirement, plan, code))

        history: List[Dict[str, Any]] = []
        result = self._run_and_record("initial run", code, tests, history)

        for round_no in range(1, self.max_fix_rounds + 1):
            if result.passed:
                break
            self.log(f"      Tests failed - developer fix round {round_no}/{self.max_fix_rounds}...")
            fixed = self.developer.revise_code(requirement, plan, code, "Unit tests failed:\n" + result.output)
            if syntax_error(fixed):
                break
            code = fixed
            result = self._run_and_record(f"after fix round {round_no}", code, tests, history)

        # Step 4 - review
        self.log(f"[4/5] {self.reviewer.name} ({self.reviewer.role}) is reviewing...")
        review = self.reviewer.review(requirement, code, tests, result.output, result.passed)

        # Step 5 - apply serious review findings, then finalise
        self.log("[5/5] Finalising code and generating the report...")
        blocking = [i for i in review["issues"] if i["severity"] in BLOCKING_SEVERITIES]
        if blocking:
            feedback = "\n".join(
                f"- [{i['severity']}] {i['description']} Suggestion: {i['suggestion']}" for i in blocking
            )
            candidate = self.developer.revise_code(requirement, plan, code, "Code review findings:\n" + feedback)
            if not syntax_error(candidate):
                cand_result = self._run_and_record("after review fixes", candidate, tests, history)
                # Keep the candidate only if it does not make things worse.
                if cand_result.passed or not result.passed:
                    code, result = candidate, cand_result

        dev_result = DevResult(
            requirement=requirement,
            plan=plan,
            code=code,
            tests=tests,
            review=review,
            test_history=history,
            final_passed=result.passed,
        )
        dev_result.report = self._build_report(dev_result, result)
        if output_dir is not None:
            dev_result.output_dir = self._save(dev_result, Path(output_dir))
        return dev_result

    # ------------------------------------------------------------------ #
    def _valid_code(self, code: str, requirement: str, plan: Dict[str, Any]) -> str:
        for _ in range(self.max_fix_rounds + 1):
            err = syntax_error(code)
            if not err:
                return code
            code = self.developer.revise_code(requirement, plan, code, f"SyntaxError: {err}")
        err = syntax_error(code)
        if err:
            raise RuntimeError(f"The developer could not produce valid Python: {err}")
        return code

    def _valid_tests(self, tests: str) -> str:
        for _ in range(self.max_fix_rounds + 1):
            err = syntax_error(tests)
            if not err:
                return tests
            tests = self.qa.revise_tests(tests, f"SyntaxError: {err}")
        err = syntax_error(tests)
        if err:
            raise RuntimeError(f"The QA engineer could not produce valid tests: {err}")
        return tests

    def _run_and_record(self, label: str, code: str, tests: str, history: List[Dict[str, Any]]) -> TestResult:
        result = run_tests(code, tests, timeout=self.test_timeout)
        history.append({"label": label, "passed": result.passed, "output": result.output})
        self.log(f"      Tests {label}: {'PASSED' if result.passed else 'FAILED'}")
        return result

    # ------------------------------------------------------------------ #
    def _build_report(self, r: DevResult, final: TestResult) -> str:
        plan = r.plan
        lines: List[str] = [
            "# Development Report",
            "",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            "## 1. Requirement",
            "",
            r.requirement,
            "",
            "## 2. Implementation Plan (Project Planner)",
            "",
            f"**Summary:** {plan['summary']}",
            "",
        ]
        if plan["assumptions"]:
            lines += ["**Assumptions:**", ""] + [f"- {a}" for a in plan["assumptions"]] + [""]
        if plan["functions"]:
            lines += ["**Planned functions:**", ""]
            for f in plan["functions"]:
                sig = f"`{f['signature']}`" if f["signature"] else f"`{f['name']}`"
                lines.append(f"- {sig} - {f['description']}")
            lines.append("")
        if plan["edge_cases"]:
            lines += ["**Edge cases to handle:**", ""] + [f"- {e}" for e in plan["edge_cases"]] + [""]
        if plan["steps"]:
            lines += ["**Steps:**", ""] + [f"{n}. {s}" for n, s in enumerate(plan["steps"], 1)] + [""]

        test_count = sum(1 for ln in r.tests.splitlines() if ln.strip().startswith("def test_"))
        lines += [
            "## 3. Implementation (Python Developer)",
            "",
            f"Final module: `solution.py` ({len(r.code.splitlines())} lines).",
            "",
            "## 4. Testing (QA Engineer)",
            "",
            f"Test file: `test_solution.py` (about {test_count} test methods).",
            "",
            "| Run | Result |",
            "|---|---|",
        ]
        lines += [f"| {h['label']} | {'PASSED' if h['passed'] else 'FAILED'} |" for h in r.test_history]
        lines.append("")
        if not final.passed:
            lines += ["**Last failing output:**", "", "```text", truncate(final.output, 2000), "```", ""]

        issues = r.review["issues"]
        lines += [
            "## 5. Code Review (Code Reviewer)",
            "",
            f"**Verdict:** {r.review['verdict']}",
            "",
            f"**Summary:** {r.review['summary']}",
            "",
        ]
        if issues:
            lines += ["| Severity | Issue | Recommendation |", "|---|---|---|"]
            for i in issues:
                desc = i["description"].replace("|", "/").replace("\n", " ")
                sug = i["suggestion"].replace("|", "/").replace("\n", " ")
                lines.append(f"| {i['severity']} | {desc} | {sug} |")
            lines.append("")
        else:
            lines += ["No issues reported.", ""]

        status = "All tests passed." if r.final_passed else "Some tests FAILED - manual review needed."
        lines += ["## 6. Final Status", "", f"**{status}**", "", "## 7. Final Code", "", "```python", r.code.rstrip(), "```", ""]
        return "\n".join(lines)

    def _save(self, r: DevResult, base_dir: Path) -> Path:
        out = base_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        (out / "solution.py").write_text(r.code, encoding="utf-8")
        (out / "test_solution.py").write_text(r.tests, encoding="utf-8")
        (out / "plan.json").write_text(json.dumps(r.plan, indent=2), encoding="utf-8")
        (out / "review.json").write_text(json.dumps(r.review, indent=2), encoding="utf-8")
        (out / "report.md").write_text(r.report, encoding="utf-8")
        return out
