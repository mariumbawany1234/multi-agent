"""Command-line entry point for the multi-agent development team.

Examples:
    python main.py --mock
    python main.py "Create a function that validates email addresses"
    python main.py --file requirement.txt --output results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm import LLMClient, MockLLM, create_llm
from orchestrator import DevTeam
from utils import load_env

DEFAULT_MOCK_REQUIREMENT = "Build a calculator module with add, subtract, multiply, divide and power."


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI multi-agent software development team")
    parser.add_argument("requirement", nargs="?", help="Software requirement as text")
    parser.add_argument("-f", "--file", help="Read the requirement from a text file")
    parser.add_argument("-o", "--output", default="output", help="Output folder (default: output)")
    parser.add_argument("-m", "--model", help="Model name (default: GROQ_MODEL / ANTHROPIC_MODEL from .env)")
    parser.add_argument("--provider", choices=["groq", "anthropic"], help="LLM provider (default: from .env)")
    parser.add_argument("--max-fix-rounds", type=int, default=2, help="Max automatic fix rounds (default: 2)")
    parser.add_argument("--mock", action="store_true", help="Offline demo mode (no API key needed)")
    return parser.parse_args(argv)


def read_requirement(args: argparse.Namespace) -> str:
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            sys.exit(f"Cannot read requirement file: {exc}")
    if args.requirement:
        return args.requirement.strip()
    if args.mock:
        return DEFAULT_MOCK_REQUIREMENT
    try:
        return input("Enter your software requirement: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nNo requirement given.")


def build_llm(args: argparse.Namespace) -> LLMClient:
    if args.mock:
        print("Running in OFFLINE DEMO mode (canned calculator answers, no API calls).\n")
        return MockLLM()
    try:
        return create_llm(args.provider, args.model)
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")


def main(argv=None) -> int:
    load_env(Path(__file__).resolve().parent / ".env")
    args = parse_args(argv)
    requirement = read_requirement(args)
    if not requirement:
        print("Error: the requirement is empty.", file=sys.stderr)
        return 2

    team = DevTeam(build_llm(args), max_fix_rounds=args.max_fix_rounds)
    try:
        result = team.run(requirement, output_dir=Path(args.output))
    except (RuntimeError, ValueError) as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # API/network errors etc.
        print(f"\nUnexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nDone.")
    print(f"Tests: {'ALL PASSED' if result.final_passed else 'SOME FAILED'}")
    print(f"Files saved in: {result.output_dir}")
    return 0 if result.final_passed else 3


if __name__ == "__main__":
    sys.exit(main())