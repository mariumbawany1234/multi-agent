# AI Multi-Agent Development Team

A small team of AI agents that turns a software requirement into tested Python code and a report.

| Agent | Role | File |
|---|---|---|
| Project Planner | Software Architect | `agents.py` -> `ProjectPlanner` |
| Python Developer | Backend Developer | `agents.py` -> `PythonDeveloper` |
| QA Engineer | Software Tester | `agents.py` -> `QAEngineer` |
| Code Reviewer | Senior Developer | `agents.py` -> `CodeReviewer` |

## Workflow (`orchestrator.py`)
1. Planner analyses the requirement and returns a JSON plan.
2. Developer writes `solution.py` (syntax is checked, auto-repaired if needed).
3. QA writes `test_solution.py` (unittest) and the tests are actually executed.
4. If tests fail, the Developer gets the failure output and fixes the code (up to `--max-fix-rounds`).
5. Reviewer reviews code + test results. Medium/high/critical findings are sent back to the Developer; the fix is kept only if tests still pass.
6. Final code and `report.md` are saved in `output/run_<timestamp>/`.

## Setup
    python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
    pip install -r requirements.txt                     # Groq mode needs no packages

Open the `.env` file and paste your key after `GROQ_API_KEY=` (get one at https://console.groq.com/keys).
Never share or commit `.env` (it is listed in `.gitignore`).
To use Claude instead: set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...` in `.env`, then `pip install anthropic`.

## Run
    python main.py --mock                                # offline demo, no key needed
    python main.py "Create a function that validates email addresses"
    python main.py --file requirement.txt --output results

Optional: `--provider groq|anthropic`, `--model <name>`, or `GROQ_MODEL` in `.env` (default `llama-3.3-70b-versatile`).

    python server.py        # web UI at http://127.0.0.1:8000

## Output files
`solution.py`, `test_solution.py`, `plan.json`, `review.json`, `report.md`

## Safety note
The generated tests execute AI-written code in a subprocess (with a timeout, but not a hardened sandbox).
Use a container or VM for untrusted requirements.