"""Tiny web server (standard library only) for the AI development team UI.

Run:  python server.py            then open http://127.0.0.1:8000

API
  GET  /api/status       -> {"api_key_set": bool, "provider": str | null}
  POST /api/run          -> 202 {"job_id": "..."}   (starts a background run)
  GET  /api/job/<id>     -> {"status": "running|done|error", "logs": [...], "result": {...}, "error": str|null}
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

from llm import LLMClient, MockLLM, configured_provider, create_llm, has_api_key
from orchestrator import DevTeam

BASE_DIR = Path(__file__).resolve().parent


def find_web_dir() -> Path:
    """Folder that holds index.html: ./web, ./public, or the project folder itself."""
    for folder in (BASE_DIR / "web", BASE_DIR / "public", BASE_DIR):
        if (folder / "index.html").is_file():
            return folder
    return BASE_DIR / "web"


WEB_DIR = find_web_dir()
OUTPUT_DIR = BASE_DIR / "output"
MAX_BODY_BYTES = 100_000
MAX_REQUIREMENT_CHARS = 5_000
MAX_JOBS = 25
MOCK_DELAY_SECONDS = 0.9  # makes the offline demo feel like real agents working

# Only these files are ever served (prevents path traversal).
STATIC_FILES: Dict[str, Tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

RUN_LOCK = threading.Lock()  # one pipeline run at a time
JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing variables win)."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key, value)


def make_mock() -> LLMClient:
    """Offline demo backend. Older llm.py versions have no `delay` option, so fall back."""
    try:
        return MockLLM(delay=MOCK_DELAY_SECONDS)
    except TypeError:
        return MockLLM()


def new_job() -> Tuple[str, Dict[str, Any]]:
    job_id = uuid.uuid4().hex[:16]
    job: Dict[str, Any] = {"status": "running", "logs": [], "result": None, "error": None}
    with JOBS_LOCK:
        JOBS[job_id] = job
        while len(JOBS) > MAX_JOBS:
            oldest = next(iter(JOBS))
            if JOBS[oldest]["status"] == "running":
                break
            del JOBS[oldest]
    return job_id, job


def run_job(job: Dict[str, Any], llm: LLMClient, requirement: str) -> None:
    """Runs in a background thread; the web page polls the job for progress."""
    try:
        result = DevTeam(llm, log=job["logs"].append).run(requirement, output_dir=OUTPUT_DIR)
        job["result"] = {
            "plan": result.plan,
            "code": result.code,
            "tests": result.tests,
            "review": result.review,
            "test_history": result.test_history,
            "passed": result.final_passed,
            "report": result.report,
            "saved_to": str(result.output_dir),
        }
        job["status"] = "done"
    except (RuntimeError, ValueError) as exc:
        job["error"] = f"Pipeline failed: {exc}"
        job["status"] = "error"
    except Exception as exc:  # API / network errors
        job["error"] = f"Unexpected error: {type(exc).__name__}: {exc}"
        job["status"] = "error"
    finally:
        RUN_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "AIDevTeam/2.1"

    # ---- helpers -------------------------------------------------------- #
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> Dict[str, Any]:
        """Return the parsed JSON body, or raise ValueError with a user-facing message."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Invalid request size.")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Request body must be a JSON object.") from None
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    # ---- routes --------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/api/status":
            self._json(200, {"api_key_set": has_api_key(), "provider": configured_provider()})
            return

        if path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            with JOBS_LOCK:
                job = JOBS.get(job_id) if job_id.isalnum() and len(job_id) <= 32 else None
            if job is None:
                self._json(404, {"error": "Unknown job."})
                return
            self._json(
                200,
                {
                    "status": job["status"],
                    "logs": list(job["logs"]),
                    "result": job["result"],
                    "error": job["error"],
                },
            )
            return

        entry = STATIC_FILES.get(path)
        if entry is None:
            self._json(404, {"error": "Not found"})
            return
        try:
            self._send(200, (WEB_DIR / entry[0]).read_bytes(), entry[1])
        except OSError:
            self._json(
                404,
                {"error": f"Missing file: {entry[0]}. Put index.html, style.css and app.js next to server.py or in a 'web' folder."},
            )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/run":
            self._json(404, {"error": "Not found"})
            return

        try:
            data = self._read_json()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return

        requirement = str(data.get("requirement", "")).strip()
        if not requirement:
            self._json(400, {"error": "Please enter a software requirement."})
            return
        if len(requirement) > MAX_REQUIREMENT_CHARS:
            self._json(400, {"error": f"Requirement is too long (max {MAX_REQUIREMENT_CHARS} characters)."})
            return

        try:
            llm: LLMClient = make_mock() if data.get("mock") else create_llm()
        except RuntimeError as exc:
            self._json(400, {"error": str(exc)})
            return

        if not RUN_LOCK.acquire(blocking=False):
            self._json(409, {"error": "Another run is in progress. Please wait and try again."})
            return

        job_id, job = new_job()
        try:
            threading.Thread(target=run_job, args=(job, llm, requirement), daemon=True).start()
        except RuntimeError as exc:
            RUN_LOCK.release()
            job["error"] = f"Could not start the run: {exc}"
            job["status"] = "error"
            self._json(500, {"error": job["error"]})
            return
        self._json(202, {"job_id": job_id})


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for the AI development team")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    args = parser.parse_args()
    load_env(BASE_DIR / ".env")

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"Cannot start server on {args.host}:{args.port} - {exc}")
    print(f"AI Dev Team UI running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()