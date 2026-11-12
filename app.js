(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const POLL_MS = 700;
  const MAX_POLL_FAILURES = 5;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const WORKING = ["Analysing requirement", "Writing code", "Running tests", "Reviewing code", "Building report"];
  const DONE = ["Plan ready", "Code written", "Tests ready", "Review done", "Report ready"];

  const state = {
    code: "", tests: "", report: "",
    startedAt: 0, seenLogs: 0, failures: 0,
    pollTimer: null, clockTimer: null,
  };

  // ------------------------------------------------------------------
  // Small DOM helpers. All dynamic text goes in through textContent,
  // so AI-generated content can never inject HTML.
  // ------------------------------------------------------------------
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function formatClock(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    const m = String(Math.floor(total / 60)).padStart(2, "0");
    const s = String(total % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------
  $("themeToggle").addEventListener("click", () => {
    const next = (document.documentElement.dataset.theme || "dark") === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage may be blocked */ }
  });

  // ------------------------------------------------------------------
  // Python syntax highlighter (tiny, tokenised, DOM-based)
  // ------------------------------------------------------------------
  const KEYWORDS = new Set([
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
    "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
  ]);
  const BUILTINS = new Set([
    "self", "cls", "int", "float", "str", "bool", "list", "dict", "set", "tuple", "len", "range", "print",
    "isinstance", "super", "object", "Exception", "ValueError", "TypeError", "KeyError", "OverflowError",
    "ZeroDivisionError", "Optional", "Union", "List", "Dict", "Any", "annotations",
  ]);
  const TOKEN_RE = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(@[A-Za-z_][\w.]*)|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)\b/g;

  function highlight(code) {
    const frag = document.createDocumentFragment();
    let last = 0;
    let prevWord = "";
    let m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(code)) !== null) {
      if (m.index > last) frag.appendChild(document.createTextNode(code.slice(last, m.index)));
      let cls = "";
      if (m[1]) { cls = "tok-cm"; prevWord = ""; }
      else if (m[2]) { cls = "tok-st"; prevWord = ""; }
      else if (m[3]) { cls = "tok-dc"; prevWord = ""; }
      else if (m[4]) { cls = "tok-nu"; prevWord = ""; }
      else {
        const word = m[5];
        if (KEYWORDS.has(word)) cls = "tok-kw";
        else if (prevWord === "def" || prevWord === "class") cls = "tok-fn";
        else if (BUILTINS.has(word)) cls = "tok-bi";
        prevWord = word;
      }
      frag.appendChild(cls ? el("span", m[0], cls) : document.createTextNode(m[0]));
      last = m.index + m[0].length;
    }
    if (last < code.length) frag.appendChild(document.createTextNode(code.slice(last)));
    return frag;
  }

  function fillCode(gutterId, viewId, text) {
    const clean = text.replace(/\s+$/, "");
    const count = clean === "" ? 1 : clean.split("\n").length;
    $(gutterId).textContent = Array.from({ length: count }, (_, i) => i + 1).join("\n");
    $(viewId).replaceChildren(highlight(clean));
  }

  // ------------------------------------------------------------------
  // Provider pill + startup status
  // ------------------------------------------------------------------
  function setPill(stateName, label) {
    const pill = $("providerPill");
    pill.dataset.state = stateName;
    pill.querySelector("span").textContent = label;
  }

  fetch("/api/status")
    .then((r) => r.json())
    .then((s) => {
      const names = { groq: "Groq", anthropic: "Claude" };
      if (s.api_key_set) {
        setPill("ok", `${names[s.provider] || "LLM"} connected`);
      } else {
        setPill("warn", "No API key");
        $("keyHint").hidden = false;
        $("mock").checked = true;
      }
    })
    .catch(() => setPill("warn", "Server offline"));

  // ------------------------------------------------------------------
  // Input niceties
  // ------------------------------------------------------------------
  const box = $("requirement");
  const updateCount = () => { $("charCount").textContent = `${box.value.length} / ${box.maxLength}`; };
  box.addEventListener("input", updateCount);
  updateCount();

  document.querySelectorAll("[data-example]").forEach((chip) => {
    chip.addEventListener("click", () => {
      box.value = chip.dataset.example;
      updateCount();
      box.focus();
    });
  });

  box.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") startRun();
  });
  $("run").addEventListener("click", startRun);

  // ------------------------------------------------------------------
  // Pipeline (agent nodes)
  // ------------------------------------------------------------------
  const nodes = Array.from(document.querySelectorAll(".node"));

  function nodeFor(stage) { return nodes[stage - 1]; }

  function paintNode(stage, kind, text) {
    const node = nodeFor(stage);
    if (!node) return;
    node.classList.remove("idle", "active", "done", "error");
    node.classList.add(kind);
    if (text !== undefined) node.querySelector(".node-status").textContent = text;
  }

  function resetPipeline() {
    nodes.forEach((_, i) => paintNode(i + 1, "idle", "Waiting"));
  }

  function setStage(stage) {
    for (let s = 1; s <= nodes.length; s += 1) {
      if (s < stage) paintNode(s, "done", DONE[s - 1]);
      else if (s === stage) paintNode(s, "active", WORKING[s - 1]);
      else paintNode(s, "idle", "Waiting");
    }
  }

  function finishPipeline() {
    for (let s = 1; s <= nodes.length; s += 1) paintNode(s, "done", DONE[s - 1]);
  }

  function failPipeline() {
    const active = nodes.findIndex((n) => n.classList.contains("active"));
    const index = active === -1 ? 0 : active;
    paintNode(index + 1, "error", "Failed");
  }

  // ------------------------------------------------------------------
  // Terminal
  // ------------------------------------------------------------------
  const cursor = el("span", "", "cursor");

  function logClass(line) {
    if (/^\[\d\/5\]/.test(line)) return "stage";
    if (/PASSED/.test(line)) return "ok";
    if (/FAILED|failed/.test(line)) return "bad";
    return "sys";
  }

  function appendLog(line) {
    const clean = line.trim();
    const row = el("div", null, `line ${logClass(clean)}`);
    row.appendChild(el("span", formatClock(Date.now() - state.startedAt), "ts"));
    row.appendChild(el("span", clean, "msg"));
    $("termBody").appendChild(row);
    $("termBody").appendChild(cursor);
    $("termBody").scrollTop = $("termBody").scrollHeight;
    updatePipelineFromLog(clean);
  }

  function updatePipelineFromLog(line) {
    const stage = line.match(/^\[(\d)\/5\]/);
    if (stage) { setStage(Number(stage[1])); return; }
    const fix = line.match(/developer fix round (\d+)/i);
    if (fix) {
      paintNode(3, "done", "Tests failed");
      paintNode(2, "active", `Fixing bugs, round ${fix[1]}`);
      return;
    }
    if (/Tests .*: PASSED/.test(line)) nodeFor(3).querySelector(".node-status").textContent = "Tests passed";
    else if (/Tests .*: FAILED/.test(line)) nodeFor(3).querySelector(".node-status").textContent = "Tests failed";
  }

  function consumeLogs(logs) {
    for (let i = state.seenLogs; i < logs.length; i += 1) appendLog(logs[i]);
    state.seenLogs = logs.length;
  }

  // ------------------------------------------------------------------
  // Run lifecycle
  // ------------------------------------------------------------------
  function showError(message) {
    const node = $("error");
    node.textContent = message;
    node.hidden = !message;
  }

  function setBusy(busy) {
    const btn = $("run");
    btn.disabled = busy;
    btn.classList.toggle("is-running", busy);
    btn.querySelector(".btn-label").textContent = busy ? "Agents running" : "Run team";
  }

  function stopTimers() {
    clearTimeout(state.pollTimer);
    clearInterval(state.clockTimer);
    state.pollTimer = null;
    state.clockTimer = null;
    cursor.remove();
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (err) {
      throw new Error(`Server returned an invalid response (HTTP ${response.status}).`);
    }
  }

  async function startRun() {
    if ($("run").disabled) return;
    const requirement = box.value.trim();
    showError("");
    if (!requirement) {
      showError("Please enter a software requirement.");
      box.focus();
      return;
    }

    setBusy(true);
    $("results").hidden = true;
    $("termBody").replaceChildren();
    $("terminal").hidden = false;
    resetPipeline();
    state.seenLogs = 0;
    state.failures = 0;
    state.startedAt = Date.now();
    $("elapsed").textContent = "00:00";
    state.clockTimer = setInterval(() => { $("elapsed").textContent = formatClock(Date.now() - state.startedAt); }, 250);
    appendLog("Submitting requirement to the agent team...");
    document.getElementById("pipeline").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requirement, mock: $("mock").checked }),
      });
      const data = await readJson(response);
      if (!response.ok) throw new Error(data.error || `Request failed (HTTP ${response.status}).`);
      poll(data.job_id);
    } catch (err) {
      fail(err.message || "Something went wrong.");
    }
  }

  async function poll(jobId) {
    let job;
    try {
      const response = await fetch(`/api/job/${encodeURIComponent(jobId)}`);
      job = await readJson(response);
      if (!response.ok) { fail(job.error || `Request failed (HTTP ${response.status}).`); return; }
    } catch (err) {
      state.failures += 1;
      if (state.failures >= MAX_POLL_FAILURES) { fail("Lost connection to the server."); return; }
      state.pollTimer = setTimeout(() => poll(jobId), POLL_MS * 2);
      return;
    }

    state.failures = 0;
    consumeLogs(job.logs || []);

    if (job.status === "done") { finish(job.result); return; }
    if (job.status === "error") { fail(job.error || "The run failed."); return; }
    state.pollTimer = setTimeout(() => poll(jobId), POLL_MS);
  }

  function fail(message) {
    stopTimers();
    failPipeline();
    setBusy(false);
    showError(message);
  }

  function finish(result) {
    const seconds = Math.max(1, Math.round((Date.now() - state.startedAt) / 1000));
    stopTimers();
    finishPipeline();
    setBusy(false);
    appendLog(`Done in ${seconds}s. Files saved to ${result.saved_to}`);
    cursor.remove();
    showResults(result, seconds);
  }

  // ------------------------------------------------------------------
  // Results rendering
  // ------------------------------------------------------------------
  function countUp(node, target, suffix) {
    const tail = suffix || "";
    if (reduceMotion || !Number.isFinite(target)) { node.textContent = `${target}${tail}`; return; }
    const duration = 800;
    const t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / duration);
      node.textContent = `${Math.round(target * (1 - Math.pow(1 - p, 3)))}${tail}`;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderMetrics(data, seconds) {
    const wrap = $("metrics");
    wrap.replaceChildren();
    const testMethods = (data.tests.match(/^\s*def test_/gm) || []).length;
    const lines = data.code.replace(/\s+$/, "").split("\n").length;
    const issues = (data.review.issues || []).length;
    const cards = [
      { label: "Tests", text: data.passed ? "PASS" : "FAIL", tone: data.passed ? "ok" : "bad", sub: `${testMethods} test methods` },
      { label: "Test runs", number: data.test_history.length, sub: data.test_history.length > 1 ? "with auto-fix rounds" : "first run" },
      { label: "Review issues", number: issues, sub: `verdict: ${data.review.verdict}` },
      { label: "Lines of code", number: lines, sub: "solution.py" },
      { label: "Duration", number: seconds, suffix: "s", sub: "end to end" },
    ];
    cards.forEach((c) => {
      const card = el("div", null, "metric");
      card.appendChild(el("div", c.label, "metric-label"));
      const value = el("div", "", `metric-value${c.tone ? " " + c.tone : ""}`);
      if (c.text !== undefined) value.textContent = c.text;
      else countUp(value, c.number, c.suffix);
      card.appendChild(value);
      card.appendChild(el("div", c.sub, "metric-sub"));
      wrap.appendChild(card);
    });
  }

  function section(panel, title) { panel.appendChild(el("h3", title, "sec-title")); }

  function renderPlan(plan) {
    const panel = $("panel-plan");
    panel.replaceChildren();
    section(panel, "Summary");
    panel.appendChild(el("p", plan.summary, "lead"));

    if (plan.functions && plan.functions.length) {
      section(panel, "Planned functions");
      const grid = el("div", null, "fn-grid");
      plan.functions.forEach((f) => {
        const card = el("div", null, "fn-card");
        card.appendChild(el("code", f.signature || f.name));
        if (f.description) card.appendChild(el("p", f.description));
        grid.appendChild(card);
      });
      panel.appendChild(grid);
    }
    if (plan.edge_cases && plan.edge_cases.length) {
      section(panel, "Edge cases");
      const tags = el("div", null, "tags");
      plan.edge_cases.forEach((e) => tags.appendChild(el("span", e, "tag")));
      panel.appendChild(tags);
    }
    if (plan.assumptions && plan.assumptions.length) {
      section(panel, "Assumptions");
      const ul = el("ul", null, "bullets");
      plan.assumptions.forEach((a) => ul.appendChild(el("li", a)));
      panel.appendChild(ul);
    }
    if (plan.steps && plan.steps.length) {
      section(panel, "Implementation steps");
      const ol = el("ol", null, "timeline");
      plan.steps.forEach((s) => ol.appendChild(el("li", s)));
      panel.appendChild(ol);
    }
  }

  function renderHistory(history) {
    const wrap = $("history");
    wrap.replaceChildren();
    history.forEach((run) => {
      const details = el("details", null, "run");
      const summary = el("summary", run.label);
      summary.appendChild(el("span", run.passed ? "PASSED" : "FAILED", `res ${run.passed ? "ok" : "bad"}`));
      details.appendChild(summary);
      details.appendChild(el("pre", run.output));
      wrap.appendChild(details);
    });
  }

  function renderReview(review) {
    const panel = $("panel-review");
    panel.replaceChildren();
    const approved = String(review.verdict).toLowerCase() === "approve";
    const verdict = el("div", null, "verdict");
    verdict.appendChild(el("span", approved ? "Approved" : "Needs changes", `badge ${approved ? "approve" : "changes"}`));
    verdict.appendChild(el("p", review.summary));
    panel.appendChild(verdict);

    const issues = review.issues || [];
    if (issues.length === 0) {
      panel.appendChild(el("p", "No issues reported.", "empty"));
      return;
    }
    const list = el("div", null, "issues");
    issues.forEach((issue) => {
      const card = el("div", null, `issue issue-${issue.severity}`);
      const top = el("div", null, "issue-top");
      top.appendChild(el("span", issue.severity, `badge sev-${issue.severity}`));
      top.appendChild(el("p", issue.description));
      card.appendChild(top);
      if (issue.suggestion) {
        const fix = el("p", null, "issue-fix");
        fix.appendChild(el("b", "Fix"));
        fix.appendChild(document.createTextNode(issue.suggestion));
        card.appendChild(fix);
      }
      list.appendChild(card);
    });
    panel.appendChild(list);
  }

  function showResults(data, seconds) {
    state.code = data.code;
    state.tests = data.tests;
    state.report = data.report;

    const banner = $("statusBanner");
    banner.className = `banner ${data.passed ? "ok" : "bad"}`;
    banner.replaceChildren();
    const text = el("div");
    text.appendChild(document.createTextNode(data.passed ? "All tests passed. Your code is ready." : "Some tests failed. Manual review needed."));
    text.appendChild(el("small", data.saved_to ? `Saved to ${data.saved_to}` : ""));
    banner.appendChild(text);

    renderMetrics(data, seconds);
    renderPlan(data.plan);
    fillCode("code-gutter", "code-view", data.code);
    fillCode("tests-gutter", "tests-view", data.tests);
    renderHistory(data.test_history || []);
    renderReview(data.review);
    $("report").textContent = data.report;
    $("log").textContent = Array.from($("termBody").querySelectorAll(".line"))
      .map((row) => row.textContent).join("\n");

    $("results").hidden = false;
    selectTab("plan");
    $("results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }

  // ------------------------------------------------------------------
  // Tabs (with arrow-key navigation)
  // ------------------------------------------------------------------
  const tabs = Array.from(document.querySelectorAll(".tab"));

  function selectTab(name) {
    tabs.forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.hidden = panel.id !== `panel-${name}`;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      if (next) { event.preventDefault(); next.focus(); selectTab(next.dataset.tab); }
    });
  });

  // ------------------------------------------------------------------
  // Copy / download
  // ------------------------------------------------------------------
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const original = btn.textContent;
      try {
        await navigator.clipboard.writeText(state[btn.dataset.copy] || "");
        btn.textContent = "Copied";
      } catch (err) {
        btn.textContent = "Copy failed";
      }
      setTimeout(() => { btn.textContent = original; }, 1500);
    });
  });

  document.querySelectorAll("[data-download]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const blob = new Blob([state[btn.dataset.download] || ""], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = btn.dataset.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    });
  });
})();