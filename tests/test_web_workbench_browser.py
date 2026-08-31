from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "autoidea" / "web" / "static"
MAC_CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def _stage(stage_id: str, number: str, name: str, artifacts: list[str], *, checkpoint: bool = False) -> dict:
    return {
        "id": stage_id,
        "number": number,
        "name": name,
        "checkpoint": checkpoint,
        "checkpoint_recorded": checkpoint,
        "status": "complete",
        "required_artifacts": artifacts,
        "missing_artifacts": [],
        "invalid_artifacts": [],
    }


PIPELINE_STAGES = [
    _stage("stage_1", "01", "Requirement intake", ["research_brief.md"]),
    _stage("stage_2", "02", "Task formalization", ["task_formalization.md"]),
    _stage("stage_3", "03", "Literature survey", ["literature_survey.md", "paper_registry.json"]),
    _stage("stage_3.5", "03.5", "Paper deep reading", ["paper_deep_reading.md"]),
    _stage("stage_4", "04", "Position analysis", ["paper_positions.json"]),
    _stage("stage_5", "05", "Hook-driven expansion", ["expanded_literature.md"]),
    _stage("stage_6", "06", "Evidence binding", ["evidence_db.json"]),
    _stage("stage_7", "07", "Knowledge synthesis", ["knowledge_synthesis.md", "research_gaps.json"], checkpoint=True),
    _stage("stage_8", "08", "Design space", ["design_space.json"]),
    _stage("stage_9", "09", "Idea generation", ["raw_ideas.json"], checkpoint=True),
    _stage("stage_9.5", "09.5", "Elo tournament", ["tournament_rankings.json"]),
    _stage("stage_10", "10", "Adversarial debate", ["debate_log.md", "idea_reviews.json"], checkpoint=True),
    _stage("stage_11", "11", "Feasibility assessment", ["feasibility_assessments.json"]),
    _stage("stage_12", "12", "Final report", ["final_report.md"]),
]


ARTIFACTS = [
    {"path": name, "kind": name.rsplit(".", 1)[-1], "size_bytes": 640, "title": name.replace("_", " ").rsplit(".", 1)[0].title()}
    for stage in PIPELINE_STAGES
    for name in stage["required_artifacts"]
]
ARTIFACTS.append({"path": "unsafe_notes.md", "kind": "md", "size_bytes": 90, "title": "Unsafe Notes"})


FULL_SNAPSHOT = {
    "workspace": {"name": "verified-run", "path": "/tmp/autoidea-browser-test/runs/verified-run"},
    "counts": {"papers": 2, "claims": 2, "gaps": 1, "ideas": 1, "artifacts": len(ARTIFACTS), "warnings": 0},
    "warnings": [],
    "artifacts": ARTIFACTS,
    "pipeline": {
        "source": "observed",
        "active_stage": "",
        "active_detail": "",
        "last_completed_stage": "stage_12",
        "next_stage": "complete",
        "completed_count": 14,
        "total_stages": 14,
        "percent": 100,
        "persisted_state_stale": False,
        "stages": PIPELINE_STAGES,
        "completion": {
            "verified": True,
            "required_artifacts_ready": True,
            "final_report_present": True,
            "final_report_path": "final_report.md",
            "checkpoint_events": ["stage_7", "stage_9", "stage_10"],
            "missing_checkpoints": [],
            "audit_passed": True,
            "audit_issues": [],
        },
    },
    "papers": [
        {
            "paper_id": "P1",
            "title": "VideoAgent: Long-form Video Understanding",
            "year": 2025,
            "source": "arxiv",
            "venue": "CVPR",
            "url": "https://example.org/p1",
            "authors": ["Ada Researcher", "Bo Engineer"],
            "relevance": "Agentic planning baseline.",
            "position": {"initial_attack": "Limited adversarial testing.", "weakest_link": "Sparse provenance.", "summary": "Strong planning baseline.", "dimensions": []},
        },
        {
            "paper_id": "P2",
            "title": "Evidence-grounded Research Synthesis",
            "year": 2024,
            "source": "semantic_scholar",
            "venue": "ACL",
            "url": "https://example.org/p2",
            "authors": ["Chen Reviewer"],
            "relevance": "Traceability baseline.",
            "position": None,
        },
    ],
    "claims": [
        {"citation_id": "C1", "claim": "Planning improves multi-hop retrieval.", "source_paper_id": "P1", "source_title": "VideoAgent", "source_url": "https://example.org/p1", "confidence": "HIGH", "evidence_type": "experiment", "section": "Results", "tags": ["planning"]},
        {"citation_id": "C2", "claim": "Evidence tables improve auditability.", "source_paper_id": "P2", "source_title": "Evidence synthesis", "source_url": "https://example.org/p2", "confidence": "MEDIUM", "evidence_type": "system", "section": "Interface", "tags": ["audit"]},
    ],
    "gaps": [
        {
            "gap_id": "G1",
            "title": "Auditable multi-hop evidence selection",
            "description": "Current planners do not expose a complete evidence chain.",
            "gap_type": "methodology_gap",
            "demand": 5,
            "coverage": 2,
            "gap_score": 3,
            "evidence_links": [
                {"citation_id": "C1", "relationship": "supports", "rationale": "C1 establishes the planning mechanism and its unresolved provenance boundary."},
                {"citation_id": "C2", "relationship": "partial_coverage", "rationale": "C2 provides partial auditability without closing the planner integration gap."},
            ],
            "why_it_matters": "Answers need inspectable provenance.",
            "potential_direction": "Integrate provenance scoring into retrieval planning.",
        }
    ],
    "ideas": [
        {"idea_id": "I1", "title": "Citation-aware retrieval planner", "one_liner": "Optimize answer quality and provenance together.", "description": "A planner grounded in diverse evidence chains.", "target_gaps": ["G1"], "supporting_evidence": ["C1", "C2"], "composite_score": 4.6, "self_assessment": {"novelty": 4, "impact": 5}},
    ],
    "design_axes": [
        {"name": "Evidence selection", "description": "How support is selected.", "values": ["single-hop", "multi-hop", "citation-aware"], "explored": ["single-hop", "multi-hop"], "unexplored": ["citation-aware"]},
    ],
    "graph": {
        "nodes": [
            {"id": "paper:P1", "label": "VideoAgent", "kind": "paper", "group": "arxiv"},
            {"id": "paper:P2", "label": "Evidence synthesis", "kind": "paper", "group": "acl"},
            {"id": "claim:C1", "label": "C1", "kind": "claim", "group": "HIGH"},
            {"id": "claim:C2", "label": "C2", "kind": "claim", "group": "MEDIUM"},
            {"id": "gap:G1", "label": "Auditable multi-hop evidence selection", "kind": "gap", "group": "methodology_gap"},
            {"id": "idea:I1", "label": "Citation-aware retrieval planner", "kind": "idea", "group": "idea"},
        ],
        "edges": [
            {"source": "paper:P1", "target": "claim:C1", "kind": "supports"},
            {"source": "paper:P2", "target": "claim:C2", "kind": "supports"},
            {"source": "claim:C1", "target": "gap:G1", "kind": "supports_gap", "detail": "C1 establishes the planning mechanism and its unresolved provenance boundary."},
            {"source": "claim:C2", "target": "gap:G1", "kind": "partially_covers_gap", "detail": "C2 provides partial auditability without closing the planner integration gap."},
            {"source": "claim:C1", "target": "idea:I1", "kind": "evidence_for"},
            {"source": "claim:C2", "target": "idea:I1", "kind": "evidence_for"},
            {"source": "gap:G1", "target": "idea:I1", "kind": "targets"},
        ],
    },
}


def _completion() -> dict:
    return dict(FULL_SNAPSHOT["pipeline"]["completion"])


RUNS_PAYLOAD: list[dict] = []
RUN_SNAPSHOTS: dict[str, dict] = {}
RUN_INPUTS: list[dict] = []
RUN_ACTIONS: list[dict] = []
CONFIG_VALUES: dict[str, object] = {}
CONFIG_ENV_OVERRIDES: dict[str, tuple[object, str]] = {}
REQUEST_COUNTS: dict[str, int] = {}


def _completed_run() -> dict:
    return {
        "run_id": "verified-run",
        "status": "pipeline_completed",
        "status_detail": "Stage 12 and artifact audit are complete.",
        "prompt": "Find a citation-aware retrieval direction.",
        "workspace": "/tmp/autoidea-browser-test/runs/verified-run",
        "run_name": "verified-run",
        "mode": "new",
        "thread_id": "thread-verified",
        "model": "test-model",
        "provider": "custom-openai",
        "seed_papers": "",
        "seed_ideas": "",
        "auto_approve": False,
        "show_thinking": True,
        "pid": None,
        "exit_code": 0,
        "started_at": "2026-08-26T08:00:00+00:00",
        "finished_at": "2026-08-26T08:04:00+00:00",
        "log_tail": "deterministic run complete",
        "interaction": None,
        "current_stage": "complete",
        "completed_stages": 14,
        "total_stages": 14,
        "completion": _completion(),
    }


def _checkpoint_run(kind: str = "checkpoint") -> dict:
    interaction = {
        "kind": kind,
        "interaction_id": "interaction-stage-7",
        "checkpoint_stage": "stage_7" if kind == "checkpoint" else "",
        "questions": [
            {"question": "What constraint matters most?", "type": "text", "required": True},
            {"question": "Approve this direction?", "type": "multiple_choice", "choices": [{"label": "Approve", "value": "approve"}, {"label": "Revise", "value": "revise"}], "required": True},
        ] if kind == "checkpoint" else [],
        "actions": [{"name": "write_file", "args": {"path": "knowledge_synthesis.md"}}] if kind == "tool_approval" else [],
    }
    return {
        **_completed_run(),
        "run_id": "waiting-run",
        "run_name": "waiting-run",
        "workspace": "/tmp/autoidea-browser-test/runs/waiting-run",
        "status": "waiting_for_input",
        "status_detail": "Waiting for a structured browser response.",
        "exit_code": None,
        "finished_at": "",
        "interaction": interaction,
        "current_stage": "stage_7",
        "completed_stages": 8,
        "completion": {**_completion(), "verified": False, "final_report_present": False, "audit_passed": None, "checkpoint_events": []},
    }


def _progress_run() -> dict:
    return {
        **_completed_run(),
        "run_id": "progress-run",
        "run_name": "progress-run",
        "workspace": "/tmp/autoidea-browser-test/runs/progress-run",
        "status": "running",
        "status_detail": "Research process is running.",
        "log_tail": "unbroken-runtime-output-" * 100,
        "exit_code": None,
        "finished_at": "",
        "current_stage": "stage_3.5",
        "completed_stages": 3,
        "progress": {
            "stage": "stage_3.5",
            "number": "03.5",
            "name": "Paper deep reading",
            "index": 4,
            "total_stages": 14,
            "phase": "retrieving_full_text",
            "activity": "fetch_paper_fulltext",
            "subject": "2505.10483v1",
            "activity_state": "running",
            "updated_at": "2026-08-30T01:05:50+08:00",
            "current": 11,
            "total": 20,
            "unit": "papers_processed",
            "percent": 55,
            "indeterminate": False,
            "counts": {
                "full_text": 9,
                "failed": 2,
                "batches_completed": 1,
                "batches_total": 4,
            },
        },
        "completion": {
            **_completion(),
            "verified": False,
            "final_report_present": False,
            "audit_passed": None,
            "checkpoint_events": [],
        },
    }


def _config_payload() -> dict:
    def field(key: str, kind: str, label: str, *, secret: bool = False, options: list[str] | None = None) -> dict:
        value = CONFIG_VALUES.get(key, "")
        effective_value, env_var = CONFIG_ENV_OVERRIDES.get(key, (value, ""))
        return {
            "key": key, "label": label, "type": kind,
            "value": "" if secret else value,
            "effective_value": "" if secret else effective_value,
            "default": "", "secret": secret, "is_set": bool(effective_value),
            "stored_is_set": bool(value),
            "masked_value": "test...alue" if secret and value else "",
            "env_var": env_var, "env_overridden": bool(env_var), "options": options or [],
        }

    return {
        "path": "/tmp/autoidea-browser-test/config.yaml",
        "groups": [
            {"id": "quick", "title": "Quick Setup", "title_zh": "快速设置", "fields": ["provider", "model", "max_tokens", "show_thinking", "auto_approve"]},
            {"id": "credentials", "title": "Credentials", "title_zh": "密钥与服务", "fields": ["openai_api_key"]},
        ],
        "fields": {
            "provider": field("provider", "select", "Provider", options=["anthropic", "openai", "custom-openai"]),
            "model": field("model", "str", "Model"),
            "max_tokens": field("max_tokens", "int", "Max tokens"),
            "show_thinking": field("show_thinking", "bool", "Show thinking"),
            "auto_approve": field("auto_approve", "bool", "Auto approve"),
            "openai_api_key": field("openai_api_key", "str", "OpenAI API key", secret=True),
        },
    }


ARTIFACT_PAYLOADS = {
    "final_report.md": {"path": "final_report.md", "kind": "md", "size_bytes": 640, "title": "Final Report", "text": "# Verified final report", "html": "<h1>Verified final report</h1><p>Evidence-grounded conclusion.</p>"},
    "unsafe_notes.md": {"path": "unsafe_notes.md", "kind": "md", "size_bytes": 90, "title": "Unsafe Notes", "text": "unsafe", "html": '<h1>Unsafe</h1><script>window.__xss = true</script><a href="javascript:window.__xss=true" onclick="window.__xss=true">bad</a>'},
}


def _reset_state() -> None:
    RUNS_PAYLOAD.clear()
    RUN_SNAPSHOTS.clear()
    RUN_INPUTS.clear()
    RUN_ACTIONS.clear()
    REQUEST_COUNTS.clear()
    CONFIG_ENV_OVERRIDES.clear()
    CONFIG_VALUES.clear()
    CONFIG_VALUES.update({"provider": "custom-openai", "model": "qwen-test", "max_tokens": 16384, "show_thinking": True, "auto_approve": True, "openai_api_key": "secret-value"})


class WorkbenchHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        REQUEST_COUNTS[path] = REQUEST_COUNTS.get(path, 0) + 1
        if path == "/":
            return self._send_bytes((STATIC_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/static/app.js":
            return self._send_bytes((STATIC_ROOT / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        if path == "/static/styles.css":
            return self._send_bytes((STATIC_ROOT / "styles.css").read_bytes(), "text/css; charset=utf-8")
        if path == "/static/autoidea-mark.svg":
            return self._send_bytes((STATIC_ROOT / "autoidea-mark.svg").read_bytes(), "image/svg+xml")
        if path == "/api/snapshot":
            return self._send_json(FULL_SNAPSHOT)
        if path == "/api/runs":
            return self._send_json(RUNS_PAYLOAD)
        if path == "/api/config":
            return self._send_json(_config_payload())
        parts = path.split("/")
        if len(parts) >= 5 and parts[1:3] == ["api", "runs"]:
            run_id = parts[3]
            if parts[4] == "snapshot":
                return self._send_json(RUN_SNAPSHOTS.get(run_id, FULL_SNAPSHOT))
            if parts[4] == "artifacts" and len(parts) >= 6:
                artifact_path = unquote("/".join(parts[5:]))
                payload = ARTIFACT_PAYLOADS.get(artifact_path)
                return self._send_json(payload) if payload else self._not_found()
        if len(parts) == 4 and parts[1:3] == ["api", "runs"]:
            run_id = parts[3]
            run = next((item for item in RUNS_PAYLOAD if item["run_id"] == run_id), None)
            return self._send_json(run) if run else self._not_found()
        return self._not_found()

    def do_PATCH(self) -> None:
        if urlparse(self.path).path != "/api/config":
            return self._not_found()
        payload = self._read_json()
        CONFIG_VALUES.update(payload.get("values", {}))
        self._send_json(_config_payload())

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        parts = path.split("/")
        if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "input":
            run_id = parts[3]
            payload = self._read_json()
            RUN_INPUTS.append({"run_id": run_id, **payload})
            for run in RUNS_PAYLOAD:
                if run["run_id"] == run_id:
                    run["interaction"] = None
                    run["status"] = "running"
            return self._send_json({"run_id": run_id, "status": "accepted"})
        if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "stop":
            run_id = parts[3]
            RUN_ACTIONS.append({"run_id": run_id, "action": "stop"})
            run = next((item for item in RUNS_PAYLOAD if item["run_id"] == run_id), _completed_run())
            return self._send_json({**run, "status": "stopped"})
        return self._not_found()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def _send_json(self, payload: object) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


CDP_RUNNER = r"""
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const endpoint = `http://127.0.0.1:${process.env.CDP_PORT}/json/list`;
async function pageTarget() {
  for (let i = 0; i < 50; i += 1) {
    try {
      const pages = await fetch(endpoint).then((response) => response.json());
      const page = pages.find((item) => item.type === "page" && item.webSocketDebuggerUrl);
      if (page) return page;
    } catch {}
    await sleep(100);
  }
  throw new Error("Chrome debugging target was not available");
}
function connect(url) {
  const socket = new WebSocket(url);
  let id = 0;
  const pending = new Map();
  const waiters = new Map();
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const promise = pending.get(message.id);
      pending.delete(message.id);
      return message.error ? promise.reject(new Error(JSON.stringify(message.error))) : promise.resolve(message.result);
    }
    const listeners = waiters.get(message.method) || [];
    waiters.delete(message.method);
    listeners.forEach((resolve) => resolve(message.params || {}));
  };
  return new Promise((resolve, reject) => {
    socket.onerror = reject;
    socket.onopen = () => resolve({
      send(method, params = {}) {
        const requestId = ++id;
        socket.send(JSON.stringify({id: requestId, method, params}));
        return new Promise((resolve, reject) => pending.set(requestId, {resolve, reject}));
      },
      wait(method) {
        return new Promise((resolve) => waiters.set(method, [...(waiters.get(method) || []), resolve]));
      },
      close() { socket.close(); },
    });
  });
}
(async () => {
  const target = await pageTarget();
  const cdp = await connect(target.webSocketDebuggerUrl);
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", {width: Number(process.env.VIEWPORT_WIDTH), height: 900, deviceScaleFactor: 1, mobile: Number(process.env.VIEWPORT_WIDTH) < 600});
  const mediaFeatures = [];
  if (process.env.REDUCED_MOTION === "1") mediaFeatures.push({name: "prefers-reduced-motion", value: "reduce"});
  if (process.env.DARK_MODE === "1") mediaFeatures.push({name: "prefers-color-scheme", value: "dark"});
  if (mediaFeatures.length) await cdp.send("Emulation.setEmulatedMedia", {features: mediaFeatures});
  await cdp.send("Page.addScriptToEvaluateOnNewDocument", {source: `window.__browserErrors=[]; addEventListener('error', e => window.__browserErrors.push(String(e.message))); addEventListener('unhandledrejection', e => window.__browserErrors.push(String(e.reason)));`});
  const loaded = cdp.wait("Page.loadEventFired");
  await cdp.send("Page.navigate", {url: process.env.APP_URL});
  await loaded;
  const expression = `(async () => {
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const waitFor = async (selector, timeout = 6000) => {
      const started = performance.now();
      while (performance.now() - started < timeout) {
        const element = document.querySelector(selector);
        if (element) return element;
        await sleep(50);
      }
      throw new Error('missing ' + selector);
    };
    ${process.env.TEST_BODY}
  })()`;
  const result = await cdp.send("Runtime.evaluate", {expression, awaitPromise: true, returnByValue: true});
  if (process.env.SCREENSHOT_PATH) {
    const fs = require("fs");
    const shot = await cdp.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
    fs.writeFileSync(process.env.SCREENSHOT_PATH, Buffer.from(shot.data, "base64"));
  }
  cdp.close();
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  console.log(JSON.stringify(result.result.value));
})().catch((error) => { console.error(error.stack || error.message); process.exit(1); });
"""


def _chrome_path() -> Path | None:
    configured = os.getenv("AUTOIDEA_CHROME_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)
    return MAC_CHROME if MAC_CHROME.is_file() else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_browser(
    body: str,
    *,
    width: int = 1440,
    reduced_motion: bool = False,
    dark_mode: bool = False,
    screenshot: Path | None = None,
) -> dict:
    chrome = _chrome_path()
    node = shutil.which("node")
    if not chrome or not node:
        pytest.skip("Chrome and Node.js are required for browser workbench tests")
    server = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cdp_port = _free_port()
    with tempfile.TemporaryDirectory() as profile:
        process = subprocess.Popen(
            [str(chrome), "--headless=new", "--disable-gpu", "--no-first-run", "--remote-debugging-address=127.0.0.1", f"--remote-debugging-port={cdp_port}", f"--user-data-dir={profile}", "about:blank"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            result = subprocess.run(
                [node, "-e", textwrap.dedent(CDP_RUNNER)], check=False, capture_output=True, text=True, timeout=20,
                env={**os.environ, "APP_URL": f"http://127.0.0.1:{server.server_port}/", "CDP_PORT": str(cdp_port), "VIEWPORT_WIDTH": str(width), "REDUCED_MOTION": "1" if reduced_motion else "0", "DARK_MODE": "1" if dark_mode else "0", "TEST_BODY": textwrap.dedent(body), "SCREENSHOT_PATH": str(screenshot) if screenshot else ""},
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            server.shutdown()
            server.server_close()
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("width", [320, 375, 768, 1024, 1440])
def test_observatory_views_are_runtime_safe_and_responsive(width: int) -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        await waitFor('#runComposer');
        const studioChromeHidden = getComputedStyle(document.querySelector('#pageHeader')).display === 'none'
          && getComputedStyle(document.querySelector('#contextTools')).display === 'none'
          && getComputedStyle(document.querySelector('#navScrim')).display === 'none';
        const views = ['live', 'results', 'map', 'papers', 'evidence', 'ideas', 'pipeline', 'artifacts', 'settings'];
        const rendered = {};
        for (const view of views) {
          document.querySelector(`[data-view="${view}"]`).click();
          await sleep(view === 'map' ? 180 : 60);
          rendered[view] = Boolean(document.querySelector('#viewRoot h2, #viewRoot table, #viewRoot .artifact-list, #viewRoot .config-layout'));
        }
        document.querySelector('[data-view="results"]').click();
        await waitFor('#inlineReport h2');
        return {
          rendered,
          report: document.querySelector('#inlineReport h2').textContent,
          overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          errors: window.__browserErrors,
          navToggleVisible: getComputedStyle(document.querySelector('#navToggle')).display !== 'none',
          viewport: document.documentElement.clientWidth,
          studioChromeHidden,
        };
        """,
        width=width,
        reduced_motion=True,
    )
    assert all(result["rendered"].values())
    assert result["report"] == "Verified final report"
    assert result["overflow"] is False
    assert result["errors"] == []
    assert result["viewport"] == width
    assert result["studioChromeHidden"] is True
    assert result["navToggleVisible"] is (width <= 980)


def test_research_trace_labels_are_legible_in_english_and_chinese() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        await waitFor('.trace-label');
        if (document.documentElement.dataset.language !== 'en') {
          document.querySelector('#languageToggle').click();
          await sleep(100);
        }
        const readTrace = () => {
          const labels = [...document.querySelectorAll('.trace-label')];
          const rows = [labels.slice(0, 7), labels.slice(7)];
          const gaps = rows.flatMap((row) => {
            const rects = row.map((label) => label.getBoundingClientRect()).sort((a, b) => a.left - b.left);
            return rects.slice(1).map((rect, index) => rect.left - rects[index].right);
          });
          return {
            labels: labels.map((label) => label.textContent),
            fontSize: Number.parseFloat(getComputedStyle(labels[0]).fontSize),
            fontWeight: Number.parseInt(getComputedStyle(labels[0]).fontWeight, 10),
            minGap: Math.min(...gaps),
            metaFontSize: Number.parseFloat(getComputedStyle(document.querySelector('.trace-meta')).fontSize),
            legendFontSize: Number.parseFloat(getComputedStyle(document.querySelector('.trace-legend')).fontSize),
          };
        };
        const english = readTrace();
        document.querySelector('#languageToggle').click();
        await sleep(100);
        const chinese = readTrace();
        return {english, chinese, errors: window.__browserErrors};
        """,
        width=1180,
        reduced_motion=True,
    )
    assert result["english"]["labels"] == [
        "Intake", "Formalize", "Survey", "Deep read", "Position", "Expand", "Evidence",
        "Synthesis", "Design", "Ideas", "Elo", "Debate", "Feasibility", "Report",
    ]
    assert result["chinese"]["labels"] == [
        "需求", "定义", "综述", "精读", "定位", "扩展", "证据",
        "综合", "空间", "想法", "排名", "辩论", "可行性", "报告",
    ]
    for language in ("english", "chinese"):
        assert result[language]["fontSize"] >= 14
        assert result[language]["fontWeight"] >= 600
        assert result[language]["minGap"] > 0
        assert result[language]["metaFontSize"] >= 12.8
        assert result[language]["legendFontSize"] >= 13.12
    assert result["errors"] == []


def test_full_automatic_mode_is_the_clear_bilingual_web_default() -> None:
    _reset_state()
    result = _run_browser(
        """
        await waitFor('#runForm');
        if (document.documentElement.dataset.language !== 'en') {
          document.querySelector('#languageToggle').click();
          await sleep(100);
        }
        document.querySelector('.advanced-options').open = true;
        const readOption = () => {
          const checkbox = document.querySelector('input[name="autoApprove"]');
          return {
            checked: checkbox.checked,
            label: checkbox.closest('label').textContent.trim(),
            help: document.querySelector('.advanced-options .auto-approve-help').textContent.trim(),
          };
        };
        const english = readOption();
        document.querySelector('#languageToggle').click();
        await sleep(100);
        document.querySelector('.advanced-options').open = true;
        const chinese = readOption();
        return {english, chinese, errors: window.__browserErrors};
        """,
        reduced_motion=True,
    )

    assert result["english"]["checked"] is True
    assert result["english"]["label"] == "Fully automatic (default)"
    assert "Stage 7, 9, and 10" in result["english"]["help"]
    assert result["chinese"]["checked"] is True
    assert result["chinese"]["label"] == "全自动运行（默认）"
    assert "无需回答" in result["chinese"]["help"]
    assert result["errors"] == []


def test_live_stage_progress_is_detailed_responsive_and_bilingual() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_progress_run())
    RUN_SNAPSHOTS["progress-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        document.querySelector('[data-view="live"]').click();
        const card = await waitFor('.stage-progress-card');
        if (document.documentElement.dataset.language !== 'en') {
          document.querySelector('#languageToggle').click();
          await sleep(100);
        }
        const readProgress = () => ({
          heading: document.querySelector('.stage-progress-heading h3').textContent.trim(),
          activity: document.querySelector('.stage-progress-heading p:not(.section-label)').textContent.trim(),
          measure: document.querySelector('.stage-progress-measure').textContent.trim(),
          metrics: document.querySelector('.stage-progress-metrics').textContent.trim(),
          status: document.querySelector('.stage-progress-summary .status-pill').textContent.trim(),
          pulse: getComputedStyle(document.querySelector('.stage-progress-summary .status-pill'), '::after').animationName,
          activityAge: document.querySelector('.stage-progress-updated time').textContent.trim(),
          activityTimestamp: document.querySelector('.stage-progress-updated time').dataset.relativeTime,
          cardClass: document.querySelector('.stage-progress-card').className,
          value: document.querySelector('.stage-subprogress').getAttribute('aria-valuenow'),
          overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        });
        const english = readProgress();
        document.querySelector('#languageToggle').click();
        await sleep(100);
        const chinese = readProgress();
        return {english, chinese, errors: window.__browserErrors};
        """,
        width=375,
        reduced_motion=True,
    )

    assert result["english"]["heading"] == "03.5 · Paper deep reading"
    assert "Retrieving paper full text" in result["english"]["activity"]
    assert "11 / 20" in result["english"]["measure"]
    assert "Full text9" in result["english"]["metrics"]
    assert result["english"]["status"] == "Running"
    assert result["english"]["pulse"] == "status-signal"
    assert result["english"]["activityAge"]
    assert result["english"]["activityTimestamp"] == "2026-08-30T01:05:50+08:00"
    assert "running" in result["english"]["cardClass"]
    assert result["english"]["value"] == "55"
    assert result["chinese"]["heading"] == "03.5 · 论文全文精读"
    assert "正在获取论文全文" in result["chinese"]["activity"]
    assert "全文成功9" in result["chinese"]["metrics"]
    assert result["chinese"]["status"] == "运行中"
    assert result["chinese"]["activityAge"]
    assert result["english"]["overflow"] is False
    assert result["chinese"]["overflow"] is False
    assert result["errors"] == []


def test_structured_checkpoint_survives_polling_and_posts_all_answers() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_checkpoint_run())
    RUN_SNAPSHOTS["waiting-run"] = {**FULL_SNAPSHOT, "workspace": {"name": "waiting-run", "path": "/tmp/autoidea-browser-test/runs/waiting-run"}}
    result = _run_browser(
        """
        document.querySelector('[data-view="live"]').click();
        const text = await waitFor('textarea[name="answer-0"]');
        text.focus();
        text.value = 'limited compute';
        text.dispatchEvent(new Event('input', {bubbles: true}));
        document.querySelector('input[name="answer-1"][value="approve"]').click();
        await sleep(2600);
        const sameText = document.querySelector('textarea[name="answer-0"]');
        const stable = sameText === text && sameText.value === 'limited compute';
        document.querySelector('#interactionForm button[type="submit"]').click();
        const submittedAt = performance.now();
        while (document.querySelector('#interactionForm') && performance.now() - submittedAt < 3000) {
          await sleep(50);
        }
        return {stable, interactionGone: !document.querySelector('#interactionForm'), errors: window.__browserErrors};
        """
    )
    assert result == {"stable": True, "interactionGone": True, "errors": []}
    assert RUN_INPUTS == [{"run_id": "waiting-run", "status": "answered", "answers": ["limited compute", "approve"]}]


def test_unchanged_polling_does_not_replace_or_reanimate_the_view() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        await waitFor('#runComposer');
        await sleep(400);
        const viewRoot = document.querySelector('#viewRoot');
        const originalView = viewRoot.firstElementChild;
        const runSelector = document.querySelector('#runSelector');
        let rootMutations = 0;
        let selectorMutations = 0;
        let animationStarts = 0;
        new MutationObserver((entries) => { rootMutations += entries.length; })
          .observe(viewRoot, {childList: true});
        new MutationObserver((entries) => { selectorMutations += entries.length; })
          .observe(runSelector, {childList: true, subtree: true});
        viewRoot.addEventListener('animationstart', () => { animationStarts += 1; });
        await sleep(5000);
        return {
          sameView: originalView === viewRoot.firstElementChild,
          rootMutations,
          selectorMutations,
          animationStarts,
          errors: window.__browserErrors,
        };
        """,
    )

    # The initial request plus one unchanged polling cycle is sufficient to
    # prove that polling does not replace the rendered view.  Requiring two
    # interval ticks in a fixed five-second window is flaky on a busy runner.
    assert REQUEST_COUNTS["/api/runs"] >= 2
    assert result == {
        "sameView": True,
        "rootMutations": 0,
        "selectorMutations": 0,
        "animationStarts": 0,
        "errors": [],
    }


def test_tool_approval_posts_structured_decision() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_checkpoint_run("tool_approval"))
    RUN_SNAPSHOTS["waiting-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        document.querySelector('[data-view="live"]').click();
        const approve = await waitFor('[data-action="tool-decision"][data-decision="approve"]');
        approve.click();
        await sleep(250);
        return {gone: !document.querySelector('[data-action="tool-decision"]'), errors: window.__browserErrors};
        """
    )
    assert result == {"gone": True, "errors": []}
    assert RUN_INPUTS == [{"run_id": "waiting-run", "decision": "approve"}]


def test_graph_shows_complete_directional_research_provenance() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        await waitFor('#runComposer');
        document.querySelector('[data-view="map"]').click();
        await waitFor('#researchGraph .graph-edge[data-kind="supports"]');
        const supportEdges = [...document.querySelectorAll('.graph-edge[data-kind="supports"]')];
        const gapEdges = [...document.querySelectorAll('.graph-edge[data-kind="supports_gap"], .graph-edge[data-kind="partially_covers_gap"], .graph-edge[data-kind="challenges_gap"]')];
        const relationshipLabels = [...document.querySelectorAll('.accessible-graph-list tbody td:nth-child(2)')]
          .map((cell) => cell.textContent.trim());
        const relationshipRationales = [...document.querySelectorAll('.accessible-graph-list tbody td:nth-child(4)')]
          .map((cell) => cell.textContent.trim());
        return {
          supportEdges: supportEdges.length,
          gapEdges: gapEdges.length,
          directional: supportEdges.every((edge) => edge.getAttribute('marker-end') === 'url(#graphArrow)'),
          gapDirectional: gapEdges.every((edge) => edge.getAttribute('marker-end') === 'url(#graphArrow)'),
          leftToRight: supportEdges.every((edge) => Number(edge.getAttribute('x1')) < Number(edge.getAttribute('x2'))),
          gapLeftToRight: gapEdges.every((edge) => Number(edge.getAttribute('x1')) < Number(edge.getAttribute('x2'))),
          coverage: [...document.querySelectorAll('.provenance-score strong')].map((node) => node.textContent),
          complete: [...document.querySelectorAll('.provenance-status')].every((node) => node.classList.contains('complete')),
          relationshipLabels,
          relationshipRationales,
          errors: window.__browserErrors,
        };
        """,
        reduced_motion=True,
    )

    assert result["supportEdges"] == 2
    assert result["gapEdges"] == 2
    assert result["directional"] is True
    assert result["gapDirectional"] is True
    assert result["leftToRight"] is True
    assert result["gapLeftToRight"] is True
    assert result["coverage"] == ["2/2", "1/1"]
    assert result["complete"] is True
    assert any(
        label in {"supports evidence", "支撑证据"}
        for label in result["relationshipLabels"]
    )
    assert any(
        label in {"supports gap", "支撑研究空白"}
        for label in result["relationshipLabels"]
    )
    assert any("planning mechanism" in rationale for rationale in result["relationshipRationales"])
    assert result["errors"] == []


def test_language_switch_retranslates_workspace_status() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        await waitFor('#runComposer');
        const toggle = document.querySelector('#languageToggle');
        if (document.documentElement.lang !== 'zh-CN') toggle.click();
        const chinese = document.querySelector('#status').textContent.trim();
        toggle.click();
        const english = document.querySelector('#status').textContent.trim();
        return {chinese, english, lang: document.documentElement.lang, errors: window.__browserErrors};
        """
    )

    assert result == {
        "chinese": "研究工作区已是最新。",
        "english": "Research workspace is current.",
        "lang": "en",
        "errors": [],
    }


def test_keyboard_graph_activation_and_mobile_navigation() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        const toggle = await waitFor('#navToggle');
        toggle.click();
        const opened = toggle.getAttribute('aria-expanded') === 'true'
          && !document.querySelector('#navScrim').hidden;
        document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
        await sleep(30);
        const closed = toggle.getAttribute('aria-expanded') === 'false'
          && document.querySelector('#navScrim').hidden
          && document.activeElement === toggle;
        document.querySelector('[data-view="map"]').click();
        const node = await waitFor('#researchGraph .graph-node');
        node.focus();
        node.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
        await sleep(80);
        const selected = document.querySelector('#mapInspector h2')?.textContent;
        return {opened, closed, selected, errors: window.__browserErrors};
        """,
        width=375,
        reduced_motion=True,
    )
    assert result["opened"] is True
    assert result["closed"] is True
    assert result["selected"] not in {None, "选择图谱节点", "Select graph node"}
    assert result["errors"] == []


def test_dark_mode_uses_dark_canvas_without_layout_overflow() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        await waitFor('#runComposer');
        const canvas = getComputedStyle(document.body).backgroundColor;
        return {
          canvas,
          dark: matchMedia('(prefers-color-scheme: dark)').matches,
          overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          errors: window.__browserErrors,
        };
        """,
        width=1024,
        dark_mode=True,
    )
    assert result["dark"] is True
    assert result["canvas"] == "rgb(13, 21, 19)"
    assert result["overflow"] is False
    assert result["errors"] == []


def test_composer_config_and_sanitized_artifact_dialog_keep_user_state() -> None:
    _reset_state()
    RUNS_PAYLOAD.append(_completed_run())
    RUN_SNAPSHOTS["verified-run"] = FULL_SNAPSHOT
    result = _run_browser(
        """
        const prompt = await waitFor('#researchPrompt');
        prompt.focus();
        prompt.value = 'stable prompt during polling';
        prompt.dispatchEvent(new Event('input', {bubbles: true}));
        await sleep(2600);
        const promptStable = document.querySelector('#researchPrompt') === prompt && prompt.value === 'stable prompt during polling';
        document.querySelector('[data-view="settings"]').click();
        const model = await waitFor('#config-model');
        model.value = 'new-local-model';
        model.dispatchEvent(new Event('input', {bubbles: true}));
        document.querySelector('#configForm button[type="submit"]').click();
        await sleep(200);
        document.querySelector('[data-view="artifacts"]').click();
        const opener = await waitFor('[data-path="unsafe_notes.md"]');
        opener.focus();
        opener.click();
        await waitFor('#artifactDialog[open] .markdown-body');
        const safe = !document.querySelector('#artifactBody script') && !document.querySelector('#artifactBody [onclick]') && !document.querySelector('#artifactBody a[href]');
        document.querySelector('#closeArtifactButton').click();
        await sleep(50);
        return {promptStable, safe, focusRestored: document.activeElement === opener, xss: Boolean(window.__xss), errors: window.__browserErrors};
        """
    )
    assert result["promptStable"] is True
    assert result["safe"] is True
    assert result["focusRestored"] is True
    assert result["xss"] is False
    assert result["errors"] == []
    assert CONFIG_VALUES["model"] == "new-local-model"
    assert not any(path.endswith("/api/config/reset") for path in REQUEST_COUNTS)
    assert REQUEST_COUNTS["/api/runs/verified-run/snapshot"] >= 1


def test_saved_config_refreshes_run_defaults_without_replacing_per_run_override() -> None:
    _reset_state()
    result = _run_browser(
        """
        const initialModel = (await waitFor('#runModel')).value;
        document.querySelector('[data-view="settings"]').click();
        const model = await waitFor('#config-model');
        model.value = 'new-saved-default';
        model.dispatchEvent(new Event('input', {bubbles: true}));
        document.querySelector('#configForm button[type="submit"]').click();
        await sleep(200);
        document.querySelector('[data-view="studio"]').click();
        const refreshedModel = (await waitFor('#runModel')).value;

        const runModel = document.querySelector('#runModel');
        runModel.value = 'per-run-override';
        runModel.dispatchEvent(new Event('input', {bubbles: true}));
        document.querySelector('[data-view="settings"]').click();
        const modelAgain = await waitFor('#config-model');
        modelAgain.value = 'another-saved-default';
        modelAgain.dispatchEvent(new Event('input', {bubbles: true}));
        document.querySelector('#configForm button[type="submit"]').click();
        await sleep(200);
        document.querySelector('[data-view="studio"]').click();
        const preservedOverride = (await waitFor('#runModel')).value;

        return {initialModel, refreshedModel, preservedOverride, errors: window.__browserErrors};
        """,
        reduced_motion=True,
    )

    assert result == {
        "initialModel": "qwen-test",
        "refreshedModel": "new-saved-default",
        "preservedOverride": "per-run-override",
        "errors": [],
    }


def test_config_save_identifies_environment_override_and_keeps_it_effective() -> None:
    _reset_state()
    CONFIG_ENV_OVERRIDES["model"] = ("deployment-model", "AUTOIDEA_MODEL")
    result = _run_browser(
        """
        const initialRunModel = (await waitFor('#runModel')).value;
        document.querySelector('[data-view="settings"]').click();
        const model = await waitFor('#config-model');
        const effectiveBefore = model.closest('.config-field').querySelector('.config-effective').textContent;
        model.value = 'saved-but-overridden';
        model.dispatchEvent(new Event('input', {bubbles: true}));
        document.querySelector('#configForm button[type="submit"]').click();
        await sleep(200);
        const status = document.querySelector('#configStatus');
        const statusText = status.textContent.trim();
        const warning = status.classList.contains('warning');
        document.querySelector('[data-view="studio"]').click();
        const effectiveRunModel = (await waitFor('#runModel')).value;
        return {initialRunModel, effectiveBefore, statusText, warning, effectiveRunModel, errors: window.__browserErrors};
        """,
        reduced_motion=True,
    )

    assert result["initialRunModel"] == "deployment-model"
    assert "deployment-model" in result["effectiveBefore"]
    assert "AUTOIDEA_MODEL" in result["effectiveBefore"]
    assert "AUTOIDEA_MODEL" in result["statusText"]
    assert result["warning"] is True
    assert result["effectiveRunModel"] == "deployment-model"
    assert result["errors"] == []
