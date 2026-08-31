# AutoIdea Research Observatory

## Install and start

Python 3.11–3.13 is required.

```bash
python -m pip install -e ".[web]"
autoidea web --workspace examples/sample_workspace --port 8765
```

Open `http://127.0.0.1:8765`. Add `--no-open` when a browser should not be launched automatically.

The workspace passed to `autoidea web` is the dashboard's root workspace. A new browser-managed run normally receives a unique child directory under:

```text
<dashboard-workspace>/runs/<run-name>
```

If that directory already contains data, AutoIdea chooses a suffixed directory instead of mixing a new run with old artifacts.

## Configuration

Viewing an existing workspace does not require a model key. Starting or resuming a real research run requires configured model-provider and search settings.

Configure AutoIdea with environment variables, the CLI, or the Observatory's Settings view. Configuration precedence remains:

```text
per-run options > environment variables > user config file > defaults
```

For normal local use, keep credentials and custom endpoint URLs in `.env`, and
save provider, model, and pipeline defaults in the Settings view or with
`autoidea config set`. Reserve `AUTOIDEA_PROVIDER` and `AUTOIDEA_MODEL` for
deployment-wide locks; when present, the Settings view identifies that its
saved value is not currently effective. Values entered in the new-run form or
passed through CLI `--provider` / `--model` always apply to that run.

The repository does not include a `config.yaml`. The file is created with
user-only permissions when a setting is first saved. Run `autoidea config path`
to show its exact location; the default is `~/.config/autoidea/config.yaml` (or
`$XDG_CONFIG_HOME/autoidea/config.yaml`).
The view groups fields into Quick Setup, Credentials, Search & Literature,
Pipeline & Ranking, Memory & Context, and Paths & Network.

Secret values are never returned to the browser in full. The API exposes only whether a secret is configured and a masked representation. Leaving a secret input blank preserves the stored value. Environment-variable overrides are identified in the interface.

## Run lifecycle

1. Start a new run from Overview and provide a research prompt. Model, provider, seed files, and execution options can be overridden for that run.
2. The server allocates a unique run ID, workspace, and thread ID and persists them before launching the child process.
3. `autoidea.web.runner` initializes the selected run and streams structured execution events to the workbench.
4. Structured questions and tool approvals are written as JSONL events. Browser answers are saved through atomic response files, so a long-running process does not depend on an open HTTP request.
5. By default, Stage 7, Stage 9, and Stage 10 each emit requested/resolved checkpoint events and continue automatically without creating browser response files. Clear **Fully automatic** to pause at those checkpoints. In manual mode, **Continue automatically** approves the current checkpoint and makes the remainder of the run automatic.
6. After Stage 12, the server verifies observed artifacts and runs the artifact audit before it reports completion.

The built-in default is fully automatic. CLI users can opt into manual review with `--manual-checkpoints`; `--auto-approve` remains available for compatibility. The Web run form exposes the same policy for new, resumed, and follow-up runs.

### New, resume, and follow-up

- **New** creates a unique workspace and a new LangGraph thread unless a thread is explicitly supplied.
- **Resume selected** reuses the selected run's workspace, thread, model, provider, and seed configuration and submits the new prompt to that saved thread. Explicitly approved mandatory checkpoints are carried forward with provenance, so a model turn that ends immediately after approval can be resumed without losing completion proof. Rejected and ambiguous legacy decisions are never inherited as approvals.
- **Follow-up** also reuses the selected context, while recording the new run's parent relationship so the research lineage remains visible.

Each launch has its own Web run record and log. A service restart does not erase persisted metadata, checkpoint events, or response files. If a child process survives the restart, the new server can continue serving its checkpoint; when that detached process exits, the server reconstructs the terminal state from the process identity, events, artifacts, and audit evidence.

## Completion contract

A child process exiting with code `0` is not enough to call a run complete. The UI displays `pipeline_completed` only when all of the following are true:

- all 14 stage entries (Stages 1–12, including 3.5 and 9.5) have valid required artifacts;
- Stage 7 contains both `knowledge_synthesis.md` and a valid `research_gaps.json` Claim-to-Gap registry;
- Stage 7, Stage 9, and Stage 10 each have a persisted requested-and-resolved checkpoint event;
- `final_report.md` exists and passes the minimum artifact checks;
- the repository's artifact audit finishes without errors.

Observed files are authoritative. `pipeline_state.json` remains useful provenance, but a stale self-reported state cannot override missing or invalid artifacts. The Pipeline and Final report views expose the four parts of this completion proof separately.

Common terminal statuses are:

| Status | Meaning |
| --- | --- |
| `pipeline_completed` | Full completion contract passed. |
| `checkpoint_reached` | The original process ended at a recoverable prompt; resume explicitly. |
| `failed` | The process failed, or exited successfully without a complete verifiable pipeline. |
| `stopped` | The user stopped the local process. Generated artifacts remain available. |
| `stale` | Persisted process metadata no longer matches a live process and no complete proof exists. |

## Stage 7 provenance contract

Stage 7 deliberately separates readable analysis from structured provenance:

| Artifact | Role |
| --- | --- |
| `knowledge_synthesis.md` | Narrative synthesis for a researcher to review. |
| `research_gaps.json` | Canonical `G<number>` gap records and typed `C<number>` Claim-to-Gap links. |

Each `evidence_links` entry in `research_gaps.json` contains a Claim ID from
`evidence_db.json`, one of `supports`, `partial_coverage`, or `challenges`, and
a gap-specific rationale. The writer validates the JSON schema, requires
`gap_score = demand - coverage`, and refuses unknown Claim IDs. The Stage 7
gate and artifact audit repeat those cross-file checks; Stage 8 and Stage 9
must then use the same registered Gap IDs.

The Literature map consumes these records directly:

```text
Paper → Claim → Research Gap → Idea
          └────────────────────→ Idea
```

It never derives Claim-to-Gap edges by searching the Markdown. Invalid or
unknown references produce a warning and no edge. Edge labels, distinct line
styles, the inspector, and the relationship table expose the relationship role
and rationale without relying on color alone.

## Main views

- **Overview** — research trace, run composer, aggregate counts, and recent runs.
- **Live run** — process status, structured human questions, tool approvals, raw log, and stop control.
- **Final report** — sanitized rendering of the selected run's `final_report.md` plus its completion proof.
- **Literature map** — keyboard-accessible Paper → Claim → Research Gap → Idea provenance graph with pan, zoom, filters, inspector, and a relationship table containing each edge's recorded rationale.
- **Papers** — source metadata, relevance, critique-first position, weakest link, and dimension analysis.
- **Evidence** — citation ID, claim, source, section, evidence type, and confidence.
- **Ideas** — candidate descriptions, scores, target gaps, evidence links, and self-assessment.
- **Pipeline** — all 14 observed stages, required artifacts, three checkpoint records, and audit result.
- **Artifacts** — run-scoped Markdown, JSON, and text records.
- **Settings** — browser-safe editing of the shared AutoIdea configuration.

Run selection scopes snapshots, artifacts, reports, logs, and completion proofs to that run. Historical files from another run are not used to declare the selected run complete.

## HTTP API

### Workspace and configuration

- `GET /api/health` — server status and root workspace.
- `GET /api/snapshot` — normalized root-workspace snapshot.
- `GET /api/snapshot?run_id=<id>` — normalized snapshot for a managed run.
- `GET /api/artifacts/{path}` — safely read a root-workspace artifact; path traversal is rejected.
- `GET /api/config` — grouped, browser-safe configuration metadata.
- `PATCH /api/config` — validate and persist selected configuration values.
- `POST /api/config/reset` — reset file-backed configuration to defaults. This is destructive and is not exposed as a routine UI action.

Example configuration update:

```json
{
  "values": {
    "provider": "openai",
    "model": "gpt-5.6-sol",
    "max_tokens": 24576,
    "show_thinking": true
  }
}
```

### Managed runs

- `GET /api/runs` — list persisted browser-managed runs with live status and completion proof.
- `POST /api/runs` — start a new, resume, or follow-up run.
- `GET /api/runs/{id}` — return one run, including the recent log tail and pending interaction.
- `GET /api/runs/{id}/snapshot` — normalized run-scoped workspace snapshot, including that run's persisted checkpoint and audit proof rather than an unscoped artifact-only estimate.
- `GET /api/runs/{id}/artifacts/{path}` — safely read a run-scoped artifact.
- `GET /api/runs/{id}/events` — return structured runner events.
- `POST /api/runs/{id}/input` — answer the current structured checkpoint or tool approval.
- `POST /api/runs/{id}/followup` — compatibility path for resuming a legacy checkpoint review.
- `POST /api/runs/{id}/stop` — stop the managed process group while preserving artifacts.

Example run request:

```json
{
  "prompt": "Generate grounded research ideas for ...",
  "run_name": "my-run",
  "workspace": "",
  "model": "",
  "provider": "",
  "thread_id": "",
  "seed_papers": "",
  "seed_ideas": "",
  "auto_approve": true,
  "show_thinking": true,
  "mode": "new",
  "parent_run_id": ""
}
```

## Runtime files

Web orchestration data is stored under the dashboard root workspace:

```text
.autoidea_web/
  runs.json
  logs/<run-id>.log
  events/<run-id>.jsonl
  responses/<run-id>/<interaction-id>.json
runs/<unique-run-name>/
  ... AutoIdea stage artifacts ...
  final_report.md
```

`runs.json` intentionally does not duplicate log tails or pending interaction payloads; those are reconstructed from the append-only log and event files.

## Local-operation and security boundary

The current workbench is designed for one trusted user on the same machine:

- keep the default `127.0.0.1` binding;
- do not expose it directly to the Internet or an untrusted LAN;
- the server can start local processes, edit AutoIdea configuration, write run workspaces, and stop managed process groups;
- artifact paths are confined to the selected workspace, and rendered Markdown is sanitized, but the API does not provide multi-user authentication, authorization, TLS, quotas, or tenant isolation;
- AutoIdea does not upload dashboard data to a hosted AutoIdea service, but configured model and search providers receive the same requests they would receive from CLI execution.

## Smoke verification

```bash
autoidea web --workspace examples/sample_workspace --port 8765 --no-open
curl -fsS http://127.0.0.1:8765/api/health
curl -fsS http://127.0.0.1:8765/api/runs
curl -fsS http://127.0.0.1:8765/api/snapshot
```

These checks verify serving and workspace parsing. A full acceptance test must additionally start an isolated run, observe all three requested/resolved checkpoint event pairs without manual answers in automatic mode, reach Stage 12, verify `final_report.md`, pass the artifact audit, and observe `pipeline_completed` after a page reload or service restart.
