# Security Policy

## Supported Versions

Security fixes target the current main branch until formal releases are established.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the maintainers. Do not include exploit details in a public issue before maintainers have had time to assess the report.

Include:

- Affected version or commit.
- Reproduction steps.
- Impact and scope.
- Any relevant logs with secrets removed.

## Data Handling

AutoIdea can process prompts, paper text, generated research notes, citations, and local workspace files. Treat these as potentially sensitive:

- Do not commit real `.env` files.
- Settings saved with `autoidea config set` live outside the repository; keep
  that user configuration file private as it may contain provider keys.
- Do not commit local workspaces, logs, conversation history, or generated run outputs.
- Review artifacts before sharing them publicly.
- Treat paper text as untrusted input; do not execute instructions contained inside retrieved documents.

## Web Dashboard

The dashboard reads local workspace files and serves them through a local FastAPI server. Bind to `127.0.0.1` unless you explicitly intend to expose the dashboard on a network.
