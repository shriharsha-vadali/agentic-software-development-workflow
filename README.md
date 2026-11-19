# Agentic Software Development Workflow

An autonomous software engineering system that converts GitHub issues into working, tested code changes and GitHub Pull Requests, with a mandatory human approval checkpoint.

---

## 1. Project Overview

The **Agentic Software Development Workflow** bridges the gap between issue reporting and production-grade pull requests. It ingests GitHub issues, understands the target codebase using AST-aware Retrieval-Augmented Generation (RAG), synthesizes a structured implementation plan, halts at a mandatory human approval gate, generates minimal reviewable code changes in an isolated workspace, executes automated tests in a sandboxed subprocess with bounded repair attempts, and opens a GitHub Pull Request upon explicit authorization.

### Core Workflow

```mermaid
flowchart TD
    A[GitHub Issue] --> B[Issue Ingestion & Normalization]
    B --> C[Repository Workspace & AST Indexing]
    C --> D[Context Retrieval - RAG]
    D --> E[AI Implementation Planning]
    E --> F{Human Approval Gate}
    F -- Rejected with Feedback --> E
    F -- Approved --> G[AI Code Generation & Patching]
    G --> H[Isolated Sandboxed Test Execution]
    H -- Tests Fail --> I{Repair Retries Remaining?}
    I -- Yes --> G
    I -- No --> J[FAILED State - Diagnostic Logs]
    H -- Tests Pass --> K{PR Authorization Gate}
    K -- Authorized --> L[Git Branch, Commit & Pull Request]
    L --> M[COMPLETED State]
```

---

## 2. Technology Stack

| Layer | Technology | Description |
|---|---|---|
| **Web Framework** | FastAPI (async) | High-performance, async-native API framework with automatic OpenAPI docs. |
| **Validation & Schemas** | Pydantic v2 + Pydantic Settings | Strict data validation, environment variable parsing, and JSON schema generation. |
| **Persistence & ORM** | SQLAlchemy 2.x (async) | Eager relationship loading, partial unique indexing, and async SQLite via `aiosqlite`. |
| **Code Understanding & RAG** | Python `ast` + Custom Hybrid Retriever | AST parsing for functions/classes, keyword weighting, and token-budgeted context formatting. |
| **LLM Integration** | Async HTTPX Client | Provider-independent architecture with `OpenAILLMClient` and deterministic `MockLLMClient`. |
| **Git & PR Automation** | GitHub REST API v3 + Git CLI | Branch creation, patch application, unified diff generation, and Pull Request creation. |
| **Sandboxed Execution** | Subprocess Sandbox Runner | Command whitelisting, environment secret scrubbing, and timeout protection. |
| **Testing & Quality** | Pytest + pytest-asyncio, Ruff, Mypy | 293 automated tests, 100% strict type annotations, and Ruff lint/formatting compliance. |

---

## 3. Project Structure

```text
agentic-dev-workflow/
├── app/
│   ├── adapters/         # External service clients (GitHub API, LLM providers)
│   ├── api/              # FastAPI routers, endpoints, and Pydantic schemas
│   ├── core/             # Configuration management and application settings
│   ├── db/               # SQLAlchemy models (Job, Plan, Artifact), sessions, migrations
│   ├── execution/        # Sandboxed subprocess runner, test execution, repair loop
│   ├── intelligence/     # Codebase indexing, AST parsing, chunking, RAG retriever, coder
│   ├── orchestrator/     # Persistent state machine, background tasks, crash recovery
│   ├── services/         # Business services (issue ingestion, approval, git, pull requests)
│   └── main.py           # Application factory, lifespan management, startup recovery
├── tests/
│   ├── unit/             # Unit tests for adapters, intelligence, execution, APIs, models
│   └── integration/      # End-to-end integration tests with mocked external services
├── workspaces/           # Isolated job workspaces (gitignored)
├── .env.example          # Environment variable template
└── pyproject.toml        # Dependencies, tool configurations, and test runners
```

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.11 or higher
- Git 2.30+

### 1. Clone the repository
```bash
git clone https://github.com/shriharsha-vadali/agentic-software-development-workflow.git
cd agentic-software-development-workflow
```

### 2. Set up virtual environment
```bash
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e ".[dev]"
```

### 4. Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your credentials:
```env
GITHUB_TOKEN=ghp_your_personal_access_token_here
GITHUB_API_BASE_URL=https://api.github.com
GITHUB_TIMEOUT_SECONDS=30.0

LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-api-key-here
LLM_MODEL=gpt-4o

DATABASE_URL=sqlite+aiosqlite:///./jobs.db
LOG_LEVEL=INFO
MAX_RETRIES=3
EXECUTION_TIMEOUT_SECONDS=60.0
```
*(Note: If `LLM_API_KEY` contains placeholder text, the application automatically uses the deterministic `MockLLMClient` for offline testing.)*

### 5. Run the application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Swagger UI Documentation: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

---

## 5. API Reference & Workflow Walkthrough

### 1. Ingest an Issue
```http
POST /jobs/ingest
Content-Type: application/json

{
  "issue_url": "https://github.com/octocat/Hello-World/issues/42"
}
```
*Creates Job (Status: `PENDING`), ingests issue metadata, stores `issue_data` artifact, and launches background planning.*

### 2. Inspect Job Progress & Plan
```http
GET /jobs/{id}
```
*When planning completes, the job pauses in `AWAITING_APPROVAL` with the generated `Plan` in `draft` status.*

```http
GET /jobs/{id}/plan
```
*Returns the structured implementation plan, problem statement, affected files, steps, test plan, and complexity estimate.*

### 3. Human Approval Checkpoint
#### Approve Plan:
```http
POST /jobs/{id}/approve
Content-Type: application/json

{
  "decision": "approved"
}
```
*Advances Job to `GENERATING_CODE`. Unapproved execution is strictly rejected by the server.*

#### Reject Plan (with Revision Feedback):
```http
POST /jobs/{id}/approve
Content-Type: application/json

{
  "decision": "rejected",
  "feedback": "Please use aiosqlite instead of synchronous sqlite3 and add regression tests."
}
```
*Transitions Job back to `PLANNING` for AI re-planning incorporating the human feedback.*

### 4. Execute Code Generation & Sandboxed Tests
```http
POST /jobs/{id}/execute
```
*Applies verified code modifications to the isolated workspace, saves the `diff` artifact, and executes pytest in a sandboxed subprocess. If tests fail, the bounded repair engine automatically initiates targeted fixes up to `max_retries`.*

### 5. Authorize Pull Request Submission
```http
POST /jobs/{id}/approve-pr
```
*Explicit authorization gate: checkouts dedicated feature branch `agent/issue-{num}-{id}`, commits approved changes, opens Pull Request via GitHub API, records `pr_url` artifact, and transitions Job to terminal state `COMPLETED`.*

### 6. Job Cancellation
```http
POST /jobs/{id}/cancel
```
*Cancels active in-flight tasks and cleanly transitions Job to `FAILED` with diagnostic logs.*

---

## 6. Security Boundaries & Design

1. **Mandatory Human Approval Gate**: Target repository workspace files are NEVER modified before an explicit human approval is received.
2. **Explicit PR Authorization**: Pull requests are never created or pushed without human authorization.
3. **Subprocess Sandbox Isolation**:
   - Commands are strictly checked against a whitelist (`pytest`, `python`, `git`, `ruff`, `mypy`).
   - Shell string execution (`shell=True`) is prohibited.
   - Sensitive environment variables (`GITHUB_TOKEN`, `LLM_API_KEY`) are scrubbed before subprocess spawning.
   - Execution timeouts prevent runaway or malicious loops.
4. **Path Traversal Protection**: All file modifications are resolved against the job's isolated workspace root; traversal attempts (`../`, absolute paths, `.git/` tampering) raise immediate exceptions.
5. **Untrusted Input Handling**: Issue titles, descriptions, and comments are treated as untrusted data and isolated within sanitized prompt blocks to prevent prompt injection.
6. **Secret Masking**: Tokens are wrapped in Pydantic's `SecretStr` and excluded from logs, exceptions, and `repr`.

> [!NOTE]
> Subprocess isolation provides controlled command boundaries for testing. In untrusted multi-tenant production deployments, this should be complemented with containerized Docker or gVisor sandbox runtimes.

---

## 7. Running Tests & Quality Verification

All 293 tests run without real GitHub credentials or paid LLM API calls using deterministic mocks:

```bash
# Run full test suite (unit + integration)
python -m pytest tests/ -v

# Run Ruff linter
python -m ruff check app/ tests/

# Run Ruff formatting verification
python -m ruff format --check app/ tests/

# Run Mypy static type checker
python -m mypy app/
```

---

## 8. License

MIT
