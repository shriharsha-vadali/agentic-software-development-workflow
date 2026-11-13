# Agentic Software Development Workflow

An AI-powered developer assistant that closes the gap between a GitHub issue and a merged Pull Request.

Given a GitHub issue URL, the agent:
1. Fetches and understands the issue
2. Analyses the target codebase using RAG (Retrieval-Augmented Generation)
3. Produces a human-readable implementation plan for review
4. Generates code changes and tests
5. Runs the tests in a sandboxed environment
6. Opens a draft Pull Request for human review

> **Status**: Phase 1 — Foundation & GitHub Integration (in progress)

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| Data validation | Pydantic + Pydantic Settings |
| ORM | SQLAlchemy 2.x (async) |
| Database | SQLite (via aiosqlite) |
| HTTP client | HTTPX (async) |
| LLM integration | OpenAI-compatible REST API |
| GitHub integration | GitHub REST API v3 |
| Testing | Pytest + pytest-asyncio |

---

## Project Structure

```
agentic-dev-workflow/
├── app/
│   ├── api/          # FastAPI routers and Pydantic request/response schemas
│   ├── adapters/     # GitHub API client and LLM service (external I/O)
│   ├── agent/        # Workflow orchestrator and agent state machine
│   ├── db/           # SQLAlchemy models and session management
│   └── services/     # Business logic (JobService)
├── tests/
│   ├── unit/         # Component tests with mocked dependencies
│   └── integration/  # End-to-end HTTP tests against a test database
├── prompts/          # Versioned LLM prompt templates
├── workspaces/       # Cloned repositories (runtime, gitignored)
├── .env.example      # Environment variable template — copy to .env
└── pyproject.toml    # Dependencies and tool configuration
```

---

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Clone the repository
```bash
git clone https://github.com/your-org/agentic-dev-workflow.git
cd agentic-dev-workflow
```

### 2. Create and activate a virtual environment
```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (macOS / Linux)
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Install application + development dependencies
pip install -e ".[dev]"
```

### 4. Configure environment variables
```bash
# Copy the template
cp .env.example .env

# Edit .env and fill in your real values:
#   GITHUB_TOKEN   — GitHub Personal Access Token
#   LLM_API_KEY    — OpenAI (or compatible) API key
#   LLM_MODEL      — Model to use (e.g. gpt-4o)
```

### 5. Start the development server
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Foundation — GitHub integration, LLM plan generation, human approval | 🚧 In Progress |
| Phase 2 | RAG-powered code generation and test execution | ⬜ Planned |
| Phase 3 | Hardening — Docker sandbox, security, real PR workflow | ⬜ Planned |
| Phase 4 | Developer experience — streaming, web UI, observability | ⬜ Planned |

---

## Security Notes

- **Never commit your `.env` file** — it is listed in `.gitignore`
- Use a [fine-grained GitHub PAT](https://github.com/settings/tokens?type=beta) with minimal scopes
- The agent only executes generated *test* code, never application code
- All test execution happens in a sandboxed subprocess (Phase 2) or Docker container (Phase 3)

---

## Contributing

This is a learning project. See the architecture document for design decisions and reasoning.

---

## License

MIT

