"""
app/core/config.py
------------------
Central configuration management for the Agentic Software Development Workflow.

All settings are loaded from environment variables. In development, Pydantic
Settings automatically reads a `.env` file in the project root so developers
do not have to export variables by hand.

Design principles
-----------------
Single source of truth
    Every configurable value lives here. No other module reads os.environ
    or os.getenv directly — they all call get_settings().

Fail fast
    Missing required fields (GITHUB_TOKEN, LLM_API_KEY) raise a clear
    ValidationError at import time, before the application accepts any requests.
    A clear error beats a cryptic AttributeError inside a route handler.

Secret safety
    API keys and tokens use Pydantic's SecretStr type. The raw value never
    appears in repr(), str(), logs, or tracebacks. The only way to read it
    is to call .get_secret_value() explicitly — a deliberate, visible action.

Singleton
    get_settings() is wrapped with @lru_cache so the .env file is parsed
    and all validators run exactly once per process, regardless of how many
    modules call get_settings(). In tests, call get_settings.cache_clear()
    to reset between test cases that override environment variables.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Type alias: restricts log_level to the five standard Python logging levels.
# Pydantic enforces this at validation time — any other string is rejected.
# ---------------------------------------------------------------------------
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """
    Application-wide settings populated from environment variables.

    Pydantic Settings resolves values in this priority order (highest first):

      1. Values passed directly to Settings(...) — used in tests.
      2. Environment variables in the running process (os.environ).
      3. Variables in the .env file at the project root.
      4. Default values defined on each field below.

    Fields with no default are REQUIRED. The application will not start
    without them.
    """

    model_config = SettingsConfigDict(
        # Read from a .env file in the current working directory.
        # In production, omit the .env file and inject variables through
        # the deployment environment (Docker, systemd, cloud secret manager).
        env_file=".env",
        env_file_encoding="utf-8",
        # Treat GITHUB_TOKEN and github_token as the same variable so
        # developers can use either casing in their .env file.
        case_sensitive=False,
        # Silently ignore any .env keys we do not recognise.
        # Set to "forbid" if you want to catch typos in variable names.
        extra="ignore",
        # Allow fields to be populated by their Python attribute name in
        # addition to any alias, useful when constructing Settings in tests.
        populate_by_name=True,
    )

    # =========================================================================
    # Application / general settings
    # =========================================================================

    log_level: LogLevel = Field(
        default="INFO",
        description=(
            "Logging verbosity for the entire application.\n"
            "  DEBUG   — every SQL query, every HTTP request/response body.\n"
            "  INFO    — normal operational events (recommended for development).\n"
            "  WARNING — only unexpected situations worth attention.\n"
            "  ERROR   — failures that prevented an operation from completing.\n"
            "  CRITICAL— application-level failures.\n"
            "Normalised to uppercase automatically, so 'debug' and 'DEBUG' are equivalent."
        ),
    )

    # =========================================================================
    # Database settings
    # =========================================================================

    database_url: str = Field(
        default="sqlite+aiosqlite:///./jobs.db",
        description=(
            "SQLAlchemy async connection string.\n"
            "Phase 1 uses SQLite — no separate database server is required.\n"
            "Format:  sqlite+aiosqlite:///./relative/path/to/file.db\n"
            "         sqlite+aiosqlite:////absolute/path/to/file.db\n"
            "         sqlite+aiosqlite:///:memory:   ← fastest, used in tests\n"
            "The 'aiosqlite' driver is mandatory because FastAPI and SQLAlchemy\n"
            "run all database I/O asynchronously.\n"
            "To migrate to PostgreSQL in a later phase, change only this string."
        ),
    )

    # =========================================================================
    # GitHub adapter settings
    # =========================================================================

    github_token: SecretStr = Field(
        description=(
            "GitHub Personal Access Token (PAT). REQUIRED.\n"
            "Minimum fine-grained scopes needed:\n"
            "  - Contents:       Read and Write  (clone repo, push branch)\n"
            "  - Pull Requests:  Write           (open draft PRs)\n"
            "  - Issues:         Read            (fetch issue metadata)\n"
            "Create a fine-grained PAT at: https://github.com/settings/tokens?type=beta\n"
            "Never use a classic token with full 'repo' scope — apply least privilege.\n"
            "Stored as SecretStr: the raw value is never logged or printed."
        ),
    )

    github_api_base_url: str = Field(
        default="https://api.github.com",
        description=(
            "Root URL for all GitHub REST API calls.\n"
            "Only change this if you are using GitHub Enterprise Server.\n"
            "Trailing slashes are stripped automatically by a validator."
        ),
    )

    github_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description=(
            "HTTP request timeout in seconds for GitHub API calls.\n"
            "Must be a positive float (default: 30.0 seconds)."
        ),
    )

    # =========================================================================
    # LLM service settings
    # =========================================================================

    llm_api_key: SecretStr = Field(
        description=(
            "API key for the LLM provider. REQUIRED.\n"
            "Works with any OpenAI-compatible endpoint:\n"
            "  - OpenAI:    https://platform.openai.com/api-keys\n"
            "  - Anthropic: https://console.anthropic.com/settings/keys\n"
            "  - Local:     Ollama does not require a key — use any string.\n"
            "Stored as SecretStr: the raw value is never logged or printed.\n"
            "To read the actual key: settings.llm_api_key.get_secret_value()"
        ),
    )

    llm_model: str = Field(
        default="gpt-4o",
        description=(
            "Model identifier sent to the LLM API for every inference call.\n"
            "Phase 1 uses this model for plan generation only.\n"
            "Examples:\n"
            "  gpt-4o                       ← OpenAI (default)\n"
            "  gpt-4o-mini                  ← OpenAI, cheaper and faster\n"
            "  claude-3-5-sonnet-20241022   ← Anthropic\n"
            "  llama3.2                     ← Local Ollama\n"
            "Swapping models requires changing only this value — no code changes."
        ),
    )

    llm_api_base_url: str = Field(
        default="https://api.openai.com/v1",
        description=(
            "Base URL for all LLM API HTTP requests.\n"
            "The adapter appends paths like '/chat/completions' to this URL.\n"
            "Override to point at:\n"
            "  - A local Ollama server:  http://localhost:11434/v1\n"
            "  - An OpenAI-compatible proxy or gateway.\n"
            "Trailing slashes are stripped automatically to prevent double slashes\n"
            "when path segments are appended (e.g. base_url + '/chat/completions')."
        ),
    )

    # =========================================================================
    # Agent behaviour settings
    # =========================================================================

    workspace_dir: Path = Field(
        default=Path("./workspaces"),
        description=(
            "Directory where the agent clones target GitHub repositories.\n"
            "Declared as pathlib.Path (not str) so callers receive a typed\n"
            "object and can use path operations directly:\n"
            "  workspace = settings.workspace_dir / owner / repo\n"
            "The application creates this directory at startup if it is absent.\n"
            "This directory is listed in .gitignore — it is runtime data."
        ),
    )

    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum times the agent retries code generation after test failures.\n"
            "Must be an integer between 1 and 10 (inclusive).\n"
            "Tradeoff:\n"
            "  Higher → more self-correction attempts, more LLM API cost.\n"
            "  Lower  → fails faster, cheaper but less likely to self-recover.\n"
            "3 is a reasonable default for a learning project."
        ),
    )

    # =========================================================================
    # Sandboxed execution / Docker isolation settings
    # =========================================================================

    execution_backend: str = Field(
        default="docker",
        description=(
            "Backend used to execute test and analysis commands.\n"
            "  'docker'           — executes inside isolated disposable Docker containers.\n"
            "  'local_subprocess' — executes on the host OS (requires allow_host_execution=True)."
        ),
    )

    allow_host_execution: bool = Field(
        default=False,
        description=(
            "Explicit security authorization to permit uncontainerized host OS execution.\n"
            "MUST be explicitly set to True when using 'local_subprocess' backend.\n"
            "Prevents silent fallback to host execution when Docker is unavailable."
        ),
    )

    docker_image: str = Field(
        default="python:3.11-slim",
        description="Docker image used for isolated project execution.",
    )

    docker_network_disabled: bool = Field(
        default=True,
        description="Disable networking in execution containers (--network none).",
    )

    docker_memory_limit: str = Field(
        default="512m",
        description="Memory limit for execution containers (e.g., '512m', '1g').",
    )

    docker_cpu_limit: float = Field(
        default=1.0,
        gt=0.0,
        le=16.0,
        description="CPU limit for execution containers (e.g., 1.0 for 1 CPU core).",
    )

    docker_pids_limit: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Process/thread limit for execution containers (--pids-limit).",
    )

    docker_read_only_rootfs: bool = Field(
        default=True,
        description="Mount the container root filesystem as read-only (--read-only).",
    )

    docker_user: str = Field(
        default="1000:1000",
        description="Non-root user UID:GID for container execution (--user).",
    )

    # =========================================================================
    # Environment & API Authentication settings
    # =========================================================================

    environment: str = Field(
        default="development",
        description=(
            "Application deployment environment: "
            "'development', 'staging', 'production', or 'test'.\n"
            "In 'production', api_auth_enabled is strictly required and enforced."
        ),
    )

    api_auth_enabled: bool = Field(
        default=False,
        description=(
            "Enable API key authentication for protected API endpoints.\n"
            "When False (default for dev/tests), requests are permitted without authentication.\n"
            "When True, requests must supply a valid API key via 'X-API-Key' or Bearer token.\n"
            "Must be True when environment is 'production'."
        ),
    )

    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Secret API key required for client authentication when api_auth_enabled is True.\n"
            "Must be supplied via environment variable API_KEY.\n"
            "Stored as SecretStr: never printed or logged."
        ),
    )

    # =========================================================================
    # Validators — run automatically during model instantiation
    # =========================================================================

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, value: object) -> str:
        """
        Accept log level in any capitalisation.

        Why:  Developers often write 'debug' in .env files; Python's logging
              module only recognises uppercase strings like 'DEBUG'.

        Before: 'debug' | 'Debug' | 'DEBUG'
        After:  'DEBUG'

        The Literal["DEBUG","INFO",...] annotation then validates the
        uppercased value, so 'verbose' would still be rejected.
        """
        return str(value).upper()

    @field_validator("github_api_base_url", "llm_api_base_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, value: object) -> str:
        """
        Remove trailing slashes from base URLs.

        Why:  The adapter builds endpoint URLs by concatenating:
                  f"{base_url}/chat/completions"
              A trailing slash on base_url would produce a double slash:
                  "https://api.openai.com/v1//chat/completions"
              which some HTTP servers reject.

        Before: 'https://api.openai.com/v1/'
        After:  'https://api.openai.com/v1'
        """
        return str(value).rstrip("/")

    @field_validator("execution_backend", mode="before")
    @classmethod
    def normalise_execution_backend(cls, value: object) -> str:
        """Normalise execution backend and restrict to supported options."""
        if not isinstance(value, str):
            raise ValueError("execution_backend must be a string")
        val = value.strip().lower()
        if val not in ("docker", "local_subprocess"):
            raise ValueError(
                f"Invalid execution_backend '{value}'. Must be 'docker' or 'local_subprocess'."
            )
        return val

    @field_validator("docker_image")
    @classmethod
    def validate_docker_image_setting(cls, value: str) -> str:
        """Validate that docker_image contains only safe characters without shell injection."""
        clean = value.strip()
        pattern = (
            r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*(?:/[a-zA-Z0-9][a-zA-Z0-9_.-]*)*(?::[a-zA-Z0-9_.-]+)?$"
        )
        if not clean or len(clean) > 255 or not re.match(pattern, clean):
            raise ValueError(
                f"Invalid or unsafe docker_image '{value}'. "
                "Must be alphanumeric with optional repository and tag (e.g. 'python:3.11-slim')."
            )
        return clean

    @field_validator("docker_memory_limit")
    @classmethod
    def validate_docker_memory_limit(cls, value: str) -> str:
        """Ensure memory limit conforms to Docker format (e.g. 512m, 1g)."""
        clean = value.strip().lower()
        if not re.match(r"^\d+[bkmg]$", clean):
            raise ValueError(
                f"Invalid docker_memory_limit '{value}'. "
                "Must be an integer followed by b, k, m, or g (e.g. '512m')."
            )
        return clean

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_sqlite_only(cls, value: object) -> str:
        """
        Enforce SQLite-only constraint for Phase 1.

        Why:  Phase 1 installs only 'aiosqlite' as the async database driver.
              Accidentally pointing at a PostgreSQL or MySQL URL would produce
              a confusing ImportError deep inside SQLAlchemy at runtime.
              This validator makes the constraint explicit and the error message
              clear.

        Phase 2 / 3 can remove this validator once additional drivers are added.
        """
        url = str(value)
        if not url.startswith("sqlite"):
            raise ValueError(
                f"Phase 1 only supports SQLite database URLs.\n"
                f"  Received: '{url}'\n"
                f"  Expected: a URL starting with 'sqlite+aiosqlite:///'\n"
                f"  Example:  sqlite+aiosqlite:///./jobs.db"
            )
        return url

    @field_validator("environment", mode="before")
    @classmethod
    def normalise_environment(cls, value: object) -> str:
        """Normalise environment to lowercase and restrict to valid options."""
        if not isinstance(value, str):
            raise ValueError("environment must be a string")
        clean = value.strip().lower()
        allowed = ("development", "staging", "production", "test")
        if clean not in allowed:
            raise ValueError(
                f"Invalid environment '{value}'. Must be one of: {', '.join(allowed)}."
            )
        return clean

    @model_validator(mode="after")
    def validate_auth_configuration(self) -> Settings:
        """
        Enforce fail-closed API authentication rules across environments.
        """
        env = self.environment.lower()
        if env == "production":
            if not self.api_auth_enabled:
                raise ValueError("api_auth_enabled must be True when environment is 'production'.")

        if self.api_auth_enabled:
            if self.api_key is None:
                raise ValueError("api_key must be configured when api_auth_enabled is True.")
            raw_key = self.api_key.get_secret_value().strip()
            if not raw_key:
                raise ValueError(
                    "api_key cannot be empty or whitespace when api_auth_enabled is True."
                )
            if env == "production":
                insecure_keys = {
                    "secret",
                    "password",
                    "changeme",
                    "default",
                    "12345678",
                    "test",
                    "admin",
                    "api_key",
                }
                if raw_key.lower() in insecure_keys or any(
                    raw_key.lower().startswith(p) for p in ("changeme", "password", "default")
                ):
                    raise ValueError(
                        f"Insecure production api_key detected: '{raw_key}' is not permitted."
                    )
                if len(raw_key) < 16:
                    raise ValueError(
                        "api_key must be at least 16 characters in production environments."
                    )
        return self

    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def log_level_int(self) -> int:
        """
        Numeric logging level for use with Python's stdlib logging module.

        Usage:
            import logging
            logging.basicConfig(level=get_settings().log_level_int)

        Why a property instead of a method:
            Properties read like data attributes, which fits the mental model
            of "this is a configuration value" rather than "this is an action".
        """
        level = logging.getLevelName(self.log_level)
        return int(level)

    @property
    def is_debug(self) -> bool:
        """
        True when log_level is DEBUG.

        Usage in route handlers:
            if get_settings().is_debug:
                logger.debug("Full request payload: %s", payload)

        Avoids scattering  settings.log_level == "DEBUG"  comparisons
        throughout the codebase — a single named concept is easier to search
        and change.
        """
        return self.log_level == "DEBUG"


# =============================================================================
# Singleton accessor
# =============================================================================


@lru_cache
def get_settings() -> Settings:
    """
    Return the application Settings singleton.

    @lru_cache ensures the .env file is parsed and all validators run
    exactly once per process, no matter how many modules call get_settings().
    This is both a performance optimisation (no repeated file I/O) and a
    correctness guarantee (every part of the app sees the same config).

    Testing pattern
    ---------------
    To use a different configuration in a test:

        import pytest
        from app.core.config import get_settings

        def test_something(monkeypatch):
            monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
            get_settings.cache_clear()          # ← discard the cached instance
            settings = get_settings()
            assert settings.llm_model == "gpt-4o-mini"
            get_settings.cache_clear()          # ← clean up for the next test

    Usage everywhere else
    ---------------------
        from app.core.config import get_settings

        settings = get_settings()
        print(settings.llm_model)
        print(settings.github_token.get_secret_value())  # only place to unwrap
    """
    return Settings()  # type: ignore[call-arg]
