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
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
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
        return logging.getLevelName(self.log_level)

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
    return Settings()

