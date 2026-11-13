"""
tests/unit/test_config.py
--------------------------
Unit tests for the application configuration layer (app/core/config.py).

Testing philosophy
------------------
Every test instantiates Settings() with explicit constructor arguments so it
never depends on a real .env file or the developer's shell environment.
This keeps tests hermetic and runnable in any CI environment.

VALID_OVERRIDES provides the two required fields (github_token, llm_api_key)
that have no default value. Any test that constructs Settings() without
testing those specific fields must include VALID_OVERRIDES to get a valid
instance.

For tests that specifically verify missing-required-field behaviour, we use
pytest's tmp_path fixture to change the working directory to a temporary
folder that contains no .env file, preventing Pydantic Settings from
discovering and loading the project-root .env.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_OVERRIDES: dict[str, str] = {
    "github_token": "ghp_test_token_1234",
    "llm_api_key": "sk-test-key-5678",
}
"""
Minimum kwargs to supply both required fields when constructing Settings().
Merge these into every Settings() call that is not specifically testing
a required-field absence:

    s = Settings(**VALID_OVERRIDES, log_level="debug")
"""


# ===========================================================================
# Tests — required fields
# ===========================================================================

class TestRequiredFields:
    """
    Verify that Settings refuses to initialise when required fields are absent.

    Strategy: change the working directory to tmp_path (a fresh temp folder
    with no .env file) so Pydantic Settings cannot fall back to any local
    .env, then omit the target field from the constructor.
    """

    def test_missing_github_token_raises_validation_error(self, monkeypatch, tmp_path):
        """
        GITHUB_TOKEN has no default and no .env fallback → ValidationError.
        The error must name the missing field.
        """
        monkeypatch.chdir(tmp_path)                       # no .env file here
        monkeypatch.delenv("GITHUB_TOKEN", raising=False) # clear any shell export

        with pytest.raises(ValidationError) as exc_info:
            Settings(llm_api_key="sk-test")

        # Confirm the error message names the right field
        missing_fields = [e["loc"][0] for e in exc_info.value.errors()]
        assert "github_token" in missing_fields

    def test_missing_llm_api_key_raises_validation_error(self, monkeypatch, tmp_path):
        """
        LLM_API_KEY has no default and no .env fallback → ValidationError.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Settings(github_token="ghp_test")

        missing_fields = [e["loc"][0] for e in exc_info.value.errors()]
        assert "llm_api_key" in missing_fields

    def test_both_required_fields_present_succeeds(self):
        """Providing both required fields must produce a valid Settings object."""
        settings = Settings(**VALID_OVERRIDES)
        assert isinstance(settings, Settings)


# ===========================================================================
# Tests — default values
# ===========================================================================

class TestDefaultValues:
    """
    Verify that every field with a default produces the expected value
    when not overridden.
    """

    def setup_method(self):
        """Create a fresh Settings instance for each test in this class."""
        self.s = Settings(**VALID_OVERRIDES)

    def test_log_level_defaults_to_info(self):
        assert self.s.log_level == "INFO"

    def test_database_url_default(self):
        assert self.s.database_url == "sqlite+aiosqlite:///./jobs.db"

    def test_github_api_base_url_default(self):
        assert self.s.github_api_base_url == "https://api.github.com"

    def test_llm_model_defaults_to_gpt4o(self):
        assert self.s.llm_model == "gpt-4o"

    def test_llm_api_base_url_default(self):
        assert self.s.llm_api_base_url == "https://api.openai.com/v1"

    def test_max_retries_defaults_to_3(self):
        assert self.s.max_retries == 3

    def test_workspace_dir_is_path_object(self):
        """
        workspace_dir must be a pathlib.Path, not a plain str.
        This matters because callers use path operations on it:
            workspace = settings.workspace_dir / "owner" / "repo"
        """
        assert isinstance(self.s.workspace_dir, Path)

    def test_workspace_dir_default_path(self):
        assert self.s.workspace_dir == Path("./workspaces")


# ===========================================================================
# Tests — SecretStr fields do not leak sensitive values
# ===========================================================================

class TestSecretFields:
    """
    Verify that API keys and tokens are never exposed through normal Python
    string operations.  The only sanctioned way to read the raw value is
    .get_secret_value().
    """

    def setup_method(self):
        self.s = Settings(**VALID_OVERRIDES)

    def test_github_token_not_in_repr(self):
        """
        repr(settings) must not contain the raw token.
        This matters for log lines like: logger.debug("Config: %r", settings)
        """
        assert "ghp_test_token_1234" not in repr(self.s)

    def test_github_token_not_in_str(self):
        """str(settings.github_token) must return the masked placeholder."""
        assert "ghp_test_token_1234" not in str(self.s.github_token)

    def test_llm_api_key_not_in_repr(self):
        assert "sk-test-key-5678" not in repr(self.s)

    def test_llm_api_key_not_in_str(self):
        assert "sk-test-key-5678" not in str(self.s.llm_api_key)

    def test_github_token_readable_via_get_secret_value(self):
        """
        get_secret_value() is the deliberate, explicit way to unwrap a secret.
        It should return the exact string that was provided.
        """
        assert self.s.github_token.get_secret_value() == "ghp_test_token_1234"

    def test_llm_api_key_readable_via_get_secret_value(self):
        assert self.s.llm_api_key.get_secret_value() == "sk-test-key-5678"


# ===========================================================================
# Tests — field validators
# ===========================================================================

class TestValidators:
    """
    Each validator in Settings has a corresponding test here.
    Tests are named after the behaviour they assert, not the validator method.
    """

    # --- log_level normalisation -------------------------------------------

    @pytest.mark.parametrize("raw,expected", [
        ("debug",   "DEBUG"),
        ("DEBUG",   "DEBUG"),
        ("Debug",   "DEBUG"),
        ("info",    "INFO"),
        ("WARNING", "WARNING"),
        ("error",   "ERROR"),
    ])
    def test_log_level_normalised_to_uppercase(self, raw, expected):
        """
        log_level accepts any capitalisation and normalises to uppercase.
        Uses parametrize to test six inputs without repeating assertion code.
        """
        s = Settings(**VALID_OVERRIDES, log_level=raw)
        assert s.log_level == expected

    def test_invalid_log_level_rejected(self):
        """
        'VERBOSE' is not a standard Python log level.
        Pydantic's Literal type annotation must reject it.
        """
        with pytest.raises(ValidationError):
            Settings(**VALID_OVERRIDES, log_level="VERBOSE")

    # --- URL trailing-slash stripping --------------------------------------

    def test_github_base_url_trailing_slash_stripped(self):
        """
        A developer might paste a URL with a trailing slash from their browser.
        The validator must silently remove it.
        """
        s = Settings(**VALID_OVERRIDES, github_api_base_url="https://api.github.com/")
        assert not s.github_api_base_url.endswith("/")
        assert s.github_api_base_url == "https://api.github.com"

    def test_llm_base_url_trailing_slash_stripped(self):
        s = Settings(**VALID_OVERRIDES, llm_api_base_url="https://api.openai.com/v1/")
        assert not s.llm_api_base_url.endswith("/")

    def test_multiple_trailing_slashes_all_stripped(self):
        s = Settings(**VALID_OVERRIDES, llm_api_base_url="https://api.openai.com/v1///")
        assert s.llm_api_base_url == "https://api.openai.com/v1"

    # --- SQLite-only enforcement -------------------------------------------

    def test_sqlite_url_accepted(self):
        """A valid SQLite URL must pass validation."""
        s = Settings(**VALID_OVERRIDES, database_url="sqlite+aiosqlite:///./test.db")
        assert s.database_url == "sqlite+aiosqlite:///./test.db"

    def test_in_memory_sqlite_url_accepted(self):
        """The in-memory SQLite URL used in tests must also be valid."""
        s = Settings(**VALID_OVERRIDES, database_url="sqlite+aiosqlite:///:memory:")
        assert ":memory:" in s.database_url

    def test_postgresql_url_rejected_with_clear_message(self):
        """
        A PostgreSQL URL is not supported in Phase 1.
        The error message must explain WHY, not just say "invalid".
        """
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                **VALID_OVERRIDES,
                database_url="postgresql+asyncpg://user:pass@localhost/db",
            )
        # The error should guide the developer, not just dump a traceback
        error_text = str(exc_info.value)
        assert "Phase 1 only supports SQLite" in error_text

    # --- max_retries bounds ------------------------------------------------

    @pytest.mark.parametrize("valid_value", [1, 3, 5, 10])
    def test_max_retries_valid_values_accepted(self, valid_value):
        s = Settings(**VALID_OVERRIDES, max_retries=valid_value)
        assert s.max_retries == valid_value

    def test_max_retries_zero_rejected(self):
        """ge=1 constraint: 0 is below the allowed minimum."""
        with pytest.raises(ValidationError):
            Settings(**VALID_OVERRIDES, max_retries=0)

    def test_max_retries_eleven_rejected(self):
        """le=10 constraint: 11 is above the allowed maximum."""
        with pytest.raises(ValidationError):
            Settings(**VALID_OVERRIDES, max_retries=11)


# ===========================================================================
# Tests — convenience properties
# ===========================================================================

class TestConvenienceProperties:
    """
    Properties add semantic meaning on top of raw field values.
    Test them separately from their underlying fields.
    """

    def test_log_level_int_matches_stdlib_constant(self):
        """
        log_level_int must return the numeric value that Python's logging
        module uses internally.  logging.DEBUG == 10, logging.INFO == 20, etc.
        """
        s = Settings(**VALID_OVERRIDES, log_level="DEBUG")
        assert s.log_level_int == logging.DEBUG

        s = Settings(**VALID_OVERRIDES, log_level="WARNING")
        assert s.log_level_int == logging.WARNING

    def test_is_debug_true_when_level_is_debug(self):
        s = Settings(**VALID_OVERRIDES, log_level="DEBUG")
        assert s.is_debug is True

    def test_is_debug_false_when_level_is_info(self):
        s = Settings(**VALID_OVERRIDES, log_level="INFO")
        assert s.is_debug is False

    @pytest.mark.parametrize("level", ["INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_is_debug_false_for_all_non_debug_levels(self, level):
        s = Settings(**VALID_OVERRIDES, log_level=level)
        assert s.is_debug is False


# ===========================================================================
# Tests — get_settings() singleton and cache behaviour
# ===========================================================================

class TestGetSettings:
    """
    get_settings() must return the same Settings instance every time it is
    called (singleton via @lru_cache) and must respect cache invalidation.
    """

    def test_returns_settings_instance(self, monkeypatch):
        """get_settings() must return a Settings object."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        get_settings.cache_clear()

        result = get_settings()
        assert isinstance(result, Settings)

        get_settings.cache_clear()  # clean up so other tests are unaffected

    def test_two_calls_return_same_object(self, monkeypatch):
        """
        @lru_cache guarantees identity (is), not just equality (==).
        Both references must point to the exact same object in memory.
        """
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        get_settings.cache_clear()

        first = get_settings()
        second = get_settings()
        assert first is second

        get_settings.cache_clear()

    def test_cache_clear_allows_new_instance_with_new_values(self, monkeypatch):
        """
        After cache_clear(), the next get_settings() call re-reads the
        environment.  This is the pattern used in tests that need to swap
        config values between test cases.
        """
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        get_settings.cache_clear()

        settings_a = get_settings()
        assert settings_a.llm_model == "gpt-4o-mini"

        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        get_settings.cache_clear()

        settings_b = get_settings()
        assert settings_b.llm_model == "gpt-4o"
        # They are different objects now
        assert settings_a is not settings_b

        get_settings.cache_clear()

