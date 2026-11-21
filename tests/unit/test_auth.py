"""
tests/unit/test_auth.py
-----------------------
Unit and integration tests for API authentication, constant-time key comparison,
FastAPI security dependencies, and route protection gates (app/core/auth.py).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.auth import constant_time_compare, get_current_api_key, verify_api_key
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.main import app

# =============================================================================
# Unit Tests — Constant-Time Comparison & Key Verification
# =============================================================================


class TestConstantTimeCompare:
    """Verify constant-time string comparison logic."""

    def test_identical_strings_return_true(self) -> None:
        assert constant_time_compare("my-api-key-123", "my-api-key-123") is True

    def test_different_strings_return_false(self) -> None:
        assert constant_time_compare("my-api-key-123", "wrong-api-key-456") is False

    def test_different_lengths_return_false(self) -> None:
        assert constant_time_compare("short", "much_longer_key_string") is False

    def test_empty_strings(self) -> None:
        assert constant_time_compare("", "") is True
        assert constant_time_compare("non-empty", "") is False


class TestVerifyApiKey:
    """Verify verify_api_key behavior with SecretStr inputs and fail-closed edge cases."""

    def test_matching_key_returns_true(self) -> None:
        expected = SecretStr("super-secret-key-42")
        assert verify_api_key("super-secret-key-42", expected) is True

    def test_mismatched_key_returns_false(self) -> None:
        expected = SecretStr("super-secret-key-42")
        assert verify_api_key("wrong-key-attempt", expected) is False

    def test_none_provided_key_returns_false(self) -> None:
        expected = SecretStr("super-secret-key-42")
        assert verify_api_key(None, expected) is False

    def test_empty_provided_key_returns_false(self) -> None:
        expected = SecretStr("super-secret-key-42")
        assert verify_api_key("", expected) is False
        assert verify_api_key("   ", expected) is False

    def test_none_expected_key_returns_false(self) -> None:
        assert verify_api_key("super-secret-key-42", None) is False

    def test_empty_expected_secret_returns_false(self) -> None:
        expected = SecretStr("   ")
        assert verify_api_key("super-secret-key-42", expected) is False


# =============================================================================
# Unit Tests — FastAPI Security Dependency (get_current_api_key)
# =============================================================================


class TestGetCurrentApiKeyDependency:
    """Direct coroutine tests for get_current_api_key dependency."""

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_request_without_key(self) -> None:
        settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=False,
            api_key=None,
        )
        res = await get_current_api_key(api_key_header=None, bearer_creds=None, settings=settings)
        assert res is None

    @pytest.mark.asyncio
    async def test_auth_enabled_valid_header_key_passes(self) -> None:
        settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key="secret-api-key-100",
        )
        res = await get_current_api_key(
            api_key_header="secret-api-key-100",
            bearer_creds=None,
            settings=settings,
        )
        assert res == "secret-api-key-100"

    @pytest.mark.asyncio
    async def test_auth_enabled_valid_bearer_token_passes(self) -> None:
        from fastapi.security import HTTPAuthorizationCredentials

        settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key="secret-api-key-100",
        )
        bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-api-key-100")
        res = await get_current_api_key(
            api_key_header=None,
            bearer_creds=bearer,
            settings=settings,
        )
        assert res == "secret-api-key-100"

    @pytest.mark.asyncio
    async def test_auth_enabled_missing_key_raises_401(self) -> None:
        settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key="secret-api-key-100",
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_current_api_key(api_key_header=None, bearer_creds=None, settings=settings)
        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail
        assert exc_info.value.headers == {"WWW-Authenticate": "ApiKey"}

    @pytest.mark.asyncio
    async def test_auth_enabled_invalid_key_raises_401(self) -> None:
        settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key="secret-api-key-100",
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_current_api_key(
                api_key_header="bad-key", bearer_creds=None, settings=settings
            )
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail
        assert exc_info.value.headers == {"WWW-Authenticate": "ApiKey"}


# =============================================================================
# Integration Tests — HTTP Route Protection & Endpoint Behavior
# =============================================================================


class TestApiRouteAuthentication:
    """Verify route protection across the FastAPI application with TestClient."""

    @pytest.fixture
    def mock_db_session(self):
        session = AsyncMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one.return_value = 0
        session.execute.return_value = mock_result
        return session

    def test_health_endpoint_is_always_public_without_auth(self, mocker) -> None:
        """GET /health probe is always unauthenticated even when api_auth_enabled is True."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        test_settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key="prod-secret-key-12345",
        )
        app.dependency_overrides[get_settings] = lambda: test_settings

        with TestClient(app) as client:
            res = client.get("/health")
            assert res.status_code == 200
            assert res.json() == {"status": "healthy"}

        app.dependency_overrides.clear()

    def test_jobs_endpoint_accessible_when_auth_disabled(self, mocker, mock_db_session) -> None:
        """When api_auth_enabled is False, /jobs endpoints are accessible without auth."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        test_settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=False,
            api_key=None,
        )
        app.dependency_overrides[get_settings] = lambda: test_settings
        app.dependency_overrides[get_db] = lambda: mock_db_session

        with TestClient(app) as client:
            res = client.get("/jobs")
            assert res.status_code == 200

        app.dependency_overrides.clear()

    def test_jobs_endpoint_requires_auth_when_enabled(self, mocker, mock_db_session) -> None:
        """When api_auth_enabled is True, missing credentials return HTTP 401."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        test_settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key="secret-api-key-test-value",
        )
        app.dependency_overrides[get_settings] = lambda: test_settings
        app.dependency_overrides[get_db] = lambda: mock_db_session

        with TestClient(app) as client:
            # 1. No credentials -> 401
            res_no_auth = client.get("/jobs")
            assert res_no_auth.status_code == 401
            assert "Missing API key" in res_no_auth.json()["detail"]
            assert res_no_auth.headers.get("www-authenticate") == "ApiKey"

            # 2. Invalid X-API-Key -> 401
            res_bad_key = client.get("/jobs", headers={"X-API-Key": "wrong-key"})
            assert res_bad_key.status_code == 401
            assert "Invalid API key" in res_bad_key.json()["detail"]

            # 3. Valid X-API-Key -> 200
            res_valid_header = client.get(
                "/jobs", headers={"X-API-Key": "secret-api-key-test-value"}
            )
            assert res_valid_header.status_code == 200

            # 4. Valid Authorization: Bearer <key> -> 200
            res_valid_bearer = client.get(
                "/jobs",
                headers={"Authorization": "Bearer secret-api-key-test-value"},
            )
            assert res_valid_bearer.status_code == 200

            # 5. Invalid Authorization: Bearer <key> -> 401
            res_bad_bearer = client.get(
                "/jobs",
                headers={"Authorization": "Bearer invalid-token"},
            )
            assert res_bad_bearer.status_code == 401
            assert "Invalid API key" in res_bad_bearer.json()["detail"]

        app.dependency_overrides.clear()

    def test_auth_error_never_exposes_configured_key_in_response(self, mocker) -> None:
        """Verify that 401 error response does not leak the secret API key."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        secret_key_value = "super-secret-unique-key-9999"
        test_settings = Settings(
            github_token="ghp_test",
            llm_api_key="sk-test",
            api_auth_enabled=True,
            api_key=secret_key_value,
        )
        app.dependency_overrides[get_settings] = lambda: test_settings

        with TestClient(app) as client:
            res = client.get("/jobs", headers={"X-API-Key": "attacker-guess"})
            assert res.status_code == 401
            response_text = res.text
            assert secret_key_value not in response_text
            assert "attacker-guess" not in response_text

        app.dependency_overrides.clear()
