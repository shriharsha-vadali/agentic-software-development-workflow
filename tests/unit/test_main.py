"""
tests/unit/test_main.py
-----------------------
Unit tests for the FastAPI application foundation, lifespan lifecycle
management, and health-check endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.health import HealthResponse
from app.main import app, create_app, lifespan


class TestHealthEndpoint:
    """Verify behaviour, status codes, and payload safety of GET /health."""

    def test_health_check_returns_200_and_healthy(self, mocker) -> None:
        """GET /health returns HTTP 200 with predictable status payload."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert data == {"status": "healthy"}

        # Validate against schema
        validated = HealthResponse.model_validate(data)
        assert validated.status == "healthy"

    def test_health_check_does_not_leak_internal_details(self, mocker) -> None:
        """Health check response contains only public status without credentials or environment."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        with TestClient(app) as client:
            response = client.get("/health")

        data = response.json()
        sensitive_keys = {
            "token",
            "key",
            "password",
            "secret",
            "database_url",
            "github_token",
            "llm_api_key",
        }
        for key in data:
            assert key.lower() not in sensitive_keys

    def test_health_check_method_not_allowed(self, mocker) -> None:
        """POST /health is rejected with 405 Method Not Allowed."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        with TestClient(app) as client:
            response = client.post("/health")

        assert response.status_code == 405


class TestApplicationLifespan:
    """Verify startup and shutdown lifecycle management."""

    async def test_lifespan_invokes_init_db_and_dispose_db(self, mocker) -> None:
        """Startup calls init_db; shutdown calls dispose_db."""
        mock_init = mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mock_dispose = mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        mock_init.assert_not_awaited()
        mock_dispose.assert_not_awaited()

        async with lifespan(app):
            mock_init.assert_awaited_once()
            mock_dispose.assert_not_awaited()

        mock_dispose.assert_awaited_once()

    def test_lifespan_executes_within_testclient_context(self, mocker) -> None:
        """Entering TestClient context triggers startup; exiting triggers shutdown."""
        mock_init = mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mock_dispose = mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        with TestClient(app):
            mock_init.assert_awaited_once()
            mock_dispose.assert_not_awaited()

        mock_dispose.assert_awaited_once()

    async def test_lifespan_ensures_workspace_directory_exists(
        self, mocker, tmp_path: Path
    ) -> None:
        """Startup creates workspace_dir if it does not yet exist."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        custom_workspace = tmp_path / "custom_workspaces"
        assert not custom_workspace.exists()

        mock_settings = mocker.MagicMock()
        mock_settings.workspace_dir = custom_workspace
        mocker.patch("app.main.get_settings", return_value=mock_settings)

        async with lifespan(app):
            assert custom_workspace.exists()
            assert custom_workspace.is_dir()

    async def test_lifespan_disposes_db_on_app_runtime_exception(self, mocker) -> None:
        """dispose_db executes even when an exception is raised during application runtime."""
        mocker.patch("app.main.init_db", new_callable=AsyncMock)
        mock_dispose = mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        class CustomRuntimeError(Exception):
            pass

        with pytest.raises(CustomRuntimeError, match="Crash during runtime"):
            async with lifespan(app):
                raise CustomRuntimeError("Crash during runtime")

        mock_dispose.assert_awaited_once()

    async def test_lifespan_disposes_db_on_init_db_failure_and_propagates(self, mocker) -> None:
        """dispose_db is attempted and original exception is propagated if init_db fails."""

        class CustomDatabaseError(Exception):
            pass

        mock_init = mocker.patch(
            "app.main.init_db",
            new_callable=AsyncMock,
            side_effect=CustomDatabaseError("Database initialization failed"),
        )
        mock_dispose = mocker.patch("app.main.dispose_db", new_callable=AsyncMock)

        with pytest.raises(CustomDatabaseError, match="Database initialization failed"):
            async with lifespan(app):
                pass

        mock_init.assert_awaited_once()
        mock_dispose.assert_awaited_once()


class TestApplicationMetadata:
    """Verify application configuration and factory initialization."""

    def test_app_metadata_configuration(self) -> None:
        """FastAPI application has expected title, version, and documentation paths."""
        test_app = create_app()
        assert test_app.title == "Agentic Software Development Workflow"
        assert test_app.version == "0.1.0"
        assert test_app.docs_url == "/docs"
        assert test_app.redoc_url == "/redoc"

    def test_app_instantiation_does_not_trigger_database_io(self, mocker) -> None:
        """Creating the app instance does not call init_db outside the lifespan."""
        mock_init = mocker.patch("app.main.init_db", new_callable=AsyncMock)
        _ = create_app()
        mock_init.assert_not_called()
