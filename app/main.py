"""
app/main.py
-----------
FastAPI application foundation, lifespan management, and API router assembly
for the Agentic Software Development Workflow.

Architecture decisions
----------------------
1. Lifespan context manager:
   Uses the modern async context manager pattern (@asynccontextmanager)
   recommended by FastAPI / Starlette, replacing legacy @app.on_event("startup")
   and @app.on_event("shutdown").
   - Startup:
     Ensures runtime workspace directory exists and initializes database tables
     via init_db().
   - Shutdown:
     Disposes of the database engine and connection pool via dispose_db().

2. Separation of concerns:
   main.py contains zero business logic and zero database queries.
   It acts strictly as an application assembly root and lifespan coordinator.

3. Configuration integration:
   Reuses get_settings() from app.core.config for application metadata and logging.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.core.config import get_settings
from app.db.session import dispose_db, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown lifecycle events.

    Startup:
    - Verifies / creates configured workspace directory.
    - Initializes persistent database tables via init_db().

    Shutdown:
    - Disposes of database engine connections via dispose_db().
    Guaranteed by try/finally even if init_db() or application runtime raises an error.
    """
    settings = get_settings()
    logger.info("Application starting up...")

    # Ensure workspace directory exists
    settings.workspace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Workspace directory verified at %s", settings.workspace_dir)

    try:
        # Initialize database tables
        logger.info("Initializing database tables...")
        await init_db()

        # Recover interrupted jobs on startup (fail-safe if tables not yet created)
        try:
            from app.adapters.github import GitHubClient
            from app.db.session import AsyncSessionLocal
            from app.orchestrator.workflow import JobOrchestrator

            github_client = GitHubClient(
                token=settings.github_token,
                base_url=settings.github_api_base_url,
            )
            orchestrator = JobOrchestrator(
                session_factory=AsyncSessionLocal,
                github_client=github_client,
            )
            recovered = await orchestrator.recover_interrupted_jobs()
            if recovered:
                logger.info("Startup recovery processed %d interrupted jobs.", recovered)
        except Exception as exc:
            logger.debug("Startup job recovery skipped: %s", exc)

        yield
    finally:
        # Shutdown: always attempt database disposal
        logger.info("Application shutting down...")
        await dispose_db()
        logger.info("Database connections disposed.")


def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    application = FastAPI(
        title="Agentic Software Development Workflow",
        description=(
            "An AI-powered developer assistant that manages autonomous workflows to "
            "resolve GitHub issues and produce Pull Requests."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Register API routers
    application.include_router(health_router)
    application.include_router(jobs_router)

    return application


app: FastAPI = create_app()
