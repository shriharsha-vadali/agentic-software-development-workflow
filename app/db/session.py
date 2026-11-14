"""
app/db/session.py
-----------------
Async database engine, session factory, and FastAPI dependency for the
Agentic Software Development Workflow.

This module is the **only** place in the application that knows how to open a
database connection. Every other module that needs database access receives a
session through FastAPI's dependency injection system (get_db) or by calling
the session factory directly in background tasks.

Architecture decisions
----------------------

1. Why async?
   FastAPI is built on asyncio (via Starlette). Every HTTP request handler is
   a coroutine. If we used a *synchronous* SQLAlchemy session inside an async
   handler, the DB query would block the entire event loop — no other requests
   could be served while that query runs. The async engine + AsyncSession
   combination lets SQLAlchemy hand control back to asyncio during I/O, so the
   event loop stays free.

2. Why a module-level engine + factory (not per-request)?
   Creating an engine is expensive: it establishes a connection pool, reads
   config, etc. The engine is created *once* at module import time and reused
   across the lifetime of the process. The session factory (async_sessionmaker)
   is a lightweight factory object; creating it once is also the right pattern.

3. Why yield in get_db?
   FastAPI's dependency injection treats `yield` as a split between setup and
   teardown. The code before `yield` runs before the route handler; the code
   after `yield` (in the finally block) always runs after the handler returns,
   even if it raised an exception. This guarantees the session is always closed.

4. Why enable SQLite foreign keys explicitly?
   SQLite disables foreign key constraints by default (PRAGMA foreign_keys=0)
   for backward compatibility. Without enabling it, inserting a Plan row that
   references a non-existent Job would succeed silently, allowing orphaned data.
   We enable it on every new connection via an event listener so the behaviour
   matches what PostgreSQL would give us in production.

5. Why is_sqlite check?
   The FK pragma is SQLite-specific. If the application is ever pointed at
   PostgreSQL (by changing DATABASE_URL), the event listener must not run —
   executing a SQLite PRAGMA against PostgreSQL raises an error. The check
   makes the codebase safe to migrate.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)


# =============================================================================
# Engine
# =============================================================================


def _build_engine(database_url: str | None = None) -> AsyncEngine:
    """
    Create and configure an async SQLAlchemy engine.

    Parameters
    ----------
    database_url:
        Override the URL from settings. Used in tests to inject an
        in-memory database without touching environment variables.
        When None, the URL is read from get_settings().database_url.

    Engine options explained
    ------------------------
    echo=False
        When True, SQLAlchemy logs every SQL statement to the standard
        logger. We tie this to settings.is_debug so SQL logging is
        controlled by the LOG_LEVEL env var — no code changes needed.

    connect_args={"check_same_thread": False}
        SQLite only. By default SQLite raises an error when a connection
        is used from a thread other than the one that created it. asyncio
        moves work between threads via its thread pool, so we must disable
        this check. The async driver (aiosqlite) handles thread safety
        at a higher level.

    pool_pre_ping=True
        Before reusing a pooled connection, SQLAlchemy sends a lightweight
        "ping" query. If the connection has gone stale (the DB file was
        deleted, or a timeout occurred), it discards the connection and
        opens a fresh one. Prevents mysterious "connection closed" errors
        after idle periods.
    """
    url = database_url or get_settings().database_url
    is_debug = get_settings().is_debug

    engine_kwargs: dict[str, Any] = {
        "echo": is_debug,
        "pool_pre_ping": True,
    }

    # SQLite-specific: disable the same-thread check because aiosqlite runs
    # the connection on a background thread managed by asyncio, not the thread
    # that called create_async_engine.
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_async_engine(url, **engine_kwargs)

    # -------------------------------------------------------------------------
    # SQLite foreign key enforcement
    # -------------------------------------------------------------------------
    # This event fires on every new raw DBAPI connection that the pool creates.
    # We use the synchronous "connect" event (not the async one) because DBAPI
    # connections are always synchronous at the lowest level — aiosqlite wraps
    # them in a thread, but the PRAGMA itself is a synchronous operation.
    if url.startswith("sqlite"):

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
            """
            Enable FK constraints for every new SQLite connection.

            Why sync_engine?
            create_async_engine wraps a standard Engine inside an AsyncEngine.
            SQLAlchemy's event system works on the underlying sync Engine;
            we access it via .sync_engine.

            Why PRAGMA in the event instead of in init_db?
            The PRAGMA must be run on every *connection*, not once per session.
            When connections are pooled, a recycled connection has forgotten the
            PRAGMA set by its previous user. The event listener fires for each
            new DBAPI connection, so it cannot be missed.
            """
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
            logger.debug("SQLite PRAGMA foreign_keys=ON applied to new connection")

    logger.info("Database engine created: %s", url.split("///")[0])  # log scheme only
    return engine


# Module-level engine — created once when this module is first imported.
# Tests that need a different database URL must create their own engine using
# _build_engine(url) and not rely on this module-level instance.
engine: AsyncEngine = _build_engine()


# =============================================================================
# Session factory
# =============================================================================

# async_sessionmaker is the SQLAlchemy 2.x replacement for the older
# sessionmaker + AsyncSession combination. It is a factory object that
# produces AsyncSession instances pre-configured with our engine.
#
# expire_on_commit=False
#   After commit(), SQLAlchemy normally marks every attribute on every ORM
#   object as "expired", meaning the next access triggers a fresh SELECT.
#   In an async context this is dangerous: if a route handler accesses an
#   expired attribute after committing, SQLAlchemy tries to issue an implicit
#   lazy SELECT — which does not work in async mode and raises an error.
#   Setting expire_on_commit=False keeps attribute values in memory after
#   commit, so callers can safely read them without an open session.
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# =============================================================================
# Database initialisation
# =============================================================================


async def init_db() -> None:
    """
    Create all database tables defined in app/db/models.py.

    Called once during application startup (registered in the FastAPI
    lifespan context manager in app/main.py).

    How it works
    ------------
    Base.metadata.create_all() compares the ORM model definitions against
    the actual database schema. For each table that does not exist, it issues
    a CREATE TABLE statement. Tables that already exist are left untouched
    (checkfirst=True is the default).

    Important: this is NOT a migration tool. It cannot alter existing
    columns, rename tables, or drop columns. For schema migrations, Phase 3
    will introduce Alembic. For now, dropping and recreating the database
    file is the upgrade path during development.

    Why run_sync?
    Base.metadata.create_all() is a synchronous SQLAlchemy operation (it
    was written before async support existed). To run it on an async engine
    we must use conn.run_sync(), which executes the synchronous function in
    the same thread pool that aiosqlite uses — keeping the event loop free.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified / created")


async def dispose_db() -> None:
    """
    Close and dispose of the module-level database engine.

    Called during application shutdown (registered in the FastAPI
    lifespan context manager in app/main.py) to gracefully terminate
    all pooled connections and release resources.
    """
    await engine.dispose()
    logger.info("Database engine disposed")


# =============================================================================
# FastAPI dependency
# =============================================================================


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields one AsyncSession per HTTP request.

    Usage in a route handler
    ------------------------
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.db.session import get_db

        @router.get("/jobs/{job_id}")
        async def get_job(job_id: int, db: AsyncSession = Depends(get_db)):
            result = await db.get(Job, job_id)
            ...

    Lifecycle
    ---------
    1. FastAPI calls get_db() before invoking the route handler.
    2. A new AsyncSession is created from AsyncSessionLocal.
    3. The session is yielded into the route handler as `db`.
    4. Upon context exit, AsyncSession.__aexit__ automatically closes
       the session, releasing its connection back to the pool.

    Why not commit inside get_db?
    The service layer is responsible for committing transactions at the
    appropriate business logic boundary. get_db() only manages the session
    lifecycle (open / close), not the transaction boundary (commit / rollback).
    This keeps session management and business logic cleanly separated.

    Error handling
    --------------
    If the route handler raises an exception, FastAPI propagates it after
    __aexit__ runs. The session is closed (and any uncommitted transaction
    is implicitly rolled back by SQLAlchemy) before the exception reaches
    FastAPI's error handler.
    """
    async with AsyncSessionLocal() as session:
        yield session
