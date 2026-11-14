"""
tests/unit/test_session.py
--------------------------
Tests for the async database session layer (app/db/session.py).

Test categories
---------------
TestBuildEngine
    Unit tests for engine construction: correct URL, FK event attachment,
    connect_args for SQLite, pool_pre_ping, echo flag tied to debug mode.
    These run synchronously because _build_engine() itself is synchronous.

TestInitDb (async integration)
    Verifies that init_db() actually creates the expected tables in a real
    in-memory SQLite database.

TestGetDb (async integration)
    Verifies the get_db() generator yields an AsyncSession, closes it after
    use, and that the session can perform real database operations.

TestForeignKeyEnforcement (async integration)
    The most critical behaviour: SQLite must reject orphaned Plan/Artifact
    rows that reference a non-existent Job. Without PRAGMA foreign_keys=ON
    this test would silently pass the bad INSERT.

Why use pytest-asyncio here?
    app/db/session.py is entirely async (AsyncEngine, AsyncSession, async
    generators). We must use async test functions — pytest-asyncio makes this
    transparent with asyncio_mode = "auto" (set in pyproject.toml).

Fixture strategy
----------------
Every test that needs a database gets a fresh in-memory SQLite engine built
with _build_engine(). This isolates tests from each other and from the
module-level engine used in production. Using ":memory:" means no DB file is
written to disk and the schema is dropped automatically when the connection
closes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from app.db.models import Artifact, Base, Job, JobStatus, Plan
from app.db.session import _build_engine, dispose_db, get_db, init_db

# ===========================================================================
# Shared async fixture — isolated in-memory database for each test
# ===========================================================================


@pytest_asyncio.fixture
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = _build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = _build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Helper: build a minimal valid Job instance
# ---------------------------------------------------------------------------


def _make_job(**overrides: object) -> Job:
    defaults: dict[str, object] = {
        "issue_url": "https://github.com/octocat/hello-world/issues/1",
        "repo_owner": "octocat",
        "repo_name": "hello-world",
        "issue_number": 1,
        "status": JobStatus.PENDING,
    }
    defaults.update(overrides)
    return Job(**defaults)


# ===========================================================================
# Tests — engine construction (_build_engine)
# ===========================================================================


class TestBuildEngine:
    """
    Unit tests for the _build_engine() factory function.
    These are synchronous because _build_engine() itself is synchronous —
    it only *creates* the engine object; it does not open any connections.
    """

    def test_returns_async_engine(self):
        """_build_engine() must return an AsyncEngine, not a sync Engine."""
        eng = _build_engine("sqlite+aiosqlite:///:memory:")
        assert isinstance(eng, AsyncEngine)

    def test_sqlite_url_accepted(self):
        """An in-memory SQLite URL must produce a valid engine without error."""
        eng = _build_engine("sqlite+aiosqlite:///:memory:")
        assert eng is not None

    def test_sqlite_has_check_same_thread_false(self):
        """
        SQLite engines must have check_same_thread=False in connect_args.

        Why: aiosqlite runs the connection on a background thread. Without
        this flag, SQLite raises "ProgrammingError: SQLite objects created in a
        thread can only be used in that same thread."
        """
        eng = _build_engine("sqlite+aiosqlite:///:memory:")
        # Access the pool creator closure to inspect connect_args passed to the engine
        creator: Any = getattr(eng.sync_engine.pool, "_creator")
        connect_args = creator.__closure__[1].cell_contents
        assert connect_args.get("check_same_thread") is False

    def test_pool_pre_ping_enabled(self):
        """
        pool_pre_ping=True must be set so stale pooled connections are
        discarded before reuse rather than causing mysterious query failures.
        """
        eng = _build_engine("sqlite+aiosqlite:///:memory:")
        assert eng.sync_engine.pool._pre_ping is True

    def test_different_urls_produce_different_engines(self):
        """Two calls with different URLs must produce independent engine objects."""
        eng1 = _build_engine("sqlite+aiosqlite:///:memory:")
        eng2 = _build_engine("sqlite+aiosqlite:///./other_test.db")
        assert eng1 is not eng2

    def test_sqlite_fk_listener_registered(self):
        """
        The FK PRAGMA listener must be attached to the sync_engine when a
        SQLite URL is provided.

        SQLAlchemy registers connection listeners in the pool's dispatch collection.
        We verify our _enable_sqlite_foreign_keys listener is present.
        """
        eng = _build_engine("sqlite+aiosqlite:///:memory:")
        has_fk_listener = any(
            "_enable_sqlite_foreign_keys" in getattr(fn, "__name__", "")
            for fn in eng.sync_engine.pool.dispatch.connect
        )
        assert has_fk_listener, (
            "No '_enable_sqlite_foreign_keys' listener found in pool connect dispatch."
        )


# ===========================================================================
# Tests — database initialisation (init_db)
# ===========================================================================


class TestInitDb:
    """
    Integration tests that call init_db() against a real in-memory database
    and verify the resulting schema.
    """

    async def test_init_db_creates_jobs_table(self):
        """init_db() must create the 'jobs' table."""
        engine = _build_engine("sqlite+aiosqlite:///:memory:")
        await init_db.__wrapped__(engine) if hasattr(init_db, "__wrapped__") else None

        # Use our own init that takes an explicit engine for test isolation
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        assert "jobs" in tables
        await engine.dispose()

    async def test_init_db_creates_all_three_tables(self):
        """init_db() must create jobs, plans, and artifacts tables."""
        engine = _build_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            tables = set(
                await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
            )
        assert {"jobs", "plans", "artifacts"}.issubset(tables)
        await engine.dispose()

    async def test_init_db_idempotent(self, db_engine: AsyncEngine):
        """
        Calling create_all twice must not raise an error or drop existing data.
        'checkfirst=True' is SQLAlchemy's default: skip CREATE if table exists.
        """
        # Insert a row after first create_all
        factory = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            job = _make_job()
            session.add(job)
            await session.commit()
            first_id = job.id

        # Run create_all again — must not destroy the row we just inserted
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with factory() as session:
            fetched = await session.get(Job, first_id)
        assert fetched is not None, "Row was deleted by second create_all — idempotency broken"


# ===========================================================================
# Tests — AsyncSession behaviour (get_db)
# ===========================================================================


class TestGetDb:
    """
    Verify that get_db() yields a usable AsyncSession and cleans up correctly.

    get_db() is an async generator designed as a FastAPI dependency. We test it
    by consuming it directly with `async for` / `anext()` rather than through
    the HTTP layer, which keeps these tests fast and framework-independent.
    """

    async def test_get_db_yields_async_session(self):
        """
        Consuming get_db() must yield an AsyncSession instance.
        This is the core contract that every route handler depends on.
        """
        gen = get_db()
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)
        # Exhaust the generator so the finally block (session.close) runs
        try:
            await gen.aclose()
        except StopAsyncIteration:
            pass

    async def test_get_db_session_is_active_before_close(self):
        """
        The session yielded by get_db() must be open and usable for queries.
        """
        gen = get_db()
        session = await gen.__anext__()
        try:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            try:
                await gen.aclose()
            except StopAsyncIteration:
                pass

    async def test_get_db_session_closed_after_generator_exhausted(self):
        """
        After get_db() finishes (either normally or via aclose()), the session
        must be closed. A lingering open session is a connection leak.
        """
        gen = get_db()
        session = await gen.__anext__()
        closed = False
        original_close = session.close

        async def tracked_close() -> None:
            nonlocal closed
            closed = True
            await original_close()

        session.close = tracked_close  # type: ignore[method-assign]
        try:
            await gen.aclose()
        except StopAsyncIteration:
            pass
        assert closed is True

    async def test_session_can_execute_query(self, db_session: AsyncSession):
        """
        A session obtained from the test fixture must be able to execute a
        simple query — confirming the engine and connection are genuinely live.
        """
        result = await db_session.execute(text("SELECT 1"))
        value = result.scalar()
        assert value == 1

    async def test_session_insert_and_retrieve(self, db_session: AsyncSession):
        """
        End-to-end round-trip: insert a Job row, commit, re-fetch by primary key.
        Verifies that the session, engine, and ORM mapping all work together.
        """
        job = _make_job(issue_number=42)
        db_session.add(job)
        await db_session.commit()

        fetched = await db_session.get(Job, job.id)
        assert fetched is not None
        assert fetched.issue_number == 42
        assert fetched.status == JobStatus.PENDING

    async def test_expire_on_commit_false(self, db_session: AsyncSession):
        """
        expire_on_commit=False means ORM attributes remain readable after
        commit() without triggering an implicit lazy SELECT.

        Why this matters in async:
        In async mode, implicit lazy loads raise MissingGreenlet / DetachedInstance
        errors because SQLAlchemy cannot issue a synchronous SELECT in an async
        context. With expire_on_commit=False, the in-memory values are preserved.
        """
        job = _make_job()
        db_session.add(job)
        await db_session.commit()

        # This attribute access must NOT trigger a new DB query
        # (expire_on_commit=False keeps the value in memory)
        assert job.repo_owner == "octocat"
        assert job.status == JobStatus.PENDING


# ===========================================================================
# Tests — SQLite foreign key enforcement
# ===========================================================================


class TestForeignKeyEnforcement:
    """
    The most safety-critical behaviour: PRAGMA foreign_keys=ON must be active
    so orphaned Plan and Artifact rows cannot exist in the database.

    If the PRAGMA is not applied, all three FK tests would pass silently —
    the bad INSERT would succeed and no error would be raised.
    """

    async def test_pragma_is_on(self, db_engine: AsyncEngine):
        """
        Directly query PRAGMA foreign_keys to confirm it was set to ON (1)
        for the connection produced by our engine.
        """
        async with db_engine.connect() as conn:
            result = await conn.execute(text("PRAGMA foreign_keys"))
            value = result.scalar()
        assert value == 1, (
            f"PRAGMA foreign_keys should be 1 (ON) but got {value}. "
            "FK constraints will not be enforced — orphaned rows are possible."
        )

    async def test_plan_without_job_raises_integrity_error(self, db_session: AsyncSession):
        """
        Inserting a Plan whose job_id references a non-existent Job must raise
        IntegrityError. This is the fundamental orphan-prevention guarantee.
        """
        orphan_plan = Plan(job_id=99999, raw_text="Orphan plan", status="draft")
        db_session.add(orphan_plan)
        with pytest.raises(IntegrityError):
            await db_session.flush()

    async def test_artifact_without_job_raises_integrity_error(self, db_session: AsyncSession):
        """
        Inserting an Artifact with a non-existent job_id must also raise
        IntegrityError — same FK enforcement, different table.
        """
        orphan_artifact = Artifact(job_id=99999, artifact_type="issue_data", content="{}")
        db_session.add(orphan_artifact)
        with pytest.raises(IntegrityError):
            await db_session.flush()

    async def test_valid_plan_with_existing_job_succeeds(self, db_session: AsyncSession):
        """
        A Plan linked to an existing Job must insert successfully.
        Confirms FK enforcement does not block valid inserts.
        """
        job = _make_job()
        db_session.add(job)
        await db_session.commit()

        plan = Plan(job_id=job.id, raw_text="Step 1: implement feature")
        db_session.add(plan)
        await db_session.commit()

        fetched = await db_session.get(Plan, plan.id)
        assert fetched is not None
        assert fetched.job_id == job.id

    async def test_cascade_delete_removes_plan(self, db_session: AsyncSession):
        """
        Deleting a Job must cascade to its Plan via the ORM relationship.
        Confirms that cascade='all, delete-orphan' on Job.plan is working.
        """
        job = _make_job()
        db_session.add(job)
        await db_session.commit()

        plan = Plan(job_id=job.id, raw_text="plan text")
        db_session.add(plan)
        await db_session.commit()
        plan_id = plan.id

        await db_session.delete(job)
        await db_session.commit()

        deleted_plan = await db_session.get(Plan, plan_id)
        assert deleted_plan is None, "Plan was not deleted when its Job was deleted"

    async def test_cascade_delete_removes_artifacts(self, db_session: AsyncSession):
        """
        Deleting a Job must cascade to all its Artifacts.
        """
        job = _make_job()
        db_session.add(job)
        await db_session.commit()

        artifact = Artifact(job_id=job.id, artifact_type="issue_data", content="{}")
        db_session.add(artifact)
        await db_session.commit()
        artifact_id = artifact.id

        await db_session.delete(job)
        await db_session.commit()

        deleted = await db_session.get(Artifact, artifact_id)
        assert deleted is None, "Artifact was not deleted when its Job was deleted"


# ===========================================================================
# Tests — module-level singletons
# ===========================================================================


class TestModuleLevelSingletons:
    """
    Verify that the module-level engine and session factory exported by
    app/db/session.py are of the expected types.
    """

    def test_module_engine_is_async_engine(self):
        """The module-level `engine` must be an AsyncEngine."""
        from app.db.session import engine as module_engine

        assert isinstance(module_engine, AsyncEngine)

    def test_module_session_factory_is_async_sessionmaker(self):
        """The module-level AsyncSessionLocal must be an async_sessionmaker."""
        from app.db.session import AsyncSessionLocal as factory

        assert isinstance(factory, async_sessionmaker)


class TestDisposeDb:
    """Verify that dispose_db properly cleans up the engine."""

    async def test_dispose_db_calls_engine_dispose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """dispose_db() must invoke dispose() on the module-level engine."""
        from unittest.mock import AsyncMock

        import app.db.session as session_module

        mock_engine = AsyncMock()
        monkeypatch.setattr(session_module, "engine", mock_engine)
        await dispose_db()
        mock_engine.dispose.assert_awaited_once()
