"""
tests/unit/test_models.py
--------------------------
Tests for the SQLAlchemy ORM models defined in app/db/models.py.

Test structure
--------------
Unit tests (TestJobStatus, TestModelImports, TestModelRepr)
    Pure Python — no database connection needed.
    Fast, run in milliseconds.

Integration tests (TestJobTable, TestPlanTable, TestArtifactTable,
                   TestRelationships, TestCascadeDelete)
    Spin up a real in-memory SQLite database using SQLAlchemy's sync API.
    Each test class gets a fresh database via the `db` fixture so tests
    are fully isolated — no row from test A can affect test B.

Why use the synchronous SQLAlchemy engine in model tests?
    app/db/models.py defines model classes; it has no async code.
    The session module (Task 4) adds the async engine. Testing models
    with the synchronous engine keeps this test file simple and free of
    asyncio boilerplate. The async behaviour is tested in test_session.py.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Artifact, Base, Job, JobStatus, Plan


# ===========================================================================
# Shared fixture — fresh in-memory database for every test class
# ===========================================================================

@pytest.fixture()
def db() -> Session:
    """
    Yield a synchronous SQLAlchemy Session backed by an in-memory SQLite DB.

    Why in-memory?
    - No files written to disk → tests are hermetic and fast.
    - The database is destroyed the moment the connection closes, so test
      isolation is guaranteed at the OS level, not just at the ORM level.

    Why synchronous here?
    - Models are plain Python classes with no async code.
    - Using the sync API avoids asyncio boilerplate in every test.
    - Task 4 (session.py) adds the async engine; its tests use asyncio.

    Foreign key enforcement
    -----------------------
    SQLite disables foreign key constraints by default (PRAGMA foreign_keys=0).
    We enable them with an engine-level event listener so that our FK
    integrity tests actually catch violations, matching production behaviour.
    """
    from sqlalchemy import event as sa_event

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # Enable FK enforcement for every connection produced by this engine.
    # Without this, INSERT of a Plan with a non-existent job_id succeeds
    # silently — a dangerous SQLite default.
    @sa_event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record):  # type: ignore[misc]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    session = SessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


# ---------------------------------------------------------------------------
# Helper: minimal valid Job kwargs (all required, non-nullable columns)
# ---------------------------------------------------------------------------

def make_job(**overrides: object) -> Job:
    """Return a Job instance with sensible defaults, allowing field overrides."""
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
# Tests — JobStatus enum
# ===========================================================================

class TestJobStatus:
    """
    JobStatus is the backbone of the agent state machine.
    Test every behavioural guarantee of the enum.
    """

    def test_all_expected_members_exist(self):
        """Every state defined in the Phase 1 plan must be present."""
        expected = {
            "PENDING", "FETCHING_ISSUE", "INDEXING_CODEBASE",
            "PLANNING", "AWAITING_APPROVAL", "GENERATING_CODE",
            "RUNNING_TESTS", "OPENING_PR", "COMPLETED", "FAILED",
        }
        actual = {m.name for m in JobStatus}
        assert actual == expected

    def test_member_value_is_a_string(self):
        """
        JobStatus inherits from str, so each member's value is a plain string.
        This matters for JSON serialisation — FastAPI returns "PENDING", not
        an object.
        """
        assert isinstance(JobStatus.PENDING.value, str)
        assert JobStatus.PENDING.value == "PENDING"

    def test_member_is_its_own_value(self):
        """
        Because JobStatus(str, Enum) makes the member itself a str subclass,
        comparing a member to its string value must be True.
        """
        assert JobStatus.PLANNING == "PLANNING"

    def test_construct_from_string(self):
        """JobStatus('PENDING') must return the PENDING member."""
        assert JobStatus("PENDING") is JobStatus.PENDING

    def test_invalid_value_raises_value_error(self):
        """Constructing from an unrecognised string must raise ValueError."""
        with pytest.raises(ValueError):
            JobStatus("TYPO")

    def test_pending_is_first_terminal_states_last(self):
        """
        Ordering sanity check: PENDING comes before terminal states.
        (enum members preserve definition order in Python 3.11+)
        """
        members = list(JobStatus)
        assert members[0] == JobStatus.PENDING
        assert JobStatus.COMPLETED in members
        assert JobStatus.FAILED in members


# ===========================================================================
# Tests — model imports and class structure
# ===========================================================================

class TestModelImports:
    """Verify that all public names are importable and of the correct type."""

    def test_base_is_declarative_base(self):
        from sqlalchemy.orm import DeclarativeBase
        assert issubclass(Base, DeclarativeBase)

    def test_job_has_correct_tablename(self):
        assert Job.__tablename__ == "jobs"

    def test_plan_has_correct_tablename(self):
        assert Plan.__tablename__ == "plans"

    def test_artifact_has_correct_tablename(self):
        assert Artifact.__tablename__ == "artifacts"

    def test_base_metadata_contains_all_tables(self):
        """Base.metadata must know about all three tables."""
        table_names = set(Base.metadata.tables.keys())
        assert {"jobs", "plans", "artifacts"}.issubset(table_names)


# ===========================================================================
# Tests — __repr__ methods
# ===========================================================================

class TestModelRepr:
    """__repr__ should produce a readable string without raising exceptions."""

    def test_job_repr_contains_key_info(self):
        job = make_job()
        r = repr(job)
        assert "Job" in r
        assert "PENDING" in r

    def test_plan_repr_contains_key_info(self):
        plan = Plan(job_id=1, raw_text="Step 1: do X", status="draft")
        r = repr(plan)
        assert "Plan" in r
        assert "draft" in r

    def test_artifact_repr_contains_key_info(self):
        artifact = Artifact(job_id=1, artifact_type="issue_data", content="{}")
        r = repr(artifact)
        assert "Artifact" in r
        assert "issue_data" in r


# ===========================================================================
# Integration tests — Job table
# ===========================================================================

class TestJobTable:
    """
    Integration tests that write and read Job rows through a real SQLite DB.
    Each test method receives a fresh `db` session via the fixture.
    """

    def test_create_and_retrieve_job(self, db: Session):
        """A Job inserted and committed must be retrievable with all fields."""
        job = make_job()
        db.add(job)
        db.commit()

        fetched = db.get(Job, job.id)
        assert fetched is not None
        assert fetched.issue_url == "https://github.com/octocat/hello-world/issues/1"
        assert fetched.repo_owner == "octocat"
        assert fetched.repo_name == "hello-world"
        assert fetched.issue_number == 1
        assert fetched.status == JobStatus.PENDING

    def test_status_stored_as_string_in_db(self, db: Session):
        """
        The DB column stores the string value "PENDING", not a Python enum.
        Verify by querying with raw SQL.
        """
        job = make_job()
        db.add(job)
        db.commit()

        row = db.execute(
            text("SELECT status FROM jobs WHERE id = :id"), {"id": job.id}
        ).fetchone()
        assert row is not None
        assert row[0] == "PENDING"

    def test_created_at_auto_populated(self, db: Session):
        """created_at must be set automatically by the database on INSERT."""
        job = make_job()
        db.add(job)
        db.commit()
        db.refresh(job)

        assert job.created_at is not None
        assert isinstance(job.created_at, datetime)

    def test_workspace_path_defaults_to_none(self, db: Session):
        """workspace_path is nullable and must default to None."""
        job = make_job()
        db.add(job)
        db.commit()

        assert job.workspace_path is None

    def test_workspace_path_can_be_set(self, db: Session):
        """workspace_path must accept a string path after initial insert."""
        job = make_job()
        db.add(job)
        db.commit()

        job.workspace_path = "/tmp/workspaces/octocat__hello-world__1"
        db.commit()
        db.refresh(job)

        assert job.workspace_path == "/tmp/workspaces/octocat__hello-world__1"

    def test_error_message_defaults_to_none(self, db: Session):
        """error_message is nullable and must default to None."""
        job = make_job()
        db.add(job)
        db.commit()

        assert job.error_message is None

    def test_status_transition_persisted(self, db: Session):
        """Changing status and committing must persist the new value."""
        job = make_job()
        db.add(job)
        db.commit()

        job.status = JobStatus.FETCHING_ISSUE
        db.commit()
        db.refresh(job)

        assert job.status == JobStatus.FETCHING_ISSUE

    def test_invalid_status_raises_error(self, db: Session):
        """
        SQLite has no native ENUM type — it stores the value as VARCHAR and
        cannot enforce valid members at the database level.  Validation must
        happen at the Python layer instead.

        This test verifies that constructing a JobStatus from an invalid string
        raises ValueError, which is the guard we rely on in the service layer
        before writing to the DB.
        """
        with pytest.raises(ValueError):
            JobStatus("INVALID_STATE")


# ===========================================================================
# Integration tests — Plan table
# ===========================================================================

class TestPlanTable:
    def test_create_plan_with_valid_job(self, db: Session):
        """A Plan linked to an existing Job must be insertable."""
        job = make_job()
        db.add(job)
        db.commit()

        plan = Plan(
            job_id=job.id,
            raw_text="1. Add endpoint\n2. Write tests",
            status="draft",
        )
        db.add(plan)
        db.commit()

        fetched = db.get(Plan, plan.id)
        assert fetched is not None
        assert fetched.raw_text == "1. Add endpoint\n2. Write tests"
        assert fetched.status == "draft"

    def test_plan_status_defaults_to_draft(self, db: Session):
        """When status is not provided explicitly, it must default to 'draft'."""
        job = make_job()
        db.add(job)
        db.commit()

        plan = Plan(job_id=job.id, raw_text="Step 1")
        db.add(plan)
        db.commit()
        db.refresh(plan)

        assert plan.status == "draft"

    def test_plan_steps_defaults_to_none(self, db: Session):
        """steps is a Phase 2 field; it must be NULL in Phase 1."""
        job = make_job()
        db.add(job)
        db.commit()

        plan = Plan(job_id=job.id, raw_text="Some plan")
        db.add(plan)
        db.commit()
        db.refresh(plan)

        assert plan.steps is None

    def test_plan_steps_accepts_json_list(self, db: Session):
        """When steps is provided, it must round-trip through JSON correctly."""
        job = make_job()
        db.add(job)
        db.commit()

        steps = [{"step": 1, "description": "Create model"}]
        plan = Plan(job_id=job.id, raw_text="plan", steps=steps)
        db.add(plan)
        db.commit()
        db.refresh(plan)

        assert plan.steps == steps

    def test_plan_without_job_raises_integrity_error(self, db: Session):
        """
        Inserting a Plan whose job_id references a non-existent Job must
        raise IntegrityError (foreign key violation).

        This is a critical safety property: orphaned plans must not exist.
        """
        plan = Plan(job_id=99999, raw_text="Orphan plan", status="draft")
        db.add(plan)
        with pytest.raises(IntegrityError):
            db.flush()

    def test_plan_job_id_unique_constraint(self, db: Session):
        """
        job_id on Plan has a UNIQUE constraint (one-to-one with Job).
        Inserting a second Plan for the same Job must raise IntegrityError.
        """
        job = make_job()
        db.add(job)
        db.commit()

        plan1 = Plan(job_id=job.id, raw_text="First plan", status="draft")
        db.add(plan1)
        db.commit()

        plan2 = Plan(job_id=job.id, raw_text="Second plan", status="draft")
        db.add(plan2)
        with pytest.raises(IntegrityError):
            db.flush()

    def test_plan_human_feedback_nullable(self, db: Session):
        """human_feedback starts NULL and can be populated on rejection."""
        job = make_job()
        db.add(job)
        db.commit()

        plan = Plan(job_id=job.id, raw_text="plan")
        db.add(plan)
        db.commit()

        assert plan.human_feedback is None

        plan.human_feedback = "Missing error handling"
        plan.status = "rejected"
        db.commit()
        db.refresh(plan)

        assert plan.human_feedback == "Missing error handling"
        assert plan.status == "rejected"


# ===========================================================================
# Integration tests — Artifact table
# ===========================================================================

class TestArtifactTable:
    def test_create_artifact(self, db: Session):
        """An Artifact linked to an existing Job must be insertable."""
        job = make_job()
        db.add(job)
        db.commit()

        payload = json.dumps({"number": 1, "title": "Add /users endpoint"})
        artifact = Artifact(
            job_id=job.id,
            artifact_type="issue_data",
            content=payload,
        )
        db.add(artifact)
        db.commit()

        fetched = db.get(Artifact, artifact.id)
        assert fetched is not None
        assert fetched.artifact_type == "issue_data"
        assert json.loads(fetched.content)["title"] == "Add /users endpoint"

    def test_multiple_artifacts_per_job(self, db: Session):
        """A Job may have many Artifacts (unlike Plan which is one-to-one)."""
        job = make_job()
        db.add(job)
        db.commit()

        for artifact_type in ("issue_data", "plan", "diff"):
            db.add(Artifact(job_id=job.id, artifact_type=artifact_type, content="{}"))
        db.commit()

        db.refresh(job)
        assert len(job.artifacts) == 3

    def test_artifact_without_job_raises_integrity_error(self, db: Session):
        """An Artifact referencing a non-existent job_id must raise IntegrityError."""
        artifact = Artifact(job_id=99999, artifact_type="plan", content="text")
        db.add(artifact)
        with pytest.raises(IntegrityError):
            db.flush()


# ===========================================================================
# Integration tests — ORM relationships
# ===========================================================================

class TestRelationships:
    """
    Verify that SQLAlchemy relationships traverse correctly in both directions.
    """

    def test_job_plan_relationship_bidirectional(self, db: Session):
        """Job.plan and Plan.job must reference each other."""
        job = make_job()
        db.add(job)
        db.commit()

        plan = Plan(job_id=job.id, raw_text="plan text")
        db.add(plan)
        db.commit()

        # Forward: Job → Plan
        db.refresh(job)
        assert job.plan is not None
        assert job.plan.id == plan.id

        # Reverse: Plan → Job
        assert plan.job.id == job.id

    def test_job_artifacts_relationship(self, db: Session):
        """Job.artifacts must return all Artifact rows for that job."""
        job = make_job()
        db.add(job)
        db.commit()

        db.add(Artifact(job_id=job.id, artifact_type="issue_data", content="{}"))
        db.add(Artifact(job_id=job.id, artifact_type="plan", content="text"))
        db.commit()

        db.refresh(job)
        assert len(job.artifacts) == 2
        types = {a.artifact_type for a in job.artifacts}
        assert types == {"issue_data", "plan"}

    def test_job_with_no_plan_returns_none(self, db: Session):
        """Job.plan must be None when no Plan has been created yet."""
        job = make_job()
        db.add(job)
        db.commit()
        db.refresh(job)

        assert job.plan is None

    def test_job_with_no_artifacts_returns_empty_list(self, db: Session):
        """Job.artifacts must return an empty list when no artifacts exist."""
        job = make_job()
        db.add(job)
        db.commit()
        db.refresh(job)

        assert job.artifacts == []


# ===========================================================================
# Integration tests — cascade delete
# ===========================================================================

class TestCascadeDelete:
    """
    Deleting a Job must automatically delete its Plan and Artifacts.
    'cascade=all, delete-orphan' on the relationships enforces this.
    """

    def test_deleting_job_deletes_plan(self, db: Session):
        job = make_job()
        db.add(job)
        db.commit()

        plan = Plan(job_id=job.id, raw_text="plan")
        db.add(plan)
        db.commit()

        plan_id = plan.id
        db.delete(job)
        db.commit()

        assert db.get(Plan, plan_id) is None

    def test_deleting_job_deletes_artifacts(self, db: Session):
        job = make_job()
        db.add(job)
        db.commit()

        artifact = Artifact(job_id=job.id, artifact_type="issue_data", content="{}")
        db.add(artifact)
        db.commit()

        artifact_id = artifact.id
        db.delete(job)
        db.commit()

        assert db.get(Artifact, artifact_id) is None

    def test_schema_has_correct_tables(self, db: Session):
        """
        Verify that Base.metadata.create_all() produced the expected tables.
        Uses SQLAlchemy's inspect() API to query the DB schema.
        """
        engine = db.get_bind()
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert {"jobs", "plans", "artifacts"}.issubset(tables)
