"""
app/db/models.py
----------------
SQLAlchemy ORM model definitions for the Agentic Software Development Workflow.

This module is the **data contract** for the entire application. Every other
layer — services, adapters, API routers — works with these classes. Changes
here ripple upward, so they should be deliberate.

Design principles
-----------------
Pure definitions, no I/O
    This module imports nothing from app.core or app.adapters. It contains
    only class definitions. The database connection is managed separately in
    app/db/session.py so models can be imported anywhere without triggering
    network or file I/O.

SQLAlchemy 2.x style
    Uses the modern Mapped[T] / mapped_column() annotation syntax introduced
    in SQLAlchemy 2.0. This is strictly typed and plays well with mypy.
    The older Column(String) syntax still works but is not used here.

Explicit over implicit
    Every column, foreign key, and relationship is declared explicitly.
    No magic auto-discovery. A new developer can read this file and
    understand the entire database schema without running the application.

Enum for state machine
    JobStatus is a Python Enum. Storing enum members in the DB (not raw
    strings) makes invalid state transitions a Python error rather than a
    silent data corruption.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# =============================================================================
# Declarative base
# =============================================================================

class Base(DeclarativeBase):
    """
    Root base class for all ORM models.

    Every model inherits from Base. SQLAlchemy uses it to:
    - Track all table definitions via Base.metadata
    - Run Base.metadata.create_all(engine) to create every table at once
    - Run Base.metadata.drop_all(engine) in tests to reset the schema

    Why DeclarativeBase instead of declarative_base()?
    DeclarativeBase is the SQLAlchemy 2.0 class-based API. It provides
    better IDE support and is compatible with the Mapped[T] annotation style.
    """

    # Shared type annotation map so SQLAlchemy knows how to handle Python's
    # `Any` type (used by the JSON column on Plan).
    type_annotation_map = {
        dict[str, Any]: JSON,
        list[Any]: JSON,
    }


# =============================================================================
# JobStatus — the agent's state machine
# =============================================================================

class JobStatus(str, enum.Enum):
    """
    All valid states a Job can occupy, in the order they occur.

    Why inherit from both str and enum.Enum?
    - str mixin: the enum member's value IS a plain string, so it serialises
      to JSON as "PENDING", not as {"value": "PENDING"}. FastAPI and Pydantic
      handle it transparently.
    - enum.Enum: prevents invalid states being stored (you cannot construct
      JobStatus("TYPO") — it raises ValueError).

    Phase 1 pipeline:
        PENDING → FETCHING_ISSUE → PLANNING → AWAITING_APPROVAL

    The pipeline pauses at AWAITING_APPROVAL and waits for a human
    POST /jobs/{id}/approve before advancing further.

    Phase 2+ states (defined here so the DB schema is complete):
        AWAITING_APPROVAL → GENERATING_CODE → RUNNING_TESTS → OPENING_PR
                                            ↘ (retry loop back to GENERATING_CODE)
        OPENING_PR → COMPLETED
        Any state → FAILED  (on unrecoverable error)

    Note: INDEXING_CODEBASE is in the full state machine but Phase 1 skips it,
    transitioning directly from FETCHING_ISSUE to PLANNING.
    """

    PENDING = "PENDING"
    FETCHING_ISSUE = "FETCHING_ISSUE"
    INDEXING_CODEBASE = "INDEXING_CODEBASE"   # Phase 2
    PLANNING = "PLANNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    GENERATING_CODE = "GENERATING_CODE"        # Phase 2
    RUNNING_TESTS = "RUNNING_TESTS"            # Phase 2
    OPENING_PR = "OPENING_PR"                 # Phase 3
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =============================================================================
# Job — the top-level record for a single agent run
# =============================================================================

class Job(Base):
    """
    One row per agent invocation triggered by a GitHub issue URL.

    The Job is the root entity. All other records (Plan, Artifact) reference
    a Job via foreign key. Deleting a Job cascades to its children.

    Lifecycle
    ---------
    Created:  POST /jobs  → status = PENDING
    Updated:  each pipeline step transitions status and sets updated_at
    Terminal: status is either COMPLETED or FAILED

    Column notes
    ------------
    repo_owner / repo_name / issue_number
        Parsed from issue_url at job creation time and stored separately so
        query code never has to split URLs. The raw issue_url is kept for
        display purposes.

    workspace_path
        Set to None initially; filled in after the repo is cloned locally.
        Nullable because the clone step may not have run yet (or may have
        failed) when the row is queried.

    error_message
        Only meaningful when status = FAILED. Contains the exception message
        or adapter error so a developer can diagnose the failure without
        trawling log files.
    """

    __tablename__ = "jobs"

    # --- Primary key ---------------------------------------------------------
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing surrogate key.",
    )

    # --- Issue identification ------------------------------------------------
    issue_url: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
        comment="Full GitHub issue URL as submitted by the developer.",
    )
    repo_owner: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="GitHub organisation or user name, parsed from issue_url.",
    )
    repo_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="GitHub repository name, parsed from issue_url.",
    )
    issue_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="GitHub issue number, parsed from issue_url.",
    )

    # --- Agent workspace -----------------------------------------------------
    workspace_path: Mapped[str | None] = mapped_column(
        String(4096),
        nullable=True,
        default=None,
        comment=(
            "Absolute path of the locally cloned repository workspace. "
            "NULL until the clone step completes successfully."
        ),
    )

    # --- State machine -------------------------------------------------------
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status_enum", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=JobStatus.PENDING,
        comment="Current position in the agent state machine.",
    )

    # --- Error tracking ------------------------------------------------------
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Exception message set when status transitions to FAILED.",
    )

    # --- Timestamps ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        comment="Row creation timestamp, set by the database on INSERT.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment=(
            "Last modification timestamp. "
            "Updated automatically by SQLAlchemy on every flush. "
            "Use this to poll for status changes."
        ),
    )

    # --- Relationships -------------------------------------------------------
    plan: Mapped[Plan | None] = relationship(
        "Plan",
        back_populates="job",
        uselist=False,          # one-to-one: a job has at most one plan
        cascade="all, delete-orphan",
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        "Artifact",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="Artifact.created_at",
    )

    def __repr__(self) -> str:
        return (
            f"<Job id={self.id} status={self.status.value!r} "
            f"issue={self.repo_owner}/{self.repo_name}#{self.issue_number}>"
        )


# =============================================================================
# Plan — the LLM-generated implementation plan
# =============================================================================

class Plan(Base):
    """
    The implementation plan produced by the LLM for a specific Job.

    One Job has at most one Plan (one-to-one relationship). The Plan is
    created when the pipeline reaches the PLANNING stage and transitions the
    Job to AWAITING_APPROVAL. The human then either approves or rejects it.

    Status values
    -------------
    draft     — just created by the LLM, awaiting human review
    approved  — human approved; pipeline will advance to GENERATING_CODE
    rejected  — human rejected; pipeline may revise and create a new Plan row

    Column notes
    ------------
    raw_text
        The LLM's verbatim output. Stored unmodified so it can always be
        displayed to the developer, even if later parsing (steps column) fails.

    steps
        Phase 2: a JSON array of structured plan steps parsed from raw_text.
        Stored as JSON so structured data can be queried without a separate
        table. NULL in Phase 1 — the LLM output is not yet parsed.

    human_feedback
        Only meaningful when status = 'rejected'. Contains the developer's
        explanation of what the plan got wrong, which the agent passes back
        to the LLM as context when regenerating.
    """

    __tablename__ = "plans"

    # --- Primary key ---------------------------------------------------------
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # --- Foreign key ---------------------------------------------------------
    job_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,            # enforces one-to-one at the DB level
        comment="The Job this plan belongs to.",
    )

    # --- Plan content --------------------------------------------------------
    raw_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Verbatim LLM output — the plan as returned by the model.",
    )
    steps: Mapped[list[Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        comment=(
            "Phase 2: structured plan steps parsed from raw_text. "
            "Each element is a dict with keys: step_number, description, file_paths. "
            "NULL in Phase 1."
        ),
    )

    # --- Review state --------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="draft",
        comment="Review state: 'draft' | 'approved' | 'rejected'.",
    )
    human_feedback: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment=(
            "Developer's rejection note. "
            "Passed back to the LLM as context when regenerating the plan."
        ),
    )

    # --- Timestamps ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # --- Relationships -------------------------------------------------------
    job: Mapped[Job] = relationship(
        "Job",
        back_populates="plan",
    )

    def __repr__(self) -> str:
        return f"<Plan id={self.id} job_id={self.job_id} status={self.status!r}>"


# =============================================================================
# Artifact — generic envelope for pipeline outputs
# =============================================================================

class Artifact(Base):
    """
    A generic record that stores any discrete output produced during a pipeline run.

    Why a generic envelope instead of typed tables?
    -----------------------------------------------
    In Phase 1 there are only two artifact types (issue_data and plan).
    Creating a separate table for each type would add schema complexity
    with no benefit at this stage. The generic envelope pattern keeps the
    schema simple and lets us add new artifact types by inserting a new row,
    not by running a migration.

    Trade-off: the content column is an unstructured string. Callers are
    responsible for serialising/deserialising (typically JSON). This is
    acceptable in Phase 1 because artifact content is only read for display
    and debugging, not for structured queries.

    Artifact types (Phase 1)
    ------------------------
    issue_data  — JSON-serialised IssueData fetched from the GitHub API.
                  Stored so the pipeline can replay without hitting the API again.
    plan        — The raw LLM plan text (redundant with Plan.raw_text, kept for
                  the unified artifact timeline view).

    Artifact types (Phase 2+)
    -------------------------
    diff        — Unified diff of generated code changes.
    test_output — Captured stdout/stderr from the Pytest run.
    pr_url      — URL of the opened draft pull request.
    """

    __tablename__ = "artifacts"

    # --- Primary key ---------------------------------------------------------
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # --- Foreign key ---------------------------------------------------------
    job_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        comment="The Job that produced this artifact.",
    )

    # --- Content -------------------------------------------------------------
    artifact_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment=(
            "Discriminator string identifying the artifact kind. "
            "One of: issue_data | plan | diff | test_output | pr_url"
        ),
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "The artifact payload. For structured data (issue_data, diff) "
            "this is a JSON string. For plain text (plan, test_output) it is raw text."
        ),
    )

    # --- Timestamps ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        comment="When this artifact was written. Orders the artifact timeline.",
    )

    # --- Relationships -------------------------------------------------------
    job: Mapped[Job] = relationship(
        "Job",
        back_populates="artifacts",
    )

    def __repr__(self) -> str:
        return (
            f"<Artifact id={self.id} job_id={self.job_id} "
            f"type={self.artifact_type!r}>"
        )
