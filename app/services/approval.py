"""
app/services/approval.py
------------------------
Service implementing the mandatory Human Approval Gate.
Enforces that code generation and workspace modifications can never proceed
without an explicitly approved implementation plan.
"""

from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Job, JobStatus
from app.orchestrator.state_machine import JobStateMachine
from app.services.workflow_auth import (
    DuplicateActionError,
    PlanApprovalRequiredError,
    UnauthorizedActionError,
    WorkflowAuthorizationError,
    WorkflowAuthorizationPolicy,
)

logger = logging.getLogger(__name__)


class ApprovalGateError(WorkflowAuthorizationError):
    """Base exception for approval gate violations."""


class InvalidJobStateForApprovalError(ApprovalGateError, UnauthorizedActionError):
    """Raised when an approval is attempted on a job not in AWAITING_APPROVAL."""


class MissingPlanError(ApprovalGateError, PlanApprovalRequiredError):
    """Raised when no plan exists for the job being approved."""


class UnapprovedPlanExecutionError(ApprovalGateError, PlanApprovalRequiredError):
    """Raised when code generation is attempted without an approved plan."""


class PlanApprovalService:
    """Coordinates plan approval, rejection, and runtime enforcement gates."""

    @staticmethod
    async def review_plan(
        db: AsyncSession,
        job_id: int,
        decision: Literal["approved", "rejected"],
        feedback: str | None = None,
    ) -> Job:
        """
        Record a human review decision on a job's implementation plan.

        Parameters
        ----------
        db : AsyncSession
            Active database session.
        job_id : int
            Target job identifier.
        decision : "approved" | "rejected"
            The human reviewer's decision.
        feedback : str | None
            Required explanation if decision is "rejected".

        Returns
        -------
        Job
            The updated Job entity.
        """
        stmt = (
            select(Job)
            .options(selectinload(Job.plan), selectinload(Job.artifacts))
            .where(Job.id == job_id)
        )
        result = await db.execute(stmt)
        job = result.scalars().first()

        if job is None:
            raise ValueError(f"Job with id {job_id} not found.")

        try:
            WorkflowAuthorizationPolicy.authorize_plan_review(job, decision, feedback)
        except UnauthorizedActionError as exc:
            raise InvalidJobStateForApprovalError(str(exc)) from exc
        except DuplicateActionError as exc:
            raise InvalidJobStateForApprovalError(str(exc)) from exc
        except PlanApprovalRequiredError as exc:
            raise MissingPlanError(str(exc)) from exc

        assert job.plan is not None

        if decision == "approved":
            job.plan.status = "approved"
            JobStateMachine.transition(job, JobStatus.GENERATING_CODE)
            logger.info("Human APPROVED plan for Job %d -> transitioned to GENERATING_CODE", job_id)
        elif decision == "rejected":
            job.plan.status = "rejected"
            job.plan.human_feedback = feedback
            JobStateMachine.transition(job, JobStatus.PLANNING)
            logger.info(
                "Human REJECTED plan for Job %d -> transitioned to PLANNING for revision", job_id
            )

        try:
            await db.commit()
        except Exception:
            await db.rollback()
            raise

        # Reload with relationships eagerly loaded to avoid MissingGreenlet
        reload_stmt = (
            select(Job)
            .options(selectinload(Job.plan), selectinload(Job.artifacts))
            .where(Job.id == job_id)
        )
        reload_res = await db.execute(reload_stmt)
        return reload_res.scalars().one()

    @staticmethod
    def assert_can_generate_code(job: Job) -> None:
        """
        Critical security gate:
        Verify that a job and its plan are strictly approved before code generation.
        Raises UnapprovedPlanExecutionError if any condition is not met.
        """
        try:
            WorkflowAuthorizationPolicy.authorize_execution(job)
        except WorkflowAuthorizationError as exc:
            raise UnapprovedPlanExecutionError(f"Security violation: {exc}") from exc
