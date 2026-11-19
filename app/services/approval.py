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

logger = logging.getLogger(__name__)


class ApprovalGateError(Exception):
    """Base exception for approval gate violations."""


class InvalidJobStateForApprovalError(ApprovalGateError):
    """Raised when an approval is attempted on a job not in AWAITING_APPROVAL."""


class MissingPlanError(ApprovalGateError):
    """Raised when no plan exists for the job being approved."""


class UnapprovedPlanExecutionError(ApprovalGateError):
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

        if job.status != JobStatus.AWAITING_APPROVAL:
            raise InvalidJobStateForApprovalError(
                f"Cannot review plan for Job {job_id} in state '{job.status.value}'. "
                "Job must be in 'AWAITING_APPROVAL'."
            )

        if job.plan is None:
            raise MissingPlanError(f"No plan found for Job {job_id}.")

        if decision == "approved":
            job.plan.status = "approved"
            job.status = JobStatus.GENERATING_CODE
            job.error_message = None
            logger.info("Human APPROVED plan for Job %d -> transitioned to GENERATING_CODE", job_id)
        elif decision == "rejected":
            if not feedback:
                raise ValueError("Feedback is required when rejecting an implementation plan.")
            job.plan.status = "rejected"
            job.plan.human_feedback = feedback
            job.status = JobStatus.PLANNING
            job.error_message = None
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
        if job.status != JobStatus.GENERATING_CODE:
            raise UnapprovedPlanExecutionError(
                f"Security violation: Cannot execute code generation on Job {job.id} "
                f"with status '{job.status.value}'. Status must be 'GENERATING_CODE'."
            )
        if job.plan is None or job.plan.status != "approved":
            raise UnapprovedPlanExecutionError(
                f"Security violation: Job {job.id} does not have an approved plan. "
                "Modifying target repository files without approval is strictly prohibited."
            )
