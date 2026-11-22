"""
app/orchestrator/state_machine.py
---------------------------------
State machine rules, transition validator, and transition errors
for the Agentic Software Development Workflow.
"""

from __future__ import annotations

import logging

from app.db.models import Job, JobStatus
from app.services.workflow_auth import (
    WorkflowAuthorizationError,
    WorkflowAuthorizationPolicy,
)

logger = logging.getLogger(__name__)


class InvalidTransitionError(WorkflowAuthorizationError):
    """Raised when an illegal state transition is attempted on a Job."""


# Valid state transitions across the full pipeline
VALID_JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.FETCHING_ISSUE, JobStatus.FAILED},
    JobStatus.FETCHING_ISSUE: {
        JobStatus.INDEXING_CODEBASE,
        JobStatus.PLANNING,
        JobStatus.FAILED,
    },
    JobStatus.INDEXING_CODEBASE: {JobStatus.PLANNING, JobStatus.FAILED},
    JobStatus.PLANNING: {JobStatus.AWAITING_APPROVAL, JobStatus.FAILED},
    JobStatus.AWAITING_APPROVAL: {
        JobStatus.GENERATING_CODE,
        JobStatus.PLANNING,
        JobStatus.FAILED,
    },
    JobStatus.GENERATING_CODE: {JobStatus.RUNNING_TESTS, JobStatus.FAILED},
    JobStatus.RUNNING_TESTS: {
        JobStatus.OPENING_PR,
        JobStatus.GENERATING_CODE,
        JobStatus.FAILED,
    },
    JobStatus.OPENING_PR: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),  # Terminal state
    JobStatus.FAILED: set(),  # Terminal state
}


class JobStateMachine:
    """Validates and applies state transitions to Job entities."""

    @classmethod
    def can_transition(cls, current_status: JobStatus, target_status: JobStatus) -> bool:
        """Return True if transitioning from current_status to target_status is permitted."""
        allowed = VALID_JOB_TRANSITIONS.get(current_status, set())
        return target_status in allowed

    @classmethod
    def transition(
        cls,
        job: Job,
        target_status: JobStatus,
        error_message: str | None = None,
        *,
        is_external: bool = False,
    ) -> None:
        """
        Apply a transition to the Job entity or raise
        InvalidTransitionError / WorkflowAuthorizationError.

        Parameters
        ----------
        job : Job
            Target Job entity.
        target_status : JobStatus
            Desired new status.
        error_message : str | None
            Required when target_status is FAILED. Cleared on non-FAILED transitions.
        is_external : bool
            True if triggered via external status PATCH request.
        """
        if not cls.can_transition(job.status, target_status):
            allowed = [s.value for s in VALID_JOB_TRANSITIONS.get(job.status, set())]
            raise InvalidTransitionError(
                f"Invalid state transition from '{job.status.value}' to '{target_status.value}'. "
                f"Cannot transition job {job.id}. Allowed next states: {allowed}"
            )

        if is_external:
            WorkflowAuthorizationPolicy.authorize_external_status_update(job, target_status)

        # Enforce workflow authorization gates on transitions
        WorkflowAuthorizationPolicy.authorize_state_transition(job, target_status)

        job.status = target_status
        if target_status == JobStatus.FAILED:
            job.error_message = error_message or "Unknown failure"
        else:
            job.error_message = None

        logger.info(
            "Job %d transitioned to %s",
            job.id,
            target_status.value,
        )
