"""
app/services/workflow_auth.py
-----------------------------
Workflow authorization policy layer.
Enforces that authenticated requests and direct service invocations cannot bypass:
- Step-by-step state machine transitions
- Mandatory Human Plan Approval checkpoint
- Explicit Human PR Authorization checkpoint
- Safe terminal state locking (COMPLETED / FAILED)
- Safe idempotent actions and rejection of repeated approvals
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.db.models import Artifact, Job, JobStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow Authorization Exceptions
# =============================================================================


class WorkflowAuthorizationError(Exception):
    """Base exception for workflow authorization policy violations."""


class UnauthorizedActionError(WorkflowAuthorizationError):
    """Raised when an operation is disallowed for the job's current status."""


class PlanApprovalRequiredError(WorkflowAuthorizationError):
    """Raised when an operation requires an approved plan that is missing or unapproved."""


class PRAuthorizationRequiredError(WorkflowAuthorizationError):
    """Raised when PR creation is attempted without explicit authorization or prerequisites."""


class DuplicateActionError(WorkflowAuthorizationError):
    """Raised when an approval, execution, or PR action is repeated."""


# =============================================================================
# Test Execution Artifact Helpers (SEC-03B, SEC-03C)
# =============================================================================


def parse_test_execution_content(content: str) -> tuple[int, int] | None:
    """
    Strictly parse and validate test_output artifact content (SEC-03B, SEC-03C).
    Requires:
    - Presence of '#### Output' delimiter
    - Structured metadata header isolated before '#### Output'
    - Exactly one canonical attempt header: '### Test Execution (Attempt <n>)'
    - Exactly one canonical exit code line: '- **Exit Code**: <int>'
    Returns (attempt_number, exit_code) or None if missing, malformed, or ambiguous.
    """
    if not content or "#### Output" not in content:
        return None

    header = content.split("#### Output", 1)[0]

    # Validate that the expected attempt header is present exactly once
    attempt_headers = re.findall(
        r"^###\s*Test Execution\s*\(Attempt\s*(\d+)\)\s*$", header, re.MULTILINE
    )
    if len(attempt_headers) != 1:
        return None

    # Validate that exactly one exit code line is present in the header
    exit_code_matches = re.findall(r"^-\s*\*\*Exit Code\*\*:\s*(-?\d+)\s*$", header, re.MULTILINE)
    if len(exit_code_matches) != 1:
        return None

    try:
        attempt_num = int(attempt_headers[0])
        exit_code = int(exit_code_matches[0])
        return (attempt_num, exit_code)
    except ValueError:
        return None


def extract_test_execution_exit_code(content: str) -> int | None:
    """
    Extract the exit code from a test_output artifact using shared strict parser rules.
    Returns None if missing, malformed, duplicate, or ambiguous (SEC-03C).
    """
    parsed = parse_test_execution_content(content)
    return parsed[1] if parsed is not None else None


def get_latest_test_output_artifact(artifacts: list[Artifact] | None) -> Artifact | None:
    """
    Select the latest test execution artifact from the list of job artifacts (SEC-03B).
    Orders test_output artifacts primarily by attempt number, and secondarily by artifact id.
    Fails closed (returns None) if:
    - Any test artifact has missing, malformed, duplicate, or ambiguous metadata
    - Ambiguous ordering exists between multiple artifacts
    """
    if not artifacts:
        return None
    test_artifacts = [a for a in artifacts if a.artifact_type == "test_output"]
    if not test_artifacts:
        return None

    parsed_entries: list[tuple[int, int | None, Artifact]] = []
    for art in test_artifacts:
        parsed = parse_test_execution_content(art.content or "")
        if parsed is None:
            # Corrupted, malformed, or missing metadata in evidence -> fail closed
            return None
        attempt_num, _ = parsed
        parsed_entries.append((attempt_num, art.id, art))

    # Reject ambiguous ordering when multiple artifacts share an attempt number
    attempt_counts: dict[int, int] = {}
    for att, _, _ in parsed_entries:
        attempt_counts[att] = attempt_counts.get(att, 0) + 1
    for att, count in attempt_counts.items():
        if count > 1:
            tied_ids = [aid for a_att, aid, _ in parsed_entries if a_att == att]
            if any(aid is None for aid in tied_ids) or len(set(tied_ids)) != len(tied_ids):
                return None

    # Deterministic max by attempt number and secondary artifact ID
    return max(
        parsed_entries,
        key=lambda entry: (entry[0], entry[1] or 0),
    )[2]


# =============================================================================
# Workflow Authorization Policy
# =============================================================================


class WorkflowAuthorizationPolicy:
    """
    Centralized workflow authorization rules for the autonomous engineering pipeline.
    Enforces authorization gates at the service/orchestrator layer so that direct
    service calls and API routes alike cannot bypass policies.
    """

    PLANNING_PERMITTED_STATES: set[JobStatus] = {
        JobStatus.PENDING,
        JobStatus.FETCHING_ISSUE,
        JobStatus.INDEXING_CODEBASE,
        JobStatus.PLANNING,
    }

    # States that can only be reached through dedicated internal workflows or human
    # approval endpoints, never via manual/external status PATCH requests.
    PROTECTED_INTERNAL_STATES: set[JobStatus] = {
        JobStatus.GENERATING_CODE,
        JobStatus.RUNNING_TESTS,
        JobStatus.OPENING_PR,
        JobStatus.COMPLETED,
    }

    @classmethod
    def authorize_external_status_update(
        cls,
        job: Job,
        target_status: JobStatus,
    ) -> None:
        """
        Authorize manual/external status PATCH requests.
        Prevents external callers from manually advancing jobs into workflow states
        that must only be reached through validated internal operations or dedicated
        human approval checkpoints (SEC-03).
        """
        if target_status in cls.PROTECTED_INTERNAL_STATES:
            raise UnauthorizedActionError(
                f"External status update to '{target_status.value}' is forbidden. "
                f"State '{target_status.value}' can only be reached through validated "
                "internal operations or dedicated approval checkpoints."
            )

    @classmethod
    def authorize_start_planning(cls, job: Job) -> None:
        """
        Authorize starting or resuming the planning phase.
        Must be in a pre-approval planning state.
        Cannot be launched if already in or past AWAITING_APPROVAL, or terminal.
        """
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise UnauthorizedActionError(
                f"Cannot start planning for Job {job.id}: "
                f"job is in terminal state '{job.status.value}'."
            )
        if job.status not in cls.PLANNING_PERMITTED_STATES:
            raise UnauthorizedActionError(
                f"Cannot start planning for Job {job.id} in state '{job.status.value}'. "
                "Job has already advanced past planning."
            )

    @classmethod
    def authorize_plan_review(
        cls,
        job: Job,
        decision: Literal["approved", "rejected"] | str,
        feedback: str | None = None,
    ) -> None:
        """
        Authorize human plan approval or rejection.
        Strictly requires AWAITING_APPROVAL and an existing plan.
        Rejects repeated reviews if the plan has already been approved.
        """
        if job.status != JobStatus.AWAITING_APPROVAL:
            raise UnauthorizedActionError(
                f"Cannot review plan for job in '{job.status.value}' state. "
                "Job must be in 'AWAITING_APPROVAL'."
            )
        if job.plan is None:
            raise PlanApprovalRequiredError(f"No plan found for job {job.id}.")

        if job.plan.status == "approved":
            raise DuplicateActionError(f"Plan for Job {job.id} has already been approved.")

        if decision == "rejected" and not (feedback and feedback.strip()):
            raise ValueError("Feedback is required when rejecting an implementation plan.")

    @classmethod
    def authorize_execution(cls, job: Job) -> None:
        """
        Authorize code generation, modification, and test execution.
        Strictly requires GENERATING_CODE status and an approved plan.
        """
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise UnauthorizedActionError(
                f"Cannot execute code generation for Job {job.id}: "
                f"job is in terminal state '{job.status.value}'."
            )
        if job.status != JobStatus.GENERATING_CODE:
            raise UnauthorizedActionError(
                f"Cannot execute code generation for Job {job.id} in state '{job.status.value}'. "
                "Job must be in 'GENERATING_CODE' status."
            )
        if job.plan is None or job.plan.status != "approved":
            raise PlanApprovalRequiredError(
                f"Cannot execute code generation for Job {job.id}: "
                "implementation plan has not been approved by a human."
            )

    @classmethod
    def authorize_pr_creation(cls, job: Job) -> None:
        """
        Authorize GitHub Pull Request submission.
        Strictly requires OPENING_PR status, approved plan, and no duplicate PR.
        """
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise UnauthorizedActionError(
                f"Cannot create PR for Job {job.id}: job is in terminal state '{job.status.value}'."
            )
        if job.status != JobStatus.OPENING_PR:
            raise UnauthorizedActionError(
                f"Cannot create PR for Job {job.id} in state '{job.status.value}'. "
                "Job must be in 'OPENING_PR' state."
            )
        if job.plan is None or job.plan.status != "approved":
            raise PlanApprovalRequiredError(
                f"Cannot create PR for Job {job.id}: Plan has not been approved."
            )
        # Check for duplicate PR creation
        artifacts = job.__dict__.get("artifacts")
        if artifacts is not None:
            for art in artifacts:
                if art.artifact_type == "pr_url":
                    raise DuplicateActionError(
                        f"A Pull Request has already been submitted for Job {job.id}."
                    )

    @classmethod
    def authorize_state_transition(
        cls,
        job: Job,
        target_status: JobStatus,
    ) -> None:
        """
        Authorize manual or automated state transitions.
        Enforces prerequisites for sensitive states.
        """
        # Transitioning to GENERATING_CODE requires plan approval
        if target_status == JobStatus.GENERATING_CODE:
            if job.plan is None or job.plan.status != "approved":
                raise PlanApprovalRequiredError(
                    f"Cannot transition Job {job.id} to 'GENERATING_CODE': "
                    "implementation plan must be explicitly approved first."
                )

        # Transitioning to OPENING_PR requires plan approval and verified test completion
        if target_status == JobStatus.OPENING_PR:
            if job.plan is None or job.plan.status != "approved":
                raise PlanApprovalRequiredError(
                    f"Cannot transition Job {job.id} to 'OPENING_PR': "
                    "implementation plan must be approved."
                )
            if job.status == JobStatus.RUNNING_TESTS:
                # SEC-03A: Fail-closed if artifacts relationship is unloaded or unavailable
                if "artifacts" not in job.__dict__ or job.artifacts is None:
                    raise UnauthorizedActionError(
                        f"Cannot transition Job {job.id} to 'OPENING_PR': "
                        "test execution evidence is unavailable "
                        "(artifacts relationship not loaded)."
                    )

                # SEC-03B: Require evidence from the latest test execution
                latest_art = get_latest_test_output_artifact(job.artifacts)
                if latest_art is None:
                    raise UnauthorizedActionError(
                        f"Cannot transition Job {job.id} to 'OPENING_PR': "
                        "automated tests have not completed successfully "
                        "(no test output artifact found)."
                    )

                # SEC-03C: Strict exit code parsing from metadata header
                exit_code = extract_test_execution_exit_code(latest_art.content or "")
                if exit_code is None:
                    raise UnauthorizedActionError(
                        f"Cannot transition Job {job.id} to 'OPENING_PR': "
                        "automated tests have not completed successfully "
                        "(missing or invalid exit code)."
                    )
                if exit_code != 0:
                    raise UnauthorizedActionError(
                        f"Cannot transition Job {job.id} to 'OPENING_PR': "
                        f"automated tests have not completed successfully (exit code {exit_code})."
                    )

        # Transitioning to COMPLETED requires a PR artifact
        if target_status == JobStatus.COMPLETED:
            if "artifacts" not in job.__dict__ or job.artifacts is None:
                raise PRAuthorizationRequiredError(
                    f"Cannot transition Job {job.id} to 'COMPLETED': "
                    "Pull Request evidence is unavailable "
                    "(artifacts relationship not loaded)."
                )
            has_pr = any(a.artifact_type == "pr_url" for a in (job.artifacts or []))
            if not has_pr:
                raise PRAuthorizationRequiredError(
                    f"Cannot transition Job {job.id} to 'COMPLETED': "
                    "a Pull Request must be authorized and submitted first."
                )

    @classmethod
    def authorize_cancellation(cls, job: Job) -> None:
        """
        Authorize job cancellation.
        Terminal states cannot be cancelled.
        """
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise UnauthorizedActionError(
                f"Cannot cancel Job {job.id}: "
                f"job is already in terminal state '{job.status.value}'."
            )
