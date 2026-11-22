"""
app/services/pr_service.py
--------------------------
Pull Request generation and automation service.
Enforces an explicit authorization gate before pushing branches or creating
remote pull requests on GitHub.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.github import GitHubClient, GitHubPullRequest
from app.db.models import Artifact, Job, JobStatus
from app.orchestrator.state_machine import JobStateMachine
from app.services.workflow_auth import (
    DuplicateActionError,
    PlanApprovalRequiredError,
    PRAuthorizationRequiredError,
    UnauthorizedActionError,
    WorkflowAuthorizationError,
    WorkflowAuthorizationPolicy,
)

logger = logging.getLogger(__name__)


class PRServiceError(Exception):
    """Base exception for PR creation failures."""


class PRAuthorizationError(PRServiceError, PRAuthorizationRequiredError):
    """Raised when PR creation is attempted without authorization or in invalid state."""


class PullRequestService:
    """Manages PR generation, body formatting, and remote GitHub submission."""

    def __init__(self, github_client: GitHubClient) -> None:
        self.github_client = github_client

    def build_pr_body(
        self,
        job: Job,
        plan_summary: str,
        technical_approach: str,
        test_summary: str = "All automated unit and regression tests passed.",
    ) -> str:
        """Construct structured PR description referencing the original issue."""
        return (
            f"## Automated Pull Request\n\n"
            f"Closes #{job.issue_number}\n\n"
            f"### Implementation Summary\n"
            f"{plan_summary}\n\n"
            f"### Technical Approach\n"
            f"{technical_approach}\n\n"
            f"### Testing & Verification\n"
            f"{test_summary}\n\n"
            f"---\n*Generated autonomously by Agentic Software Development Workflow.*"
        )

    async def create_pull_request_for_job(
        self,
        db: AsyncSession,
        job_id: int,
        feature_branch: str,
        base_branch: str = "main",
    ) -> GitHubPullRequest:
        """
        Submit a GitHub Pull Request for an approved and tested job.
        Strictly enforces that Job is in OPENING_PR and plan is approved.
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

        # Enforcement gate
        try:
            WorkflowAuthorizationPolicy.authorize_pr_creation(job)
        except (
            UnauthorizedActionError,
            PlanApprovalRequiredError,
            DuplicateActionError,
            WorkflowAuthorizationError,
        ) as exc:
            raise PRAuthorizationError(str(exc)) from exc

        assert job.plan is not None

        title = f"fix({job.repo_name}): resolve #{job.issue_number}"
        body = self.build_pr_body(
            job=job,
            plan_summary=job.plan.raw_text[:500],
            technical_approach="Applied verified changes adhering to approved plan.",
        )

        logger.info(
            "Creating remote PR on %s/%s for branch %s",
            job.repo_owner,
            job.repo_name,
            feature_branch,
        )

        pr = await self.github_client.create_pull_request(
            owner=job.repo_owner,
            repo=job.repo_name,
            title=title,
            body=body,
            head=feature_branch,
            base=base_branch,
        )

        # Store PR URL artifact
        artifact = Artifact(
            job_id=job.id,
            artifact_type="pr_url",
            content=pr.html_url
            or f"https://github.com/{job.repo_owner}/{job.repo_name}/pull/{pr.number}",
        )
        db.add(artifact)
        if "artifacts" in job.__dict__ and job.artifacts is not None:
            job.artifacts.append(artifact)
        else:
            job.artifacts = [artifact]

        # Advance job to terminal COMPLETED state via state machine
        JobStateMachine.transition(job, JobStatus.COMPLETED)

        try:
            await db.commit()
            await db.refresh(job)
        except Exception:
            await db.rollback()
            raise

        logger.info("Successfully opened PR #%d for Job %d -> COMPLETED", pr.number, job.id)
        return pr
