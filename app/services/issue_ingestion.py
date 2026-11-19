"""
app/services/issue_ingestion.py
-------------------------------
Service for fetching, validating, and normalizing GitHub issues,
and persisting them as Job records with associated issue_data artifacts.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.github import (
    GitHubClient,
    GitHubComment,
    GitHubIssue,
    GitHubRepository,
)
from app.db.models import Artifact, Job, JobStatus
from app.intelligence.planner import NormalizedIssueData

logger = logging.getLogger(__name__)

__all__ = [
    "DuplicateJobError",
    "IssueIngestionError",
    "IssueIngestionService",
    "NormalizedIssueData",
    "PullRequestNotAllowedError",
]


class IssueIngestionError(Exception):
    """Base exception for issue ingestion failures."""


class PullRequestNotAllowedError(IssueIngestionError):
    """Raised when the specified issue number is actually a Pull Request."""


class DuplicateJobError(IssueIngestionError):
    """Raised when an active Job already exists for the repository issue."""


class IssueIngestionService:
    """Service orchestrating validation, GitHub retrieval, and Job creation."""

    def __init__(self, github_client: GitHubClient) -> None:
        self.github_client = github_client

    async def ingest_issue(
        self,
        db: AsyncSession,
        owner: str,
        repo: str,
        issue_number: int,
    ) -> tuple[Job, NormalizedIssueData]:
        """
        Validate, retrieve, normalize, and persist a GitHub issue into a new Job.

        Parameters
        ----------
        db : AsyncSession
            Active database session.
        owner : str
            GitHub repository owner.
        repo : str
            GitHub repository name.
        issue_number : int
            GitHub issue number.

        Returns
        -------
        tuple[Job, NormalizedIssueData]
            The created Job and normalized issue data.
        """
        # 1. Check for existing active job to fail fast
        existing_stmt = select(Job).where(
            func.lower(Job.repo_owner) == owner.lower(),
            func.lower(Job.repo_name) == repo.lower(),
            Job.issue_number == issue_number,
            Job.status.not_in([JobStatus.COMPLETED, JobStatus.FAILED]),
        )
        existing_result = await db.execute(existing_stmt)
        if existing_result.scalars().first() is not None:
            raise DuplicateJobError(
                f"An active job already exists for issue {owner}/{repo}#{issue_number}."
            )

        # 2. Fetch repository metadata
        logger.info("Fetching repository metadata for %s/%s", owner, repo)
        repo_data: GitHubRepository = await self.github_client.get_repository(owner, repo)

        # 3. Fetch issue
        logger.info("Fetching issue %s/%s#%d", owner, repo, issue_number)
        issue: GitHubIssue = await self.github_client.get_issue(owner, repo, issue_number)

        # 4. Validate not a PR
        if issue.pull_request is not None:
            raise PullRequestNotAllowedError(
                f"Item #{issue_number} in {owner}/{repo} is a Pull Request, not an Issue."
            )

        # 5. Fetch all comments
        logger.info("Fetching comments for %s/%s#%d", owner, repo, issue_number)
        comments: list[GitHubComment] = await self.github_client.get_all_issue_comments(
            owner, repo, issue_number
        )

        # 6. Normalize issue data
        normalized = NormalizedIssueData(
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            issue_url=issue.html_url or f"https://github.com/{owner}/{repo}/issues/{issue_number}",
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            author=issue.user.login,
            labels=[lbl.name for lbl in issue.labels],
            comments_count=len(comments),
            comments=[
                {
                    "id": c.id,
                    "author": c.user.login,
                    "body": c.body,
                    "created_at": c.created_at.isoformat(),
                }
                for c in comments
            ],
            repository_metadata={
                "id": repo_data.id,
                "full_name": repo_data.full_name,
                "private": repo_data.private,
                "default_branch": repo_data.default_branch,
                "description": repo_data.description or "",
            },
        )

        # 7. Create Job in database
        job = Job(
            issue_url=normalized.issue_url,
            repo_owner=owner,
            repo_name=repo,
            issue_number=issue_number,
            status=JobStatus.PENDING,
        )
        db.add(job)

        try:
            await db.flush()
        except IntegrityError as exc:
            await db.rollback()
            raise DuplicateJobError(
                f"An active job already exists for issue {owner}/{repo}#{issue_number}."
            ) from exc

        # 8. Attach issue_data artifact
        artifact = Artifact(
            job_id=job.id,
            artifact_type="issue_data",
            content=normalized.model_dump_json(),
        )
        db.add(artifact)

        try:
            await db.commit()
            await db.refresh(job)
        except Exception:
            await db.rollback()
            raise

        logger.info(
            "Successfully ingested issue %s/%s#%d as Job %d", owner, repo, issue_number, job.id
        )
        return job, normalized
