"""
tests/unit/test_issue_ingestion.py
----------------------------------
Unit tests for IssueIngestionService (app/services/issue_ingestion.py).
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.github import GitHubClient
from app.db.models import Base, Job, JobStatus
from app.services.issue_ingestion import (
    DuplicateJobError,
    IssueIngestionService,
    PullRequestNotAllowedError,
)

# SQLite test engine with temporary in-memory database
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_session() -> AsyncSession:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class TestIssueIngestionService:
    """Verify GitHub issue ingestion into database jobs."""

    async def test_ingest_issue_success(self, test_session: AsyncSession) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/repos/octocat/Hello-World":
                return httpx.Response(
                    200,
                    json={
                        "id": 1,
                        "name": "Hello-World",
                        "full_name": "octocat/Hello-World",
                        "private": False,
                        "default_branch": "main",
                        "owner": {"login": "octocat", "id": 1},
                    },
                )
            if path == "/repos/octocat/Hello-World/issues/42":
                return httpx.Response(
                    200,
                    json={
                        "id": 100,
                        "number": 42,
                        "title": "Bug in main loop",
                        "body": "Crash occurs on line 10",
                        "state": "open",
                        "user": {"login": "alice", "id": 2},
                        "labels": [{"name": "bug"}],
                        "created_at": "2026-11-10T12:00:00Z",
                        "updated_at": "2026-11-10T12:00:00Z",
                        "html_url": "https://github.com/octocat/Hello-World/issues/42",
                    },
                )
            if path == "/repos/octocat/Hello-World/issues/42/comments":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": 101,
                            "body": "Reproduced on Linux",
                            "user": {"login": "bob", "id": 3},
                            "created_at": "2026-11-10T12:30:00Z",
                            "updated_at": "2026-11-10T12:30:00Z",
                        }
                    ],
                )
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            service = IssueIngestionService(client)
            job, normalized = await service.ingest_issue(test_session, "octocat", "Hello-World", 42)

        assert job.id is not None
        assert job.status == JobStatus.PENDING
        assert job.repo_owner == "octocat"
        assert job.repo_name == "Hello-World"
        assert job.issue_number == 42

        assert normalized.title == "Bug in main loop"
        assert normalized.author == "alice"
        assert normalized.comments_count == 1
        assert normalized.comments[0]["body"] == "Reproduced on Linux"

    async def test_ingest_rejects_pull_request(self, test_session: AsyncSession) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/repos/octocat/Hello-World":
                return httpx.Response(
                    200,
                    json={
                        "id": 1,
                        "name": "Hello-World",
                        "full_name": "octocat/Hello-World",
                        "owner": {"login": "octocat", "id": 1},
                    },
                )
            if path == "/repos/octocat/Hello-World/issues/50":
                return httpx.Response(
                    200,
                    json={
                        "id": 105,
                        "number": 50,
                        "title": "A pull request",
                        "state": "open",
                        "user": {"login": "alice", "id": 2},
                        "created_at": "2026-11-10T12:00:00Z",
                        "updated_at": "2026-11-10T12:00:00Z",
                        "pull_request": {
                            "url": "https://api.github.com/repos/octocat/Hello-World/pulls/50"
                        },
                    },
                )
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            service = IssueIngestionService(client)
            with pytest.raises(PullRequestNotAllowedError, match="is a Pull Request"):
                await service.ingest_issue(test_session, "octocat", "Hello-World", 50)

    async def test_ingest_duplicate_active_job_rejected(self, test_session: AsyncSession) -> None:
        # Pre-seed active job
        existing_job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/42",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=42,
            status=JobStatus.FETCHING_ISSUE,
        )
        test_session.add(existing_job)
        await test_session.commit()

        client = GitHubClient(token="ghp_test")
        service = IssueIngestionService(client)
        with pytest.raises(DuplicateJobError, match="An active job already exists"):
            await service.ingest_issue(test_session, "octocat", "Hello-World", 42)
