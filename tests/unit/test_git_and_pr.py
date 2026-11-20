"""
tests/unit/test_git_and_pr.py
-----------------------------
Unit tests for GitWorkspaceService and PullRequestService (app/services/).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.github import GitHubClient
from app.db.models import Base, Job, JobStatus, Plan
from app.services.git_service import GitServiceError, GitWorkspaceService
from app.services.pr_service import PRAuthorizationError, PullRequestService

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


class TestGitWorkspaceService:
    """Verify git operations inside an isolated workspace."""

    async def test_git_init_branch_commit_and_diff(self, tmp_path: Path) -> None:
        service = GitWorkspaceService()
        ws = tmp_path / "job_git"
        ws.mkdir()

        # 1. Init
        await service.init_or_verify_repo(ws)
        assert (ws / ".git").exists()

        # 2. Branch
        branch = await service.create_feature_branch(ws, "agent/issue-42")
        assert branch == "agent/issue-42"

        # 3. Create file and commit
        (ws / "file.txt").write_text("hello world\n", encoding="utf-8")
        sha = await service.stage_and_commit(ws, "feat: add file.txt")
        assert len(sha) == 40  # Full git SHA

        # 4. Modify and get diff
        (ws / "file.txt").write_text("hello world modified\n", encoding="utf-8")
        diff = await service.get_diff(ws)
        assert "+hello world modified" in diff

    async def test_git_operations_bypass_repository_hooks(self, tmp_path: Path) -> None:
        """Verify that repository-controlled git hooks are never executed."""
        service = GitWorkspaceService()
        ws = tmp_path / "job_hook_test"
        ws.mkdir()

        await service.init_or_verify_repo(ws)

        # Plant a malicious pre-commit hook that creates a canary file or would exit 1
        hooks_dir = ws / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        canary_file = ws / "MALICIOUS_HOOK_EXECUTED.txt"
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_hook.write_text(
            f"#!/bin/sh\necho 'hacked' > '{canary_file.as_posix()}'\nexit 1\n",
            encoding="utf-8",
        )

        # Stage and commit
        (ws / "safe_file.txt").write_text("secure content\n", encoding="utf-8")
        sha = await service.stage_and_commit(ws, "feat: commit with hook present")

        # Verify commit succeeded
        assert len(sha) == 40
        # Verify hook was NOT executed
        assert not canary_file.exists(), "Security breach: repository pre-commit hook was executed!"

    def test_git_workspace_service_unique_hooks_directory(self) -> None:
        """Each GitWorkspaceService instance gets a dedicated unique hooks directory."""
        service1 = GitWorkspaceService()
        service2 = GitWorkspaceService()

        assert service1.empty_hooks_dir != service2.empty_hooks_dir
        assert service1.empty_hooks_dir.exists()
        assert service2.empty_hooks_dir.exists()
        assert list(service1.empty_hooks_dir.iterdir()) == []
        assert list(service2.empty_hooks_dir.iterdir()) == []

        cmd1 = service1._build_git_cmd(["status"])
        cmd2 = service2._build_git_cmd(["status"])

        assert f"core.hooksPath={service1.empty_hooks_dir.as_posix()}" in cmd1
        assert f"core.hooksPath={service2.empty_hooks_dir.as_posix()}" in cmd2

    def test_git_workspace_service_mkdtemp_failure(self) -> None:
        """Failure to create private empty hooks directory fails closed with GitServiceError."""
        with patch("tempfile.mkdtemp", side_effect=OSError("Read-only filesystem")):
            with pytest.raises(
                GitServiceError, match="Failed to create secure empty git hooks directory"
            ):
                GitWorkspaceService()


class TestPullRequestService:
    """Verify PR creation authorization gates and GitHub API coordination."""

    async def test_create_pr_success(self, test_session: AsyncSession) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/pulls"
            return httpx.Response(
                201,
                json={
                    "id": 900,
                    "number": 15,
                    "title": "fix(Hello-World): resolve #42",
                    "body": "Fixes #42",
                    "state": "open",
                    "html_url": "https://github.com/octocat/Hello-World/pull/15",
                },
            )

        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/42",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=42,
            status=JobStatus.OPENING_PR,
        )
        test_session.add(job)
        await test_session.flush()

        plan = Plan(job_id=job.id, raw_text="Approved plan content", status="approved")
        test_session.add(plan)
        await test_session.commit()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            pr_service = PullRequestService(client)

            pr = await pr_service.create_pull_request_for_job(
                test_session, job.id, feature_branch="agent/issue-42"
            )

        assert pr.number == 15
        assert pr.html_url == "https://github.com/octocat/Hello-World/pull/15"

        # Job must be transitioned to COMPLETED
        await test_session.refresh(job)
        assert job.status == JobStatus.COMPLETED

    async def test_create_pr_unauthorized_state_rejected(self, test_session: AsyncSession) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/42",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=42,
            status=JobStatus.GENERATING_CODE,
        )
        test_session.add(job)
        await test_session.commit()

        client = GitHubClient(token="ghp_test")
        pr_service = PullRequestService(client)

        with pytest.raises(PRAuthorizationError, match="Job must be in 'OPENING_PR'"):
            await pr_service.create_pull_request_for_job(
                test_session, job.id, feature_branch="agent/issue-42"
            )
