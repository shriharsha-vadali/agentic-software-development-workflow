"""
tests/unit/test_orchestrator.py
-------------------------------
Unit tests for JobStateMachine and JobOrchestrator (app/orchestrator/).
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.github import GitHubClient
from app.adapters.llm import MockLLMClient
from app.db.models import Base, Job, JobStatus, Plan
from app.intelligence.workspace import WorkspaceManager
from app.orchestrator.state_machine import InvalidTransitionError, JobStateMachine
from app.orchestrator.workflow import JobOrchestrator
from app.services.approval import UnapprovedPlanExecutionError

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_session_factory():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class TestJobStateMachine:
    """Verify state transition rules and terminal locking."""

    def test_valid_transitions(self) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            status=JobStatus.PENDING,
        )
        # PENDING -> FETCHING_ISSUE
        JobStateMachine.transition(job, JobStatus.FETCHING_ISSUE)
        assert job.status == JobStatus.FETCHING_ISSUE

        # FETCHING_ISSUE -> INDEXING_CODEBASE
        JobStateMachine.transition(job, JobStatus.INDEXING_CODEBASE)
        assert job.status == JobStatus.INDEXING_CODEBASE

        # INDEXING_CODEBASE -> PLANNING
        JobStateMachine.transition(job, JobStatus.PLANNING)
        assert job.status == JobStatus.PLANNING

        # PLANNING -> AWAITING_APPROVAL
        JobStateMachine.transition(job, JobStatus.AWAITING_APPROVAL)
        assert job.status == JobStatus.AWAITING_APPROVAL

    def test_invalid_transitions_blocked(self) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            status=JobStatus.PENDING,
        )
        # Cannot jump PENDING -> GENERATING_CODE
        with pytest.raises(InvalidTransitionError, match="Cannot transition job"):
            JobStateMachine.transition(job, JobStatus.GENERATING_CODE)

        # Cannot jump COMPLETED -> anything
        job.status = JobStatus.COMPLETED
        with pytest.raises(InvalidTransitionError):
            JobStateMachine.transition(job, JobStatus.PENDING)


class TestJobOrchestrator:
    """Verify orchestrator execution flow and approval gates."""

    async def test_run_planning_phase_pauses_at_approval(
        self, test_session_factory, tmp_path
    ) -> None:
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
            if path == "/repos/octocat/Hello-World/issues/42":
                return httpx.Response(
                    200,
                    json={
                        "id": 10,
                        "number": 42,
                        "title": "Add feature",
                        "body": "Need feature",
                        "state": "open",
                        "user": {"login": "octocat", "id": 1},
                        "created_at": "2026-11-10T12:00:00Z",
                        "updated_at": "2026-11-10T12:00:00Z",
                    },
                )
            if path == "/repos/octocat/Hello-World/issues/42/comments":
                return httpx.Response(200, json=[])
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            llm = MockLLMClient(
                predefined_responses={
                    "GitHub Issue Information": {
                        "summary": "Plan summary",
                        "problem_statement": "Problem",
                        "technical_approach": "Approach",
                        "steps": [
                            {"step_number": 1, "description": "Step 1", "file_paths": ["main.py"]}
                        ],
                        "test_plan": "Test plan",
                    }
                }
            )
            ws_mgr = WorkspaceManager(base_workspaces_dir=tmp_path)
            orchestrator = JobOrchestrator(
                session_factory=test_session_factory,
                github_client=client,
                llm_client=llm,
                workspace_manager=ws_mgr,
            )

            # Create initial Job
            async with test_session_factory() as db:
                job = Job(
                    issue_url="https://github.com/octocat/Hello-World/issues/42",
                    repo_owner="octocat",
                    repo_name="Hello-World",
                    issue_number=42,
                    status=JobStatus.PENDING,
                )
                db.add(job)
                await db.commit()
                job_id = job.id

            # Run planning phase
            paused_job = await orchestrator.run_planning_phase(job_id)

            assert paused_job.status == JobStatus.AWAITING_APPROVAL
            assert paused_job.plan is not None
            assert paused_job.plan.status == "draft"

    async def test_execution_phase_fails_if_unapproved(
        self, test_session_factory, tmp_path
    ) -> None:
        async with test_session_factory() as db:
            job = Job(
                issue_url="https://github.com/octocat/Hello-World/issues/42",
                repo_owner="octocat",
                repo_name="Hello-World",
                issue_number=42,
                status=JobStatus.AWAITING_APPROVAL,
            )
            db.add(job)
            await db.flush()
            plan = Plan(job_id=job.id, raw_text="Draft", status="draft")
            db.add(plan)
            await db.commit()
            job_id = job.id

        client = GitHubClient(token="ghp_test")
        orchestrator = JobOrchestrator(
            session_factory=test_session_factory,
            github_client=client,
            workspace_manager=WorkspaceManager(base_workspaces_dir=tmp_path),
        )

        with pytest.raises(UnapprovedPlanExecutionError):
            await orchestrator.run_execution_phase(job_id)

    async def test_cancel_job(self, test_session_factory, tmp_path) -> None:
        async with test_session_factory() as db:
            job = Job(
                issue_url="https://github.com/octocat/Hello-World/issues/42",
                repo_owner="octocat",
                repo_name="Hello-World",
                issue_number=42,
                status=JobStatus.PLANNING,
            )
            db.add(job)
            await db.commit()
            job_id = job.id

        client = GitHubClient(token="ghp_test")
        orchestrator = JobOrchestrator(
            session_factory=test_session_factory,
            github_client=client,
            workspace_manager=WorkspaceManager(base_workspaces_dir=tmp_path),
        )

        cancelled_job = await orchestrator.cancel_job(job_id)
        assert cancelled_job.status == JobStatus.FAILED
        assert "cancelled by user" in (cancelled_job.error_message or "")

    async def test_recover_interrupted_jobs(self, test_session_factory, tmp_path) -> None:
        async with test_session_factory() as db:
            # Seed an interrupted job (e.g. process killed during GENERATING_CODE)
            job = Job(
                issue_url="https://github.com/octocat/Hello-World/issues/42",
                repo_owner="octocat",
                repo_name="Hello-World",
                issue_number=42,
                status=JobStatus.GENERATING_CODE,
            )
            db.add(job)
            await db.commit()

        client = GitHubClient(token="ghp_test")
        orchestrator = JobOrchestrator(
            session_factory=test_session_factory,
            github_client=client,
            workspace_manager=WorkspaceManager(base_workspaces_dir=tmp_path),
        )

        count = await orchestrator.recover_interrupted_jobs()
        assert count == 1

        async with test_session_factory() as db:
            refreshed = (await db.execute(Job.__table__.select())).first()
            assert refreshed.status == "FAILED"
            assert "Safe recovery applied" in refreshed.error_message
