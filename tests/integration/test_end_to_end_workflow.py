"""
tests/integration/test_end_to_end_workflow.py
---------------------------------------------
Comprehensive end-to-end integration test of the complete
Agentic Software Development Workflow using deterministic mock providers.

Simulates:
GitHub Issue
    ↓
Issue Understanding & Ingestion
    ↓
Repository Analysis & Codebase Indexing
    ↓
Context Retrieval (RAG)
    ↓
AI Implementation Planning
    ↓
Mandatory Human Approval Gate
    ↓
AI Code Generation
    ↓
Isolated Subprocess Test Execution & Repair
    ↓
PR Authorization Gate
    ↓
Git Branch, Commit & Pull Request Submission
    ↓
COMPLETED State
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.adapters.github import GitHubClient
from app.adapters.llm import MockLLMClient
from app.db.models import Base, JobStatus
from app.intelligence.workspace import WorkspaceManager
from app.orchestrator.workflow import JobOrchestrator
from app.services.approval import UnapprovedPlanExecutionError


@pytest.fixture
async def e2e_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class TestEndToEndWorkflow:
    """Full lifecycle integration tests."""

    async def test_full_pipeline_happy_path(self, e2e_session_factory, tmp_path) -> None:
        """Complete workflow from issue ingestion to pull request creation."""

        # 1. Setup Mock GitHub API
        def github_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/repos/octocat/calculator":
                return httpx.Response(
                    200,
                    json={
                        "id": 1,
                        "name": "calculator",
                        "full_name": "octocat/calculator",
                        "default_branch": "main",
                        "owner": {"login": "octocat", "id": 1},
                    },
                )
            if path == "/repos/octocat/calculator/issues/101":
                return httpx.Response(
                    200,
                    json={
                        "id": 501,
                        "number": 101,
                        "title": "Add subtract function to calculator",
                        "body": "Please implement a subtract(a, b) function that returns a - b.",
                        "state": "open",
                        "user": {"login": "octocat", "id": 1},
                        "created_at": "2026-11-10T12:00:00Z",
                        "updated_at": "2026-11-10T12:00:00Z",
                        "html_url": "https://github.com/octocat/calculator/issues/101",
                    },
                )
            if path == "/repos/octocat/calculator/issues/101/comments":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": 1,
                            "user": {"login": "reviewer", "id": 2},
                            "body": "Make sure to include unit tests in tests/test_calc.py",
                            "created_at": "2026-11-10T12:10:00Z",
                            "updated_at": "2026-11-10T12:10:00Z",
                        }
                    ],
                )
            if path == "/repos/octocat/calculator/pulls":
                return httpx.Response(
                    201,
                    json={
                        "id": 888,
                        "number": 12,
                        "title": "fix(calculator): resolve #101",
                        "state": "open",
                        "html_url": "https://github.com/octocat/calculator/pull/12",
                    },
                )
            return httpx.Response(404)

        # 2. Setup Mock LLM Client
        mock_plan = {
            "summary": "Implement subtract function and unit tests",
            "problem_statement": "Calculator is missing subtract capability",
            "technical_approach": "Create calc.py with subtract(a, b) and tests/test_calc.py",
            "files_to_create": ["calc.py", "tests/test_calc.py"],
            "files_to_modify": [],
            "files_to_delete": [],
            "steps": [
                {"step_number": 1, "description": "Implement subtract", "file_paths": ["calc.py"]},
                {
                    "step_number": 2,
                    "description": "Add tests",
                    "file_paths": ["tests/test_calc.py"],
                },
            ],
            "test_plan": "Run pytest tests/test_calc.py",
            "risks": "None",
            "estimated_complexity": "low",
        }

        mock_code = {
            "summary": "Added subtract function and unit test",
            "changes": [
                {
                    "path": "calc.py",
                    "action": "create",
                    "content": "def subtract(a: int, b: int) -> int:\n    return a - b\n",
                },
                {
                    "path": "tests/test_calc.py",
                    "action": "create",
                    "content": (
                        "from calc import subtract\n\n"
                        "def test_subtract():\n    assert subtract(10, 4) == 6\n"
                    ),
                },
            ],
        }

        llm_client = MockLLMClient(
            predefined_responses={
                "GitHub Issue Information": mock_plan,
                "Approved Implementation Plan": mock_code,
            }
        )

        ws_mgr = WorkspaceManager(base_workspaces_dir=tmp_path)

        async with httpx.AsyncClient(transport=httpx.MockTransport(github_handler)) as mock_http:
            gh_client = GitHubClient(token="ghp_test", http_client=mock_http)
            orchestrator = JobOrchestrator(
                session_factory=e2e_session_factory,
                github_client=gh_client,
                llm_client=llm_client,
                workspace_manager=ws_mgr,
            )

            # PHASE 1: Issue Ingestion & Job Creation
            async with e2e_session_factory() as db:
                from app.services.issue_ingestion import IssueIngestionService

                ingest_svc = IssueIngestionService(gh_client)
                job, normalized = await ingest_svc.ingest_issue(db, "octocat", "calculator", 101)
                job_id = job.id
                assert job.status == JobStatus.PENDING

            # PHASE 2: Planning Phase (Indexing, Retrieval, Plan Generation)
            paused_job = await orchestrator.run_planning_phase(job_id)
            assert paused_job.status == JobStatus.AWAITING_APPROVAL
            assert paused_job.plan is not None
            assert paused_job.plan.status == "draft"

            # Check that an unapproved execution attempt is strictly rejected!
            with pytest.raises(UnapprovedPlanExecutionError):
                await orchestrator.run_execution_phase(job_id)

            # PHASE 3: Human Approval Gate
            async with e2e_session_factory() as db:
                from app.services.approval import PlanApprovalService

                approved_job = await PlanApprovalService.review_plan(
                    db, job_id, decision="approved"
                )
                assert approved_job.status == JobStatus.GENERATING_CODE
                assert approved_job.plan.status == "approved"

            # PHASE 4: Execution Phase (Code Generation, Sandbox Pytest Execution)
            executed_job = await orchestrator.run_execution_phase(job_id)
            assert executed_job.status == JobStatus.OPENING_PR

            # Verify diff and test_output artifacts were persisted
            art_types = [a.artifact_type for a in executed_job.artifacts]
            assert "diff" in art_types
            assert "test_output" in art_types

            # PHASE 5: Human PR Authorization Gate & PR Submission
            completed_job = await orchestrator.complete_pull_request(job_id)
            assert completed_job.status == JobStatus.COMPLETED

            # Verify PR URL artifact
            pr_arts = [a for a in completed_job.artifacts if a.artifact_type == "pr_url"]
            assert len(pr_arts) == 1
            assert "https://github.com/octocat/calculator/pull/12" in pr_arts[0].content

    async def test_plan_rejection_and_revision_flow(self, e2e_session_factory, tmp_path) -> None:
        """Plan rejected by human -> revised by AI -> approved -> completed."""

        def github_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/repos/octocat/service":
                return httpx.Response(
                    200,
                    json={
                        "id": 1,
                        "name": "service",
                        "full_name": "octocat/service",
                        "owner": {"login": "octocat", "id": 1},
                    },
                )
            if path == "/repos/octocat/service/issues/5":
                return httpx.Response(
                    200,
                    json={
                        "id": 5,
                        "number": 5,
                        "title": "Use async db engine",
                        "body": "Migrate to async engine",
                        "state": "open",
                        "user": {"login": "octocat", "id": 1},
                        "created_at": "2026-11-10T12:00:00Z",
                        "updated_at": "2026-11-10T12:00:00Z",
                    },
                )
            if path == "/repos/octocat/service/issues/5/comments":
                return httpx.Response(200, json=[])
            return httpx.Response(404)

        mock_plan = {
            "summary": "Async DB plan",
            "problem_statement": "Need async",
            "technical_approach": "Use aiosqlite",
            "steps": [{"step_number": 1, "description": "Step 1", "file_paths": ["db.py"]}],
            "test_plan": "Unit tests",
        }

        llm_client = MockLLMClient(predefined_responses={"GitHub Issue Information": mock_plan})
        ws_mgr = WorkspaceManager(base_workspaces_dir=tmp_path)

        async with httpx.AsyncClient(transport=httpx.MockTransport(github_handler)) as mock_http:
            gh_client = GitHubClient(token="ghp_test", http_client=mock_http)
            orchestrator = JobOrchestrator(
                session_factory=e2e_session_factory,
                github_client=gh_client,
                llm_client=llm_client,
                workspace_manager=ws_mgr,
            )

            # Ingest & Plan
            async with e2e_session_factory() as db:
                from app.services.issue_ingestion import IssueIngestionService

                job, _ = await IssueIngestionService(gh_client).ingest_issue(
                    db, "octocat", "service", 5
                )
                job_id = job.id

            await orchestrator.run_planning_phase(job_id)

            # Human rejects plan with feedback
            async with e2e_session_factory() as db:
                from app.services.approval import PlanApprovalService

                rejected_job = await PlanApprovalService.review_plan(
                    db,
                    job_id,
                    decision="rejected",
                    feedback="Please use asyncpg instead of aiosqlite",
                )
                assert rejected_job.status == JobStatus.PLANNING
                assert rejected_job.plan.status == "rejected"
                assert "asyncpg" in (rejected_job.plan.human_feedback or "")

            # Re-run planning phase: regenerates plan with feedback and returns to AWAITING_APPROVAL
            revised_job = await orchestrator.run_planning_phase(job_id)
            assert revised_job.status == JobStatus.AWAITING_APPROVAL
            assert revised_job.plan.status == "draft"
