"""
tests/unit/test_workflow_authorization.py
-----------------------------------------
Comprehensive verification for Phase 2 - Task 2, Milestone 2:
Workflow Authorization & Approval Enforcement.

Verifies that authenticated requests and direct service/orchestrator calls cannot bypass:
1. Workflow state restrictions and invalid state transitions
2. Mandatory human plan approval checkpoint
3. Explicit human PR authorization checkpoint
4. Rejection of repeated plan approvals and duplicate PR submissions
5. Terminal state locking (COMPLETED and FAILED cannot be restarted, executed, or cancelled)
6. Service and orchestrator layer enforcement independent of API routes
7. Full valid approved workflow end-to-end progression
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.adapters.github import GitHubClient, GitHubPullRequest
from app.db.models import Artifact, Base, Job, JobStatus, Plan
from app.db.session import _build_engine, get_db
from app.main import app
from app.orchestrator.state_machine import (
    JobStateMachine,
)
from app.orchestrator.workflow import JobOrchestrator
from app.services.approval import (
    PlanApprovalService,
    UnapprovedPlanExecutionError,
)
from app.services.pr_service import (
    PRAuthorizationError,
    PullRequestService,
)
from app.services.workflow_auth import (
    DuplicateActionError,
    PlanApprovalRequiredError,
    PRAuthorizationRequiredError,
    UnauthorizedActionError,
    WorkflowAuthorizationError,
    WorkflowAuthorizationPolicy,
    extract_test_execution_exit_code,
    get_latest_test_output_artifact,
    parse_test_execution_content,
)

# =============================================================================
# Database and HTTP Client Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_db(tmp_path: Path) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create an isolated SQLite database file per test and clean up on teardown."""
    db_file = tmp_path / "test_workflow_auth.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    engine: AsyncEngine = _build_engine(db_url)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    yield session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(
    test_db: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide an AsyncClient configured with test database dependency override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with test_db() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# =============================================================================
# Helpers
# =============================================================================

_issue_counter = itertools.count(1000)


async def _seed_job(
    session_factory: async_sessionmaker[AsyncSession],
    status: JobStatus = JobStatus.PENDING,
    plan_status: str | None = None,
    artifacts: list[tuple[str, str]] | None = None,
) -> int:
    """Helper to seed a Job with unique issue_number, optional plan and artifacts."""
    issue_num = next(_issue_counter)
    async with session_factory() as session:
        job = Job(
            issue_url=f"https://github.com/octocat/Hello-World/issues/{issue_num}",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=issue_num,
            status=status,
        )
        session.add(job)
        await session.flush()

        if plan_status is not None:
            plan = Plan(
                job_id=job.id,
                raw_text="# Implementation Plan\nFix the issue safely.",
                status=plan_status,
            )
            session.add(plan)

        if artifacts:
            for art_type, art_content in artifacts:
                session.add(
                    Artifact(
                        job_id=job.id,
                        artifact_type=art_type,
                        content=art_content,
                    )
                )

        await session.commit()
        return job.id


# =============================================================================
# 1. API Actions in Invalid Job States
# =============================================================================


class TestAPIInvalidJobStates:
    """Verify that authenticated requests cannot perform actions in invalid job states."""

    async def test_start_job_invalid_states_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Cannot start planning on jobs that already passed planning or are terminal."""
        for disallowed_status in (
            JobStatus.AWAITING_APPROVAL,
            JobStatus.GENERATING_CODE,
            JobStatus.RUNNING_TESTS,
            JobStatus.OPENING_PR,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ):
            job_id = await _seed_job(test_db, status=disallowed_status)
            response = await client.post(f"/jobs/{job_id}/start")
            assert response.status_code == 409, f"Failed for {disallowed_status}"
            assert "Cannot start planning" in response.json()["detail"]

    async def test_approve_plan_invalid_states_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Cannot approve plan on jobs not in AWAITING_APPROVAL."""
        for disallowed_status in (
            JobStatus.PENDING,
            JobStatus.FETCHING_ISSUE,
            JobStatus.INDEXING_CODEBASE,
            JobStatus.PLANNING,
            JobStatus.GENERATING_CODE,
            JobStatus.RUNNING_TESTS,
            JobStatus.OPENING_PR,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ):
            job_id = await _seed_job(test_db, status=disallowed_status, plan_status="draft")
            response = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
            assert response.status_code == 409, f"Failed for {disallowed_status}"
            assert "Cannot review plan for job in" in response.json()["detail"]

    async def test_execute_job_invalid_states_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Cannot execute code generation on jobs not in GENERATING_CODE."""
        for disallowed_status in (
            JobStatus.PENDING,
            JobStatus.PLANNING,
            JobStatus.AWAITING_APPROVAL,
            JobStatus.RUNNING_TESTS,
            JobStatus.OPENING_PR,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ):
            job_id = await _seed_job(test_db, status=disallowed_status, plan_status="approved")
            response = await client.post(f"/jobs/{job_id}/execute")
            assert response.status_code == 409, f"Failed for {disallowed_status}"
            assert "Cannot execute code generation" in response.json()["detail"]

    async def test_approve_pr_invalid_states_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Cannot approve PR on jobs not in OPENING_PR."""
        for disallowed_status in (
            JobStatus.PENDING,
            JobStatus.AWAITING_APPROVAL,
            JobStatus.GENERATING_CODE,
            JobStatus.RUNNING_TESTS,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ):
            job_id = await _seed_job(test_db, status=disallowed_status, plan_status="approved")
            response = await client.post(f"/jobs/{job_id}/approve-pr")
            assert response.status_code == 409, f"Failed for {disallowed_status}"
            assert "Cannot create PR" in response.json()["detail"]

    async def test_cancel_job_terminal_states_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Cannot cancel jobs that are already in COMPLETED or FAILED terminal states."""
        for terminal_status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job_id = await _seed_job(test_db, status=terminal_status)
            response = await client.post(f"/jobs/{job_id}/cancel")
            assert response.status_code == 409
            assert "already in terminal state" in response.json()["detail"]


# =============================================================================
# 2. Execution Blocked Before Plan Approval
# =============================================================================


class TestExecutionBlockedWithoutPlanApproval:
    """Verify execution is strictly prevented until human plan approval."""

    async def test_execute_with_draft_plan_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Execution requires plan status == 'approved', rejecting draft plans."""
        job_id = await _seed_job(test_db, status=JobStatus.GENERATING_CODE, plan_status="draft")
        response = await client.post(f"/jobs/{job_id}/execute")
        assert response.status_code == 409
        assert "implementation plan has not been approved" in response.json()["detail"]

    async def test_execute_with_rejected_plan_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Execution strictly rejects rejected plans."""
        job_id = await _seed_job(test_db, status=JobStatus.GENERATING_CODE, plan_status="rejected")
        response = await client.post(f"/jobs/{job_id}/execute")
        assert response.status_code == 409
        assert "implementation plan has not been approved" in response.json()["detail"]

    async def test_execute_without_plan_rejected(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Execution rejects jobs lacking a plan record."""
        job_id = await _seed_job(test_db, status=JobStatus.GENERATING_CODE, plan_status=None)
        response = await client.post(f"/jobs/{job_id}/execute")
        assert response.status_code == 409

    def test_plan_approval_service_assert_can_generate_code_direct(self) -> None:
        """PlanApprovalService.assert_can_generate_code fails closed directly."""
        unapproved_job = Job(
            id=1,
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            status=JobStatus.GENERATING_CODE,
        )
        # Missing plan
        with pytest.raises(UnapprovedPlanExecutionError, match="Security violation"):
            PlanApprovalService.assert_can_generate_code(unapproved_job)

        # Draft plan
        unapproved_job.plan = Plan(job_id=1, raw_text="Draft", status="draft")
        with pytest.raises(UnapprovedPlanExecutionError, match="Security violation"):
            PlanApprovalService.assert_can_generate_code(unapproved_job)

        # Status not GENERATING_CODE
        unapproved_job.status = JobStatus.AWAITING_APPROVAL
        unapproved_job.plan.status = "approved"
        with pytest.raises(UnapprovedPlanExecutionError, match="Security violation"):
            PlanApprovalService.assert_can_generate_code(unapproved_job)


# =============================================================================
# 3. PR Creation Blocked Without Explicit Authorization
# =============================================================================


class TestPRCreationBlockedWithoutAuthorization:
    """Verify Pull Request creation strictly requires explicit human PR authorization."""

    async def test_pr_creation_requires_opening_pr_status(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Attempting to trigger PR creation while still in RUNNING_TESTS returns 409."""
        job_id = await _seed_job(test_db, status=JobStatus.RUNNING_TESTS, plan_status="approved")
        response = await client.post(f"/jobs/{job_id}/approve-pr")
        assert response.status_code == 409
        assert "OPENING_PR" in response.json()["detail"]

    async def test_pr_creation_requires_approved_plan(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Direct call to PullRequestService fails if plan is not approved."""
        job_id = await _seed_job(test_db, status=JobStatus.OPENING_PR, plan_status="draft")
        pr_service = PullRequestService(GitHubClient(token="ghp_test"))

        async with test_db() as session:
            with pytest.raises(PRAuthorizationError, match="Plan has not been approved"):
                await pr_service.create_pull_request_for_job(
                    session, job_id, feature_branch="agent/feature"
                )

    def test_workflow_policy_authorize_pr_creation_fails_closed(self) -> None:
        """WorkflowAuthorizationPolicy.authorize_pr_creation enforces gate rules."""
        job = Job(
            id=5,
            issue_url="https://github.com/octocat/Hello-World/issues/5",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=5,
            status=JobStatus.RUNNING_TESTS,
        )
        with pytest.raises(UnauthorizedActionError, match="OPENING_PR"):
            WorkflowAuthorizationPolicy.authorize_pr_creation(job)

        job.status = JobStatus.OPENING_PR
        job.plan = None
        with pytest.raises(PlanApprovalRequiredError, match="Plan has not been approved"):
            WorkflowAuthorizationPolicy.authorize_pr_creation(job)

    async def test_successful_pr_creation_transitions_to_completed_with_artifact(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """PR creation persists PR artifact and transitions via state machine to COMPLETED."""
        job_id = await _seed_job(test_db, status=JobStatus.OPENING_PR, plan_status="approved")
        github_client = GitHubClient(token="ghp_test")
        pr_service = PullRequestService(github_client)

        mock_pr = GitHubPullRequest(
            id=101,
            number=42,
            title="fix(Hello-World): resolve #100",
            html_url="https://github.com/octocat/Hello-World/pull/42",
        )
        with patch.object(github_client, "create_pull_request", return_value=mock_pr):
            async with test_db() as session:
                created_pr = await pr_service.create_pull_request_for_job(
                    session, job_id, feature_branch="agent/feature"
                )
                assert created_pr.number == 42

            async with test_db() as session:
                stmt = select(Job).options(selectinload(Job.artifacts)).where(Job.id == job_id)
                res = await session.execute(stmt)
                job = res.scalars().one()
                assert job.status == JobStatus.COMPLETED
                pr_artifacts = [a for a in job.artifacts if a.artifact_type == "pr_url"]
                assert len(pr_artifacts) == 1
                assert pr_artifacts[0].content == "https://github.com/octocat/Hello-World/pull/42"

    def test_transition_to_completed_without_pr_artifact_rejected(self) -> None:
        """Attempting to transition to COMPLETED without a PR artifact is rejected."""
        job = Job(
            id=55,
            issue_url="https://github.com/octocat/Hello-World/issues/55",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=55,
            status=JobStatus.OPENING_PR,
            artifacts=[],
        )
        with pytest.raises(PRAuthorizationRequiredError, match="Pull Request must be authorized"):
            JobStateMachine.transition(job, JobStatus.COMPLETED)

    async def test_failed_pr_creation_does_not_leave_job_completed(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Failed PR creation (API error) rolls back cleanly and leaves job in OPENING_PR."""
        job_id = await _seed_job(test_db, status=JobStatus.OPENING_PR, plan_status="approved")
        github_client = GitHubClient(token="ghp_test")
        pr_service = PullRequestService(github_client)

        with patch.object(
            github_client, "create_pull_request", side_effect=RuntimeError("GitHub API timeout")
        ):
            async with test_db() as session:
                with pytest.raises(RuntimeError, match="GitHub API timeout"):
                    await pr_service.create_pull_request_for_job(
                        session, job_id, feature_branch="agent/feature"
                    )

            async with test_db() as session:
                stmt = select(Job).options(selectinload(Job.artifacts)).where(Job.id == job_id)
                res = await session.execute(stmt)
                job = res.scalars().one()
                assert job.status == JobStatus.OPENING_PR
                pr_artifacts = [a for a in job.artifacts if a.artifact_type == "pr_url"]
                assert len(pr_artifacts) == 0

    def test_direct_imports_no_circular_dependencies(self) -> None:
        """Verify that all core workflow and service modules import cleanly without circularity."""
        import importlib

        modules = [
            "app.orchestrator.state_machine",
            "app.services.workflow_auth",
            "app.services.approval",
            "app.services.pr_service",
            "app.orchestrator.workflow",
        ]
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None


# =============================================================================
# 4. Repeated Plan Approval Rejection
# =============================================================================


class TestRepeatedPlanApproval:
    """Verify that repeated plan approval requests are safely rejected."""

    async def test_cannot_approve_already_approved_job(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Once approved and transitioned to GENERATING_CODE, repeated approval returns 409."""
        job_id = await _seed_job(test_db, status=JobStatus.AWAITING_APPROVAL, plan_status="draft")

        # First approval succeeds
        resp1 = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "GENERATING_CODE"

        # Second approval fails with 409 Conflict
        resp2 = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert resp2.status_code == 409
        assert "Cannot review plan for job in 'GENERATING_CODE' state" in resp2.json()["detail"]

    def test_workflow_policy_rejects_already_approved_plan(self) -> None:
        """WorkflowAuthorizationPolicy.authorize_plan_review rejects already approved plans."""
        job = Job(
            id=12,
            issue_url="https://github.com/octocat/Hello-World/issues/12",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=12,
            status=JobStatus.AWAITING_APPROVAL,
            plan=Plan(job_id=12, raw_text="Plan", status="approved"),
        )
        with pytest.raises(DuplicateActionError, match="already been approved"):
            WorkflowAuthorizationPolicy.authorize_plan_review(job, decision="approved")


# =============================================================================
# 5. Repeated PR Authorization / Duplicate PR Submission
# =============================================================================


class TestDuplicatePRSubmission:
    """Verify duplicate PR submissions for the same job are strictly rejected."""

    async def test_duplicate_pr_rejected_when_artifact_exists(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """If a PR URL artifact already exists, repeated approve-pr call is rejected with 409."""
        job_id = await _seed_job(
            test_db,
            status=JobStatus.OPENING_PR,
            plan_status="approved",
            artifacts=[("pr_url", "https://github.com/octocat/Hello-World/pull/15")],
        )
        response = await client.post(f"/jobs/{job_id}/approve-pr")
        assert response.status_code == 409
        assert "Pull Request has already been submitted" in response.json()["detail"]

    async def test_pr_service_rejects_duplicate_pr(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """PullRequestService rejects opening a second PR for the same job."""
        job_id = await _seed_job(
            test_db,
            status=JobStatus.OPENING_PR,
            plan_status="approved",
            artifacts=[("pr_url", "https://github.com/octocat/Hello-World/pull/99")],
        )
        pr_service = PullRequestService(GitHubClient(token="ghp_test"))

        async with test_db() as session:
            with pytest.raises(PRAuthorizationError, match="already been submitted"):
                await pr_service.create_pull_request_for_job(
                    session, job_id, feature_branch="agent/feature"
                )


# =============================================================================
# 6. Terminal State Locking (Cancelled/Failed Jobs Cannot Execute or Restart)
# =============================================================================


class TestTerminalStateLocking:
    """Verify completed and failed jobs are locked against further lifecycle operations."""

    @pytest.mark.parametrize("terminal_status", [JobStatus.COMPLETED, JobStatus.FAILED])
    async def test_terminal_jobs_reject_all_mutations(
        self,
        client: httpx.AsyncClient,
        test_db: async_sessionmaker[AsyncSession],
        terminal_status: JobStatus,
    ) -> None:
        job_id = await _seed_job(test_db, status=terminal_status, plan_status="approved")

        # 1. Cannot start planning
        r_start = await client.post(f"/jobs/{job_id}/start")
        assert r_start.status_code == 409

        # 2. Cannot approve plan
        r_app = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert r_app.status_code == 409

        # 3. Cannot execute
        r_exec = await client.post(f"/jobs/{job_id}/execute")
        assert r_exec.status_code == 409

        # 4. Cannot approve PR
        r_pr = await client.post(f"/jobs/{job_id}/approve-pr")
        assert r_pr.status_code == 409

        # 5. Cannot cancel
        r_cancel = await client.post(f"/jobs/{job_id}/cancel")
        assert r_cancel.status_code == 409

        # 6. Cannot transition via PATCH /status
        r_patch = await client.patch(f"/jobs/{job_id}/status", json={"status": "PENDING"})
        assert r_patch.status_code == 409


# =============================================================================
# 7. State Transition Rules and Gate Prerequisites
# =============================================================================


class TestStateTransitionGates:
    """Verify state machine transition gates and prerequisite enforcement."""

    async def test_cannot_skip_states_via_status_patch(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Skipping intermediate states is blocked by state machine transition rules."""
        job_id = await _seed_job(test_db, status=JobStatus.PENDING)

        # Cannot jump PENDING -> OPENING_PR
        resp = await client.patch(f"/jobs/{job_id}/status", json={"status": "OPENING_PR"})
        assert resp.status_code == 409

    async def test_patch_cannot_force_opening_pr_from_running_tests(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """SEC-03: External PATCH cannot force OPENING_PR from RUNNING_TESTS."""
        job_id = await _seed_job(test_db, status=JobStatus.RUNNING_TESTS, plan_status="approved")
        resp = await client.patch(f"/jobs/{job_id}/status", json={"status": "OPENING_PR"})
        assert resp.status_code == 409
        assert "External status update to 'OPENING_PR' is forbidden" in resp.json()["detail"]

    @pytest.mark.parametrize(
        ("initial_status", "protected_target"),
        [
            (JobStatus.AWAITING_APPROVAL, JobStatus.GENERATING_CODE),
            (JobStatus.GENERATING_CODE, JobStatus.RUNNING_TESTS),
            (JobStatus.RUNNING_TESTS, JobStatus.OPENING_PR),
            (JobStatus.OPENING_PR, JobStatus.COMPLETED),
        ],
    )
    async def test_patch_cannot_bypass_protected_transitions(
        self,
        client: httpx.AsyncClient,
        test_db: async_sessionmaker[AsyncSession],
        initial_status: JobStatus,
        protected_target: JobStatus,
    ) -> None:
        """External PATCH is blocked for all protected workflow transitions."""
        job_id = await _seed_job(test_db, status=initial_status, plan_status="approved")
        resp = await client.patch(f"/jobs/{job_id}/status", json={"status": protected_target.value})
        assert resp.status_code == 409
        assert (
            f"External status update to '{protected_target.value}' is forbidden"
            in resp.json()["detail"]
        )

    async def test_patch_allows_administrative_failure(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """External PATCH to FAILED remains permitted for administrative failure handling."""
        job_id = await _seed_job(test_db, status=JobStatus.RUNNING_TESTS, plan_status="approved")
        resp = await client.patch(
            f"/jobs/{job_id}/status",
            json={"status": "FAILED", "error_message": "Administrative abort"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FAILED"
        assert data["error_message"] == "Administrative abort"

    def test_internal_transition_to_opening_pr_requires_passing_tests(self) -> None:
        """JobStateMachine.transition enforces verified passing test output for OPENING_PR."""
        job = Job(
            id=77,
            issue_url="https://github.com/octocat/Hello-World/issues/77",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=77,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=77, raw_text="Plan", status="approved"),
            artifacts=[],
        )

        # 1. No test output artifact -> blocked
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

        # 2. Failing test output artifact (exit code 1) -> blocked
        failing_artifact = Artifact(
            id=1,
            job_id=77,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 1\n\n"
                "#### Output\n```\nFAILED\n```"
            ),
        )
        job.artifacts = [failing_artifact]
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

        # 3. Passing test output artifact (exit code 0) -> allowed
        passing_artifact = Artifact(
            id=2,
            job_id=77,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 2)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        job.artifacts = [passing_artifact]
        JobStateMachine.transition(job, JobStatus.OPENING_PR)
        assert job.status == JobStatus.OPENING_PR

    def test_opening_pr_fails_closed_when_artifacts_unloaded(self) -> None:
        """SEC-03A: Opening PR fails closed if artifacts relationship is unloaded or None."""
        job = Job(
            id=78,
            issue_url="https://github.com/octocat/Hello-World/issues/78",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=78,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=78, raw_text="Plan", status="approved"),
        )
        # Simulate unloaded relationship where 'artifacts' is not in __dict__
        if "artifacts" in job.__dict__:
            del job.__dict__["artifacts"]

        with pytest.raises(UnauthorizedActionError, match="test execution evidence is unavailable"):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

        # Simulate artifacts attribute explicitly None
        job.__dict__["artifacts"] = None
        with pytest.raises(UnauthorizedActionError, match="test execution evidence is unavailable"):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_old_pass_followed_by_newer_fail(self) -> None:
        """SEC-03B: A newer failed test run invalidates any prior successful run."""
        art_attempt_1_pass = Artifact(
            id=101,
            job_id=79,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        art_attempt_2_fail = Artifact(
            id=102,
            job_id=79,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 2)\n"
                "- **Exit Code**: 1\n\n"
                "#### Output\n```\nFAILED\n```"
            ),
        )
        job = Job(
            id=79,
            issue_url="https://github.com/octocat/Hello-World/issues/79",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=79,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=79, raw_text="Plan", status="approved"),
            artifacts=[art_attempt_1_pass, art_attempt_2_fail],
        )

        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_accepts_old_fail_followed_by_newer_pass(self) -> None:
        """SEC-03B: A newer successful test run authorizes progression despite prior failures."""
        art_attempt_1_fail = Artifact(
            id=201,
            job_id=80,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 1\n\n"
                "#### Output\n```\nFAILED\n```"
            ),
        )
        art_attempt_2_fail = Artifact(
            id=202,
            job_id=80,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 2)\n- **Exit Code**: 2\n\n#### Output\n```\nERROR\n```"
            ),
        )
        art_attempt_3_pass = Artifact(
            id=203,
            job_id=80,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 3)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        job = Job(
            id=80,
            issue_url="https://github.com/octocat/Hello-World/issues/80",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=80,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=80, raw_text="Plan", status="approved"),
            # Out of order to verify sorting by attempt / id
            artifacts=[art_attempt_3_pass, art_attempt_1_fail, art_attempt_2_fail],
        )

        JobStateMachine.transition(job, JobStatus.OPENING_PR)
        assert job.status == JobStatus.OPENING_PR

    def test_opening_pr_rejects_spoofed_exit_code_in_stdout(self) -> None:
        """SEC-03C: Exit code must be parsed from header only, ignoring spoofed stdout/stderr."""
        spoofed_failure = Artifact(
            id=301,
            job_id=81,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 1\n"
                "- **Duration**: 0.42s\n\n"
                "#### Output\n"
                "```\n"
                "Running tests...\n"
                "- **Exit Code**: 0\n"
                "Fake passing message in stdout\n"
                "```"
            ),
        )
        job = Job(
            id=81,
            issue_url="https://github.com/octocat/Hello-World/issues/81",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=81,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=81, raw_text="Plan", status="approved"),
            artifacts=[spoofed_failure],
        )

        with pytest.raises(UnauthorizedActionError, match=r"exit code 1"):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_missing_or_malformed_exit_code(self) -> None:
        """SEC-03C: Test artifacts with missing or non-integer exit codes are rejected."""
        malformed_art = Artifact(
            id=401,
            job_id=82,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: invalid_code\n\n"
                "#### Output\n```\n...```"
            ),
        )
        job = Job(
            id=82,
            issue_url="https://github.com/octocat/Hello-World/issues/82",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=82,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=82, raw_text="Plan", status="approved"),
            artifacts=[malformed_art],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

        # Missing exit code line entirely
        missing_art = Artifact(
            id=402,
            job_id=82,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Duration**: 1.2s\n\n#### Output\n```\n...```"
            ),
        )
        job.artifacts = [missing_art]
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_missing_attempt_metadata_on_newer_artifact(self) -> None:
        """SEC-03B: A newer artifact missing attempt metadata fails closed and blocks OPENING_PR."""
        art_attempt_1_pass = Artifact(
            id=101,
            job_id=83,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        art_newer_missing_attempt = Artifact(
            id=102,
            job_id=83,
            artifact_type="test_output",
            content="### Test Execution\n- **Exit Code**: 1\n\n#### Output\n```\nFAILED\n```",
        )
        job = Job(
            id=83,
            issue_url="https://github.com/octocat/Hello-World/issues/83",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=83,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=83, raw_text="Plan", status="approved"),
            artifacts=[art_attempt_1_pass, art_newer_missing_attempt],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_malformed_attempt_metadata(self) -> None:
        """SEC-03B: Malformed attempt metadata (e.g. non-numeric) fails closed."""
        malformed_attempt_art = Artifact(
            id=501,
            job_id=84,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt ABC)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        job = Job(
            id=84,
            issue_url="https://github.com/octocat/Hello-World/issues/84",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=84,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=84, raw_text="Plan", status="approved"),
            artifacts=[malformed_attempt_art],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_duplicate_attempt_headers(self) -> None:
        """SEC-03B: Duplicate attempt headers in an artifact fail closed."""
        dup_attempt_art = Artifact(
            id=601,
            job_id=85,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "### Test Execution (Attempt 2)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        job = Job(
            id=85,
            issue_url="https://github.com/octocat/Hello-World/issues/85",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=85,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=85, raw_text="Plan", status="approved"),
            artifacts=[dup_attempt_art],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_colliding_attempts_with_missing_ids(self) -> None:
        """SEC-03B: Multiple artifacts with same attempt and missing IDs fail closed."""
        art_a = Artifact(
            id=None,
            job_id=86,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 1\n\n"
                "#### Output\n```\nFAILED\n```"
            ),
        )
        art_b = Artifact(
            id=None,
            job_id=86,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 0\n\n"
                "#### Output\n```\nPASSED\n```"
            ),
        )
        job = Job(
            id=86,
            issue_url="https://github.com/octocat/Hello-World/issues/86",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=86,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=86, raw_text="Plan", status="approved"),
            artifacts=[art_a, art_b],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_duplicate_exit_code_fields(self) -> None:
        """SEC-03C: Duplicate exit-code fields in the metadata header are rejected."""
        dup_exit_code_art = Artifact(
            id=701,
            job_id=87,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 0\n"
                "- **Exit Code**: 1\n\n"
                "#### Output\n```\n...\n```"
            ),
        )
        job = Job(
            id=87,
            issue_url="https://github.com/octocat/Hello-World/issues/87",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=87,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=87, raw_text="Plan", status="approved"),
            artifacts=[dup_exit_code_art],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_missing_output_delimiter(self) -> None:
        """SEC-03C: Missing '#### Output' delimiter is rejected."""
        no_delim_art = Artifact(
            id=801,
            job_id=88,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Exit Code**: 0\nSome output without delimiter"
            ),
        )
        job = Job(
            id=88,
            issue_url="https://github.com/octocat/Hello-World/issues/88",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=88,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=88, raw_text="Plan", status="approved"),
            artifacts=[no_delim_art],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_rejects_missing_attempt_header(self) -> None:
        """SEC-03C: Missing attempt header is rejected."""
        no_attempt_header_art = Artifact(
            id=901,
            job_id=89,
            artifact_type="test_output",
            content=("- **Exit Code**: 0\n\n#### Output\n```\nPASSED\n```"),
        )
        job = Job(
            id=89,
            issue_url="https://github.com/octocat/Hello-World/issues/89",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=89,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=89, raw_text="Plan", status="approved"),
            artifacts=[no_attempt_header_art],
        )
        with pytest.raises(
            UnauthorizedActionError, match="automated tests have not completed successfully"
        ):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_opening_pr_accepts_canonical_exit_code_zero(self) -> None:
        """SEC-03C: Valid canonical artifact with exit code zero authorizes OPENING_PR."""
        valid_art = Artifact(
            id=1001,
            job_id=90,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 0\n"
                "- **Duration**: 1.23s\n\n"
                "#### Output\n```\nAll 10 tests passed!\n```"
            ),
        )
        job = Job(
            id=90,
            issue_url="https://github.com/octocat/Hello-World/issues/90",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=90,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=90, raw_text="Plan", status="approved"),
            artifacts=[valid_art],
        )
        JobStateMachine.transition(job, JobStatus.OPENING_PR)
        assert job.status == JobStatus.OPENING_PR

    def test_opening_pr_rejects_canonical_nonzero_exit_code(self) -> None:
        """SEC-03C: Canonical artifact with nonzero exit code blocks OPENING_PR."""
        failing_art = Artifact(
            id=1101,
            job_id=91,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n"
                "- **Exit Code**: 2\n"
                "- **Duration**: 0.5s\n\n"
                "#### Output\n```\nPytest exit code 2 (interrupted)\n```"
            ),
        )
        job = Job(
            id=91,
            issue_url="https://github.com/octocat/Hello-World/issues/91",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=91,
            status=JobStatus.RUNNING_TESTS,
            plan=Plan(job_id=91, raw_text="Plan", status="approved"),
            artifacts=[failing_art],
        )
        with pytest.raises(UnauthorizedActionError, match=r"exit code 2"):
            JobStateMachine.transition(job, JobStatus.OPENING_PR)

    def test_extract_test_execution_exit_code_unit(self) -> None:
        """SEC-03C: Unit tests for extract_test_execution_exit_code helper."""
        assert extract_test_execution_exit_code("") is None
        assert extract_test_execution_exit_code("invalid content") is None

        # Clean header with 0
        content_0 = "### Test Execution (Attempt 1)\n- **Exit Code**: 0\n#### Output\n```\nOK\n```"
        assert extract_test_execution_exit_code(content_0) == 0

        # Clean header with 1
        content_1 = (
            "### Test Execution (Attempt 1)\n- **Exit Code**: 1\n#### Output\n```\nFAIL\n```"
        )
        assert extract_test_execution_exit_code(content_1) == 1

        # Negative exit code
        content_neg = (
            "### Test Execution (Attempt 1)\n- **Exit Code**: -9\n#### Output\n```\nKILL\n```"
        )
        assert extract_test_execution_exit_code(content_neg) == -9

        # Exit code only in stdout, header missing
        content_spoof_only = "### Test Execution\n#### Output\n```\n- **Exit Code**: 0\n```"
        assert extract_test_execution_exit_code(content_spoof_only) is None

        # Malformed integer
        content_malformed = (
            "### Test Execution (Attempt 1)\n- **Exit Code**: 0x0\n#### Output\n```\nOK\n```"
        )
        assert extract_test_execution_exit_code(content_malformed) is None

        # Missing delimiter
        content_no_delim = "### Test Execution (Attempt 1)\n- **Exit Code**: 0\n```\nOK\n```"
        assert extract_test_execution_exit_code(content_no_delim) is None

        # Duplicate exit code lines
        content_dup_exit = (
            "### Test Execution (Attempt 1)\n"
            "- **Exit Code**: 0\n"
            "- **Exit Code**: 1\n"
            "#### Output\n```\nOK\n```"
        )
        assert extract_test_execution_exit_code(content_dup_exit) is None

    def test_get_latest_test_output_artifact_unit(self) -> None:
        """SEC-03B: Unit tests for get_latest_test_output_artifact helper."""
        assert get_latest_test_output_artifact(None) is None
        assert get_latest_test_output_artifact([]) is None

        # Only non-test artifacts
        diff_art = Artifact(id=1, artifact_type="diff", content="diff")
        plan_art = Artifact(id=2, artifact_type="plan", content="plan")
        assert get_latest_test_output_artifact([diff_art, plan_art]) is None

        # Multiple test artifacts with attempts
        a1 = Artifact(
            id=10,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Exit Code**: 1\n\n#### Output\n```\n...```"
            ),
        )
        a2 = Artifact(
            id=20,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 2)\n- **Exit Code**: 0\n\n#### Output\n```\n...```"
            ),
        )
        assert get_latest_test_output_artifact([a2, a1]) is a2

        # Same attempt number with distinct IDs: picks higher ID
        b1 = Artifact(
            id=30,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Exit Code**: 1\n\n#### Output\n```\n...```"
            ),
        )
        b2 = Artifact(
            id=40,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Exit Code**: 0\n\n#### Output\n```\n...```"
            ),
        )
        assert get_latest_test_output_artifact([b1, b2]) is b2

        # Missing attempt header fails closed
        c1 = Artifact(
            id=50,
            artifact_type="test_output",
            content="### Test Execution\n- **Exit Code**: 0\n\n#### Output\n```\n...```",
        )
        assert get_latest_test_output_artifact([a1, c1]) is None

        # Same attempt number with None IDs fails closed
        d1 = Artifact(
            id=None,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Exit Code**: 1\n\n#### Output\n```\n...```"
            ),
        )
        d2 = Artifact(
            id=None,
            artifact_type="test_output",
            content=(
                "### Test Execution (Attempt 1)\n- **Exit Code**: 0\n\n#### Output\n```\n...```"
            ),
        )
        assert get_latest_test_output_artifact([d1, d2]) is None

    def test_parse_test_execution_content_unit(self) -> None:
        """SEC-03B/C: Unit tests for shared parse_test_execution_content helper."""
        assert parse_test_execution_content("") is None
        assert parse_test_execution_content("plain text") is None

        # Valid canonical
        valid = (
            "### Test Execution (Attempt 3)\n"
            "- **Exit Code**: 0\n"
            "- **Duration**: 2.5s\n\n"
            "#### Output\n```\nAll tests passed\n```"
        )
        assert parse_test_execution_content(valid) == (3, 0)

        # Missing delimiter
        no_delim = "### Test Execution (Attempt 1)\n- **Exit Code**: 0\n"
        assert parse_test_execution_content(no_delim) is None

        # Missing attempt header
        no_attempt = "- **Exit Code**: 0\n\n#### Output\n```\n```"
        assert parse_test_execution_content(no_attempt) is None

        # Duplicate attempt header
        dup_attempt = (
            "### Test Execution (Attempt 1)\n"
            "### Test Execution (Attempt 2)\n"
            "- **Exit Code**: 0\n\n"
            "#### Output\n"
        )
        assert parse_test_execution_content(dup_attempt) is None

        # Duplicate exit code
        dup_code = (
            "### Test Execution (Attempt 1)\n"
            "- **Exit Code**: 0\n"
            "- **Exit Code**: 1\n\n"
            "#### Output\n"
        )
        assert parse_test_execution_content(dup_code) is None

    def test_job_state_machine_transition_direct(self) -> None:
        """JobStateMachine.transition directly enforces prerequisite gates."""
        job = Job(
            id=1,
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            status=JobStatus.AWAITING_APPROVAL,
            plan=None,
        )
        with pytest.raises(PlanApprovalRequiredError):
            JobStateMachine.transition(job, JobStatus.GENERATING_CODE)

        job.status = JobStatus.OPENING_PR
        job.artifacts = []
        with pytest.raises(PRAuthorizationRequiredError):
            JobStateMachine.transition(job, JobStatus.COMPLETED)


# =============================================================================
# 8. Service and Orchestrator Layer Direct Enforcement
# =============================================================================


class TestOrchestratorLayerEnforcement:
    """Verify authorization rules are enforced in service/orchestrator layer independently."""

    async def test_orchestrator_planning_phase_terminal_job_blocked(
        self, test_db: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        """Direct call to orchestrator.run_planning_phase rejects terminal jobs."""
        job_id = await _seed_job(test_db, status=JobStatus.COMPLETED)
        orchestrator = JobOrchestrator(
            session_factory=test_db,
            github_client=GitHubClient(token="ghp_test"),
        )
        with pytest.raises(UnauthorizedActionError, match="terminal state"):
            await orchestrator.run_planning_phase(job_id)

    async def test_orchestrator_execution_phase_unapproved_job_blocked(
        self, test_db: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        """Direct call to orchestrator.run_execution_phase rejects unapproved jobs."""
        job_id = await _seed_job(test_db, status=JobStatus.AWAITING_APPROVAL, plan_status="draft")
        orchestrator = JobOrchestrator(
            session_factory=test_db,
            github_client=GitHubClient(token="ghp_test"),
        )
        with pytest.raises(UnapprovedPlanExecutionError, match="Security violation"):
            await orchestrator.run_execution_phase(job_id)

    async def test_orchestrator_complete_pull_request_unauthorized_state_blocked(
        self, test_db: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        """Direct call to orchestrator.complete_pull_request rejects non-OPENING_PR states."""
        job_id = await _seed_job(test_db, status=JobStatus.GENERATING_CODE, plan_status="approved")
        orchestrator = JobOrchestrator(
            session_factory=test_db,
            github_client=GitHubClient(token="ghp_test"),
        )
        with pytest.raises(WorkflowAuthorizationError, match="OPENING_PR"):
            await orchestrator.complete_pull_request(job_id)

    async def test_orchestrator_cancel_terminal_job_blocked(
        self, test_db: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        """Direct call to orchestrator.cancel_job rejects already completed jobs."""
        job_id = await _seed_job(test_db, status=JobStatus.COMPLETED)
        orchestrator = JobOrchestrator(
            session_factory=test_db,
            github_client=GitHubClient(token="ghp_test"),
        )
        with pytest.raises(UnauthorizedActionError, match="already in terminal state"):
            await orchestrator.cancel_job(job_id)


# =============================================================================
# 9. Valid Workflow Progression Through Human Checkpoints
# =============================================================================


class TestValidWorkflowCheckpoints:
    """Verify that a legitimate workflow successfully transitions through each checkpoint."""

    async def test_valid_approval_and_pr_submission(
        self,
        client: httpx.AsyncClient,
        test_db: async_sessionmaker[AsyncSession],
    ) -> None:
        """A job in AWAITING_APPROVAL can be approved and subsequently complete PR creation."""
        # 1. Create job in AWAITING_APPROVAL with draft plan
        job_id = await _seed_job(test_db, status=JobStatus.AWAITING_APPROVAL, plan_status="draft")

        # 2. Human approves plan -> job transitions to GENERATING_CODE
        approve_resp = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert approve_resp.status_code == 200
        data = approve_resp.json()
        assert data["status"] == "GENERATING_CODE"
        assert data["plan"]["status"] == "approved"

        # 3. Simulate execution progressing to OPENING_PR (loading plan eagerly)
        async with test_db() as session:
            stmt = (
                select(Job)
                .options(selectinload(Job.plan), selectinload(Job.artifacts))
                .where(Job.id == job_id)
            )
            res = await session.execute(stmt)
            job = res.scalars().one()
            JobStateMachine.transition(job, JobStatus.RUNNING_TESTS)
            test_art = Artifact(
                job_id=job.id,
                artifact_type="test_output",
                content=(
                    "### Test Execution (Attempt 1)\n"
                    "- **Exit Code**: 0\n\n"
                    "#### Output\n```\nPASSED\n```\n"
                ),
            )
            session.add(test_art)
            job.artifacts.append(test_art)
            JobStateMachine.transition(job, JobStatus.OPENING_PR)
            await session.commit()

        # 4. Human authorizes PR creation via approve-pr endpoint
        async def mock_complete(jid: int) -> Job:
            async with test_db() as s:
                stmt = (
                    select(Job)
                    .options(selectinload(Job.plan), selectinload(Job.artifacts))
                    .where(Job.id == jid)
                )
                res = await s.execute(stmt)
                j = res.scalars().one()
                art = Artifact(
                    job_id=j.id,
                    artifact_type="pr_url",
                    content="https://github.com/octocat/Hello-World/pull/42",
                )
                s.add(art)
                j.artifacts.append(art)
                JobStateMachine.transition(j, JobStatus.COMPLETED)
                await s.commit()
                return j

        with patch.object(
            JobOrchestrator,
            "complete_pull_request",
            side_effect=mock_complete,
        ):
            pr_resp = await client.post(f"/jobs/{job_id}/approve-pr")
            assert pr_resp.status_code == 200
            assert pr_resp.json()["status"] == "COMPLETED"


# =============================================================================
# 10. Approval & PR Service State Machine & Relationship Safety
# =============================================================================


class TestApprovalAndPRServiceTransitions:
    """
    Verify that approval and PR services route transitions via JobStateMachine
    and handle relationships safely.
    """

    async def test_plan_approval_routes_through_state_machine(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Approving a plan transitions to GENERATING_CODE via JobStateMachine."""
        job_id = await _seed_job(test_db, status=JobStatus.AWAITING_APPROVAL, plan_status="draft")
        approval_service = PlanApprovalService()

        with patch.object(
            JobStateMachine, "transition", wraps=JobStateMachine.transition
        ) as spy_transition:
            async with test_db() as session:
                updated_job = await approval_service.review_plan(
                    session, job_id, decision="approved"
                )
                assert updated_job.status == JobStatus.GENERATING_CODE
                assert updated_job.plan.status == "approved"
                spy_transition.assert_called_once()
                args, _ = spy_transition.call_args
                assert args[1] == JobStatus.GENERATING_CODE

    async def test_plan_rejection_routes_through_state_machine(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Rejecting a plan transitions to PLANNING via JobStateMachine."""
        job_id = await _seed_job(test_db, status=JobStatus.AWAITING_APPROVAL, plan_status="draft")
        approval_service = PlanApprovalService()

        with patch.object(
            JobStateMachine, "transition", wraps=JobStateMachine.transition
        ) as spy_transition:
            async with test_db() as session:
                updated_job = await approval_service.review_plan(
                    session, job_id, decision="rejected", feedback="Please add more details"
                )
                assert updated_job.status == JobStatus.PLANNING
                assert updated_job.plan.status == "rejected"
                assert updated_job.plan.human_feedback == "Please add more details"
                spy_transition.assert_called_once()
                args, _ = spy_transition.call_args
                assert args[1] == JobStatus.PLANNING

    async def test_pr_creation_routes_through_state_machine(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """PR creation transitions to COMPLETED via JobStateMachine."""
        job_id = await _seed_job(test_db, status=JobStatus.OPENING_PR, plan_status="approved")
        github_client = GitHubClient(token="ghp_test")
        pr_service = PullRequestService(github_client)

        mock_pr = GitHubPullRequest(
            id=102,
            number=43,
            title="fix(Hello-World): resolve #5",
            html_url="https://github.com/octocat/Hello-World/pull/43",
        )
        with (
            patch.object(github_client, "create_pull_request", return_value=mock_pr),
            patch.object(
                JobStateMachine, "transition", wraps=JobStateMachine.transition
            ) as spy_transition,
        ):
            async with test_db() as session:
                await pr_service.create_pull_request_for_job(
                    session, job_id, feature_branch="agent/feature"
                )
                spy_transition.assert_called_once()
                args, _ = spy_transition.call_args
                assert args[1] == JobStatus.COMPLETED

    def test_pr_creation_handles_unloaded_artifacts_safely(self) -> None:
        """
        WorkflowAuthorizationPolicy.authorize_pr_creation does not fail with
        MissingGreenlet on unloaded artifacts.
        """
        job = Job(
            id=99,
            issue_url="https://github.com/octocat/Hello-World/issues/99",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=99,
            status=JobStatus.OPENING_PR,
        )
        assert "artifacts" not in job.__dict__
        job.plan = Plan(
            id=1,
            job_id=99,
            raw_text="Plan",
            status="approved",
        )
        # Should execute cleanly without triggering lazy-load or raising exception
        WorkflowAuthorizationPolicy.authorize_pr_creation(job)

    def test_transition_to_completed_with_unloaded_artifacts_fails_closed(self) -> None:
        """
        Transitioning to COMPLETED fails closed with PRAuthorizationRequiredError
        if artifacts are unloaded.
        """
        job = Job(
            id=100,
            issue_url="https://github.com/octocat/Hello-World/issues/100",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=100,
            status=JobStatus.OPENING_PR,
        )
        assert "artifacts" not in job.__dict__
        with pytest.raises(PRAuthorizationRequiredError, match="artifacts relationship not loaded"):
            JobStateMachine.transition(job, JobStatus.COMPLETED)
