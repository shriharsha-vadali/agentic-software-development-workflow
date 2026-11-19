"""
tests/unit/test_approval_service.py
-----------------------------------
Unit tests for the Human Approval Gate service (app/services/approval.py).
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, Job, JobStatus, Plan
from app.services.approval import (
    InvalidJobStateForApprovalError,
    MissingPlanError,
    PlanApprovalService,
    UnapprovedPlanExecutionError,
)

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


class TestPlanApprovalService:
    """Verify approval gate enforcement, plan transitions, and security checks."""

    async def test_approve_plan_success(self, test_session: AsyncSession) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/10",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=10,
            status=JobStatus.AWAITING_APPROVAL,
        )
        test_session.add(job)
        await test_session.flush()

        plan = Plan(job_id=job.id, raw_text="Sample plan", status="draft")
        test_session.add(plan)
        await test_session.commit()

        updated_job = await PlanApprovalService.review_plan(
            test_session, job.id, decision="approved"
        )
        assert updated_job.status == JobStatus.GENERATING_CODE
        assert updated_job.plan is not None
        assert updated_job.plan.status == "approved"

        # Must pass assertion check
        PlanApprovalService.assert_can_generate_code(updated_job)

    async def test_reject_plan_success(self, test_session: AsyncSession) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/10",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=10,
            status=JobStatus.AWAITING_APPROVAL,
        )
        test_session.add(job)
        await test_session.flush()

        plan = Plan(job_id=job.id, raw_text="Sample plan", status="draft")
        test_session.add(plan)
        await test_session.commit()

        updated_job = await PlanApprovalService.review_plan(
            test_session, job.id, decision="rejected", feedback="Please use poetry instead"
        )
        assert updated_job.status == JobStatus.PLANNING
        assert updated_job.plan is not None
        assert updated_job.plan.status == "rejected"
        assert updated_job.plan.human_feedback == "Please use poetry instead"

        # Must block code generation
        with pytest.raises(UnapprovedPlanExecutionError, match="Security violation"):
            PlanApprovalService.assert_can_generate_code(updated_job)

    async def test_review_requires_awaiting_approval_status(
        self, test_session: AsyncSession
    ) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/10",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=10,
            status=JobStatus.PENDING,
        )
        test_session.add(job)
        await test_session.commit()

        with pytest.raises(
            InvalidJobStateForApprovalError, match="Job must be in 'AWAITING_APPROVAL'"
        ):
            await PlanApprovalService.review_plan(test_session, job.id, decision="approved")

    async def test_review_requires_existing_plan(self, test_session: AsyncSession) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/10",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=10,
            status=JobStatus.AWAITING_APPROVAL,
        )
        test_session.add(job)
        await test_session.commit()

        with pytest.raises(MissingPlanError, match="No plan found"):
            await PlanApprovalService.review_plan(test_session, job.id, decision="approved")
