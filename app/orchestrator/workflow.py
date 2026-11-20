"""
app/orchestrator/workflow.py
----------------------------
Core persistent workflow orchestrator.
Coordinates the end-to-end autonomous engineering pipeline across all states,
enforcing human approval checkpoints, test execution, bounded repair loops,
and background execution recovery.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.adapters.github import GitHubClient
from app.adapters.llm import BaseLLMClient, get_llm_client
from app.db.models import Artifact, Job, JobStatus
from app.execution.repair import RepairEngine
from app.execution.runner import SandboxRunner
from app.intelligence.coder import CodeGenerator
from app.intelligence.context_builder import ContextBuilder
from app.intelligence.indexer import CodebaseIndexer
from app.intelligence.planner import (
    ImplementationPlan,
    ImplementationStep,
    PlanningEngine,
)
from app.intelligence.retriever import CodebaseRetriever
from app.intelligence.workspace import WorkspaceManager
from app.orchestrator.state_machine import JobStateMachine
from app.services.approval import PlanApprovalService
from app.services.git_service import GitWorkspaceService
from app.services.issue_ingestion import NormalizedIssueData
from app.services.pr_service import PullRequestService

logger = logging.getLogger(__name__)


class JobOrchestrator:
    """Coordinates persistent execution of the autonomous engineering pipeline."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        github_client: GitHubClient,
        llm_client: BaseLLMClient | None = None,
        workspace_manager: WorkspaceManager | None = None,
        runner: SandboxRunner | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.github_client = github_client
        self.llm_client = llm_client or get_llm_client()
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.runner = runner or SandboxRunner()

        # Component services
        self.indexer = CodebaseIndexer(self.workspace_manager)
        self.context_builder = ContextBuilder()
        self.planning_engine = PlanningEngine(self.llm_client)
        self.code_generator = CodeGenerator(self.llm_client, self.workspace_manager)
        self.repair_engine = RepairEngine(self.runner, self.code_generator)
        self.git_service = GitWorkspaceService(self.runner)
        self.pr_service = PullRequestService(self.github_client)

        # Track active background tasks: job_id -> Task
        self._active_tasks: dict[int, asyncio.Task[Any]] = {}

    async def _get_job(self, db: AsyncSession, job_id: int) -> Job:
        """Fetch Job eagerly loading relationships with fresh DB state."""
        stmt = (
            select(Job)
            .options(selectinload(Job.plan), selectinload(Job.artifacts))
            .where(Job.id == job_id)
            .execution_options(populate_existing=True)
        )
        result = await db.execute(stmt)
        job = result.scalars().first()
        if job is None:
            raise ValueError(f"Job {job_id} not found in database.")
        return job

    async def _get_or_create_issue_data(self, db: AsyncSession, job: Job) -> NormalizedIssueData:
        """Retrieve existing issue_data artifact or fetch from GitHub and attach to job."""
        for art in job.artifacts:
            if art.artifact_type == "issue_data":
                return NormalizedIssueData.model_validate_json(art.content)

        # Fetch from GitHub directly and attach artifact to the existing Job
        repo_data = await self.github_client.get_repository(job.repo_owner, job.repo_name)
        issue = await self.github_client.get_issue(job.repo_owner, job.repo_name, job.issue_number)
        comments = await self.github_client.get_all_issue_comments(
            job.repo_owner, job.repo_name, job.issue_number
        )

        normalized = NormalizedIssueData(
            owner=job.repo_owner,
            repo=job.repo_name,
            issue_number=job.issue_number,
            issue_url=issue.html_url
            or f"https://github.com/{job.repo_owner}/{job.repo_name}/issues/{job.issue_number}",
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

        artifact = Artifact(
            job_id=job.id,
            artifact_type="issue_data",
            content=normalized.model_dump_json(),
        )
        db.add(artifact)
        await db.commit()
        return normalized

    # -------------------------------------------------------------------------
    # Pipeline Step 1: Ingestion, Indexing & Planning (Pauses at AWAITING_APPROVAL)
    # -------------------------------------------------------------------------

    async def run_planning_phase(self, job_id: int) -> Job:
        """
        Execute the pipeline from PENDING through to AWAITING_APPROVAL.
        Always pauses at AWAITING_APPROVAL for mandatory human authorization.
        """
        async with self.session_factory() as db:
            job = await self._get_job(db, job_id)
            try:
                # 1. Fetching Issue
                if job.status == JobStatus.PENDING:
                    JobStateMachine.transition(job, JobStatus.FETCHING_ISSUE)
                    await db.commit()

                issue_data = await self._get_or_create_issue_data(db, job)

                # 2. Indexing Codebase
                if job.status == JobStatus.FETCHING_ISSUE:
                    JobStateMachine.transition(job, JobStatus.INDEXING_CODEBASE)
                    # Prepare workspace
                    ws_path = self.workspace_manager.create_workspace(job.id)
                    job.workspace_path = str(ws_path)
                    await db.commit()

                # Index workspace
                ws_path = Path(
                    job.workspace_path or self.workspace_manager.get_job_workspace(job.id)
                )
                index = self.indexer.index_workspace(ws_path)
                retriever = CodebaseRetriever(index)
                query = f"{issue_data.title}\n{issue_data.body}"
                ranked_chunks = retriever.retrieve(query, top_k=10)
                codebase_context = self.context_builder.build_context(ranked_chunks)

                # 3. Planning
                if job.status == JobStatus.INDEXING_CODEBASE:
                    JobStateMachine.transition(job, JobStatus.PLANNING)
                    await db.commit()

                # Generate plan and transition to AWAITING_APPROVAL
                human_feedback = job.plan.human_feedback if job.plan else None
                await self.planning_engine.create_and_persist_plan(
                    db=db,
                    job=job,
                    issue=issue_data,
                    context=codebase_context,
                    human_feedback=human_feedback,
                )

                logger.info(
                    "Job %d paused in %s awaiting human plan approval.",
                    job.id,
                    job.status.value,
                )
                return job

            except Exception as err:
                logger.error("Job %d failed during planning phase: %s", job_id, err, exc_info=True)
                JobStateMachine.transition(job, JobStatus.FAILED, error_message=str(err))
                await db.commit()
                raise

    # -------------------------------------------------------------------------
    # Pipeline Step 2: Code Generation, Testing & Repair
    # -------------------------------------------------------------------------

    async def run_execution_phase(self, job_id: int) -> Job:
        """
        Execute code generation, test execution, and repair after human approval.
        Enforces that plan is approved before modifying workspace files.
        """
        async with self.session_factory() as db:
            job = await self._get_job(db, job_id)

            # Security assertion: Plan must be approved!
            PlanApprovalService.assert_can_generate_code(job)

            try:
                ws_path = Path(
                    job.workspace_path or self.workspace_manager.get_job_workspace(job.id)
                )
                issue_data = await self._get_or_create_issue_data(db, job)

                # Reconstruct ImplementationPlan from stored steps and raw_text
                assert job.plan is not None
                plan_struct = ImplementationPlan(
                    summary=job.plan.raw_text.splitlines()[0].replace(
                        "# Implementation Plan: ", ""
                    ),
                    problem_statement="See approved plan.",
                    technical_approach=job.plan.raw_text,
                    steps=[
                        ImplementationStep(
                            step_number=s.get("step_number", 1),
                            description=s.get("description", ""),
                            file_paths=s.get("file_paths", []),
                        )
                        for s in (job.plan.steps or [])
                    ],
                    test_plan="Execute automated unit tests in isolated workspace.",
                )

                # 1. Code Generation
                logger.info("Generating code for Job %d", job.id)
                gen_result = await self.code_generator.generate_code(
                    plan=plan_struct,
                    codebase_context=f"Issue: #{job.issue_number} - {issue_data.title}",
                )

                # Apply changes safely
                modified_files, diff = self.code_generator.validate_and_apply_changes(
                    job.id, gen_result.changes
                )

                # Store diff artifact
                if diff:
                    db.add(Artifact(job_id=job.id, artifact_type="diff", content=diff))
                    await db.commit()

                # 2. Transition to RUNNING_TESTS
                JobStateMachine.transition(job, JobStatus.RUNNING_TESTS)
                await db.commit()

                # 3. Test execution and repair loop
                repair_result = await self.repair_engine.run_tests_and_repair_loop(
                    db=db,
                    job=job,
                    workspace_path=ws_path,
                    plan=plan_struct,
                    codebase_context=f"Issue: #{job.issue_number} - {issue_data.title}",
                )

                if not repair_result.success:
                    error_msg = (
                        repair_result.error_message
                        or "Automated tests failed after repair attempts."
                    )
                    JobStateMachine.transition(job, JobStatus.FAILED, error_message=error_msg)
                    await db.commit()
                    return job

                # 4. Tests passed -> Transition to OPENING_PR (awaiting PR authorization)
                JobStateMachine.transition(job, JobStatus.OPENING_PR)
                await db.commit()

                # Reload job with fresh artifacts
                job = await self._get_job(db, job_id)
                logger.info("Job %d completed execution and is now in %s", job.id, job.status.value)
                return job

            except Exception as err:
                logger.error("Job %d failed during execution phase: %s", job_id, err, exc_info=True)
                JobStateMachine.transition(job, JobStatus.FAILED, error_message=str(err))
                await db.commit()
                raise

    # -------------------------------------------------------------------------
    # Pipeline Step 3: Git Branch & Pull Request Creation
    # -------------------------------------------------------------------------

    async def complete_pull_request(self, job_id: int, base_branch: str = "main") -> Job:
        """
        Create dedicated feature branch, commit changes, and submit GitHub PR.
        Requires explicit human authorization gate.
        """
        async with self.session_factory() as db:
            job = await self._get_job(db, job_id)

            if job.status != JobStatus.OPENING_PR:
                raise ValueError(
                    f"Cannot complete PR for Job {job_id} with status '{job.status.value}'. "
                    "Must be in 'OPENING_PR'."
                )

            ws_path = Path(job.workspace_path or self.workspace_manager.get_job_workspace(job.id))
            feature_branch = f"agent/issue-{job.issue_number}-{job.id}"

            try:
                # 1. Checkout feature branch and commit
                await self.git_service.create_feature_branch(ws_path, feature_branch)
                commit_msg = f"fix({job.repo_name}): resolve #{job.issue_number}"
                await self.git_service.stage_and_commit(ws_path, commit_msg)

                # 2. Submit Pull Request via PRService
                await self.pr_service.create_pull_request_for_job(
                    db=db,
                    job_id=job.id,
                    feature_branch=feature_branch,
                    base_branch=base_branch,
                )

                # Reload job
                job = await self._get_job(db, job_id)
                logger.info("Job %d fully COMPLETED with Pull Request.", job.id)
                return job

            except Exception as err:
                logger.error("Job %d failed creating pull request: %s", job_id, err, exc_info=True)
                JobStateMachine.transition(job, JobStatus.FAILED, error_message=str(err))
                await db.commit()
                raise

    # -------------------------------------------------------------------------
    # Background Task Management & Recovery
    # -------------------------------------------------------------------------

    def start_background_planning(self, job_id: int) -> asyncio.Task[Any]:
        """Launch planning phase asynchronously without blocking API request."""
        task = asyncio.create_task(
            self.run_planning_phase(job_id),
            name=f"job_planning_{job_id}",
        )
        self._active_tasks[job_id] = task
        task.add_done_callback(lambda _: self._active_tasks.pop(job_id, None))
        return task

    def start_background_execution(self, job_id: int) -> asyncio.Task[Any]:
        """Launch code generation & test phase asynchronously after approval."""
        task = asyncio.create_task(
            self.run_execution_phase(job_id),
            name=f"job_exec_{job_id}",
        )
        self._active_tasks[job_id] = task
        task.add_done_callback(lambda _: self._active_tasks.pop(job_id, None))
        return task

    async def cancel_job(self, job_id: int) -> Job:
        """Cancel an in-flight job and transition to FAILED."""
        # Cancel running task if active
        task = self._active_tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
            logger.info("Cancelled running background task for Job %d", job_id)

        async with self.session_factory() as db:
            job = await self._get_job(db, job_id)
            if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                JobStateMachine.transition(
                    job, JobStatus.FAILED, error_message="Job cancelled by user."
                )
                await db.commit()
                await db.refresh(job)
            return job

    async def recover_interrupted_jobs(self) -> int:
        """
        Inspect persisted jobs on application startup.
        Safe recovery: non-terminal jobs that were mid-flight (e.g. FETCHING_ISSUE,
        PLANNING, GENERATING_CODE) are cleanly marked for safe recovery.
        """
        recovered_count = 0
        async with self.session_factory() as db:
            stmt = select(Job).where(
                Job.status.not_in(
                    [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.AWAITING_APPROVAL]
                )
            )
            result = await db.execute(stmt)
            interrupted_jobs = result.scalars().all()

            for job in interrupted_jobs:
                logger.warning(
                    "Found interrupted Job %d in state '%s'. Marking for safe recovery.",
                    job.id,
                    job.status.value,
                )
                # Fail cleanly with diagnostic so user can re-trigger or inspect
                job.status = JobStatus.FAILED
                job.error_message = (
                    f"Job interrupted in '{job.status.value}' due to process restart. "
                    "Safe recovery applied."
                )
                recovered_count += 1

            if recovered_count > 0:
                await db.commit()
                logger.info("Recovered %d interrupted jobs on startup.", recovered_count)

        return recovered_count
