"""
app/api/jobs.py
---------------
FastAPI router for Job lifecycle management, pagination, state transitions,
associated plans, and artifacts in the Agentic Software Development Workflow.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.github import GitHubClient, get_github_client
from app.api.schemas import (
    ApprovePlanRequest,
    ArtifactResponse,
    CreateJobRequest,
    ErrorResponse,
    JobDetailResponse,
    JobResponse,
    PlanResponse,
    UpdateJobStatusRequest,
)
from app.db.models import Job, JobStatus
from app.db.session import AsyncSessionLocal, get_db
from app.orchestrator.workflow import JobOrchestrator
from app.services.issue_ingestion import (
    DuplicateJobError,
    IssueIngestionService,
    PullRequestNotAllowedError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_orchestrator(
    github_client: GitHubClient = Depends(get_github_client),
) -> JobOrchestrator:
    """FastAPI dependency for JobOrchestrator."""
    return JobOrchestrator(
        session_factory=AsyncSessionLocal,
        github_client=github_client,
    )


# =============================================================================
# State Machine Valid Transitions
# =============================================================================

VALID_JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.FETCHING_ISSUE, JobStatus.FAILED},
    JobStatus.FETCHING_ISSUE: {
        JobStatus.INDEXING_CODEBASE,
        JobStatus.PLANNING,
        JobStatus.FAILED,
    },
    JobStatus.INDEXING_CODEBASE: {JobStatus.PLANNING, JobStatus.FAILED},
    JobStatus.PLANNING: {JobStatus.AWAITING_APPROVAL, JobStatus.FAILED},
    JobStatus.AWAITING_APPROVAL: {
        JobStatus.GENERATING_CODE,
        JobStatus.PLANNING,
        JobStatus.FAILED,
    },
    JobStatus.GENERATING_CODE: {JobStatus.RUNNING_TESTS, JobStatus.FAILED},
    JobStatus.RUNNING_TESTS: {
        JobStatus.OPENING_PR,
        JobStatus.GENERATING_CODE,
        JobStatus.FAILED,
    },
    JobStatus.OPENING_PR: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),  # Terminal state
    JobStatus.FAILED: set(),  # Terminal state
}


def _job_detail_query(job_id: int) -> Select[Job]:
    """Construct an async select statement eagerly loading relationships."""
    return (
        select(Job)
        .options(selectinload(Job.plan), selectinload(Job.artifacts))
        .where(Job.id == job_id)
        .execution_options(populate_existing=True)
    )


# =============================================================================
# Job Endpoints
# =============================================================================


@router.post(
    "",
    response_model=JobDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workflow job",
    description=(
        "Initiates a new agent workflow for a validated GitHub issue. "
        "Enforces duplicate detection for currently active (non-terminal) jobs."
    ),
    responses={
        status.HTTP_201_CREATED: {
            "model": JobDetailResponse,
            "description": "Job successfully created in PENDING state.",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "An active job already exists for the specified issue.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Validation error on issue or repository URL parameters.",
        },
    },
)
async def create_job(
    payload: CreateJobRequest,
    db: AsyncSession = Depends(get_db),
) -> JobDetailResponse:
    """Create a new Job in PENDING status if no active job exists for the target issue."""
    # Check for existing active (non-terminal) jobs for this issue
    active_stmt = select(Job).where(
        func.lower(Job.repo_owner) == payload.repo_owner.lower(),
        func.lower(Job.repo_name) == payload.repo_name.lower(),
        Job.issue_number == payload.resolved_issue_number,
        Job.status.not_in([JobStatus.COMPLETED, JobStatus.FAILED]),
    )
    result = await db.execute(active_stmt)
    existing_job = result.scalars().first()

    if existing_job is not None:
        logger.warning(
            "Conflict: active job #%s exists for %s/%s#%s with status %s",
            existing_job.id,
            payload.repo_owner,
            payload.repo_name,
            payload.resolved_issue_number,
            existing_job.status.value,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"An active job (#{existing_job.id}) already exists for "
                f"{payload.repo_owner}/{payload.repo_name}#{payload.resolved_issue_number} "
                f"with status '{existing_job.status.value}'."
            ),
        )

    new_job = Job(
        issue_url=payload.canonical_issue_url,
        repo_owner=payload.repo_owner,
        repo_name=payload.repo_name,
        issue_number=payload.resolved_issue_number,
        status=JobStatus.PENDING,
    )
    db.add(new_job)

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        logger.warning(
            "IntegrityError: duplicate active job for %s/%s#%s: %s",
            payload.repo_owner,
            payload.repo_name,
            payload.resolved_issue_number,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"An active job already exists for "
                f"{payload.repo_owner}/{payload.repo_name}#{payload.resolved_issue_number}."
            ),
        ) from exc
    except Exception:
        await db.rollback()
        raise

    # Re-query with eager loading to guarantee no MissingGreenlet during serialization
    res = await db.execute(_job_detail_query(new_job.id))
    created_job = res.scalars().one()

    logger.info("Created job #%s for %s", created_job.id, created_job.issue_url)
    return JobDetailResponse.model_validate(created_job)


@router.get(
    "",
    response_model=list[JobResponse],
    status_code=status.HTTP_200_OK,
    summary="List workflow jobs with pagination",
    description="Returns a paginated list of jobs ordered by creation timestamp descending.",
    responses={
        status.HTTP_200_OK: {
            "model": list[JobResponse],
            "description": "Paginated list of workflow jobs.",
        },
    },
)
async def list_jobs(
    limit: Annotated[
        int, Query(ge=1, le=100, description="Maximum number of items to return")
    ] = 20,
    offset: Annotated[int, Query(ge=0, description="Number of items to skip")] = 0,
    job_status: Annotated[
        JobStatus | None, Query(alias="status", description="Filter by status")
    ] = None,
    db: AsyncSession = Depends(get_db),
) -> list[JobResponse]:
    """Retrieve workflow jobs with offset/limit pagination and optional status filter."""
    stmt = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    if job_status is not None:
        stmt = stmt.where(Job.status == job_status)
    stmt = stmt.offset(offset).limit(limit)

    result = await db.execute(stmt)
    return [JobResponse.model_validate(j) for j in result.scalars().all()]


@router.get(
    "/{job_id}",
    response_model=JobDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve job details by ID",
    description="Returns full job details including plan and artifacts via eager loading.",
    responses={
        status.HTTP_200_OK: {
            "model": JobDetailResponse,
            "description": "Job details found.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Job not found.",
        },
    },
)
async def get_job(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
) -> JobDetailResponse:
    """Retrieve a single job with all related plan and artifact records."""
    result = await db.execute(_job_detail_query(job_id))
    job = result.scalars().first()

    if job is None:
        logger.debug("Job #%s not found", job_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    return JobDetailResponse.model_validate(job)


@router.patch(
    "/{job_id}/status",
    response_model=JobDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Update job status with state machine transition validation",
    description=(
        "Transitions the job status. Validates that the requested transition "
        "is permitted by the agent state machine."
    ),
    responses={
        status.HTTP_200_OK: {
            "model": JobDetailResponse,
            "description": "Job status updated successfully.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Job not found.",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Invalid state machine transition requested.",
        },
    },
)
async def update_job_status(
    payload: UpdateJobStatusRequest,
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
) -> JobDetailResponse:
    """Advance or fail a job's status according to the valid state machine transition map."""
    result = await db.execute(_job_detail_query(job_id))
    job = result.scalars().first()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    allowed_transitions = VALID_JOB_TRANSITIONS.get(job.status, set())
    if payload.status not in allowed_transitions:
        logger.warning(
            "Invalid transition requested for job #%s: %s -> %s",
            job_id,
            job.status.value,
            payload.status.value,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Invalid state transition from '{job.status.value}' to '{payload.status.value}'."
            ),
        )

    job.status = payload.status
    if payload.status == JobStatus.FAILED:
        if payload.error_message is not None:
            job.error_message = payload.error_message
    else:
        # Clear stale error messages when transitioning to a non-FAILED state
        job.error_message = None

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    res = await db.execute(_job_detail_query(job_id))
    updated_job = res.scalars().one()
    logger.info("Job #%s status updated to %s", job_id, updated_job.status.value)
    return JobDetailResponse.model_validate(updated_job)


@router.get(
    "/{job_id}/plan",
    response_model=PlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve implementation plan for a job",
    description="Returns the implementation plan associated with the specified job.",
    responses={
        status.HTTP_200_OK: {
            "model": PlanResponse,
            "description": "Implementation plan found.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Job or plan not found.",
        },
    },
)
async def get_job_plan(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
) -> PlanResponse:
    """Retrieve the plan associated with the given job ID."""
    stmt = select(Job).options(selectinload(Job.plan)).where(Job.id == job_id)
    result = await db.execute(stmt)
    job = result.scalars().first()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    if job.plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No plan found for job {job_id}.",
        )

    return PlanResponse.model_validate(job.plan)


@router.get(
    "/{job_id}/artifacts",
    response_model=list[ArtifactResponse],
    status_code=status.HTTP_200_OK,
    summary="Retrieve generated artifacts for a job",
    description="Returns all artifacts generated during the lifecycle of the job.",
    responses={
        status.HTTP_200_OK: {
            "model": list[ArtifactResponse],
            "description": "List of artifacts produced by the job.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Job not found.",
        },
    },
)
async def get_job_artifacts(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
) -> list[ArtifactResponse]:
    """Retrieve all artifacts associated with the given job ID."""
    stmt = select(Job).options(selectinload(Job.artifacts)).where(Job.id == job_id)
    result = await db.execute(stmt)
    job = result.scalars().first()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    return [ArtifactResponse.model_validate(art) for art in job.artifacts]


@router.post(
    "/{job_id}/approve",
    response_model=JobDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve or reject implementation plan",
    description=(
        "Human-in-the-loop approval endpoint. Validates that the job is in "
        "AWAITING_APPROVAL state before updating plan status and transitioning the job."
    ),
    responses={
        status.HTTP_200_OK: {
            "model": JobDetailResponse,
            "description": "Plan reviewed and job state transitioned.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Job or plan not found.",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Job is not in AWAITING_APPROVAL state.",
        },
    },
)
async def approve_plan(
    payload: ApprovePlanRequest,
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
) -> JobDetailResponse:
    """Approve or reject the plan for a job in AWAITING_APPROVAL state."""
    result = await db.execute(_job_detail_query(job_id))
    job = result.scalars().first()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with id {job_id} not found.",
        )

    if job.status != JobStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot review plan for job in '{job.status.value}' state. "
                "Job must be in 'AWAITING_APPROVAL'."
            ),
        )

    if job.plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No plan found for job {job_id}.",
        )

    if payload.decision == "approved":
        job.plan.status = "approved"
        job.status = JobStatus.GENERATING_CODE
        job.error_message = None
    elif payload.decision == "rejected":
        job.plan.status = "rejected"
        job.plan.human_feedback = payload.feedback
        job.status = JobStatus.PLANNING
        job.error_message = None

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    res = await db.execute(_job_detail_query(job.id))
    updated_job = res.scalars().one()
    logger.info(
        "Job #%s plan %s; transitioned to %s",
        job_id,
        payload.decision,
        updated_job.status.value,
    )
    return JobDetailResponse.model_validate(updated_job)


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    response_model=JobDetailResponse,
    summary="Ingest GitHub issue, normalize, and launch autonomous planning",
)
async def ingest_job(
    payload: CreateJobRequest,
    db: AsyncSession = Depends(get_db),
    github_client: GitHubClient = Depends(get_github_client),
    orchestrator: JobOrchestrator = Depends(get_orchestrator),
) -> JobDetailResponse:
    """Ingest issue directly from GitHub API and queue background planning."""
    ingestion_service = IssueIngestionService(github_client)
    try:
        job, _ = await ingestion_service.ingest_issue(
            db=db,
            owner=payload.repo_owner,
            repo=payload.repo_name,
            issue_number=payload.resolved_issue_number,
        )
    except DuplicateJobError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err)) from err
    except PullRequestNotAllowedError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err)
        ) from err

    # Launch background planning phase
    orchestrator.start_background_planning(job.id)

    res = await db.execute(_job_detail_query(job.id))
    detail = res.scalars().one()
    return JobDetailResponse.model_validate(detail)


@router.post(
    "/{job_id}/start",
    status_code=status.HTTP_200_OK,
    response_model=JobDetailResponse,
    summary="Start or resume background planning phase for a job",
)
async def start_job(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
    orchestrator: JobOrchestrator = Depends(get_orchestrator),
) -> JobDetailResponse:
    """Trigger the background planning phase for a pending or indexing job."""
    job = (await db.execute(_job_detail_query(job_id))).scalars().first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )

    orchestrator.start_background_planning(job.id)
    return JobDetailResponse.model_validate(job)


@router.post(
    "/{job_id}/execute",
    status_code=status.HTTP_200_OK,
    response_model=JobDetailResponse,
    summary="Trigger execution phase for an approved job",
)
async def execute_job(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
    orchestrator: JobOrchestrator = Depends(get_orchestrator),
) -> JobDetailResponse:
    """Trigger background code generation and sandboxed test execution for approved job."""
    job = (await db.execute(_job_detail_query(job_id))).scalars().first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )

    if job.status != JobStatus.GENERATING_CODE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot execute Job {job_id} in status '{job.status.value}'. "
                "Job must be in 'GENERATING_CODE' (approved)."
            ),
        )

    orchestrator.start_background_execution(job.id)
    return JobDetailResponse.model_validate(job)


@router.post(
    "/{job_id}/approve-pr",
    status_code=status.HTTP_200_OK,
    response_model=JobDetailResponse,
    summary="Authorize and submit GitHub Pull Request for verified job",
)
async def approve_pr(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
    orchestrator: JobOrchestrator = Depends(get_orchestrator),
) -> JobDetailResponse:
    """Human authorization checkpoint to commit and open remote Pull Request."""
    job = (await db.execute(_job_detail_query(job_id))).scalars().first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )

    if job.status != JobStatus.OPENING_PR:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot approve PR for Job {job_id} in status '{job.status.value}'. "
                "Job must be in 'OPENING_PR'."
            ),
        )

    try:
        updated = await orchestrator.complete_pull_request(job_id)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)
        ) from err

    return JobDetailResponse.model_validate(updated)


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_200_OK,
    response_model=JobDetailResponse,
    summary="Cancel an active job",
)
async def cancel_job(
    job_id: Annotated[int, Path(gt=0, description="Positive integer job identifier")],
    db: AsyncSession = Depends(get_db),
    orchestrator: JobOrchestrator = Depends(get_orchestrator),
) -> JobDetailResponse:
    """Cancel an in-flight job, terminating tasks and marking status as FAILED."""
    job = (await db.execute(_job_detail_query(job_id))).scalars().first()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )

    cancelled = await orchestrator.cancel_job(job_id)
    res = await db.execute(_job_detail_query(cancelled.id))
    return JobDetailResponse.model_validate(res.scalars().one())
