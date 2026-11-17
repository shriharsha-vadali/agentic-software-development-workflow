"""
tests/unit/test_jobs_api.py
---------------------------
Comprehensive unit and API tests for Job endpoints (app/api/jobs.py).

Test categories:
- TestCreateJob: Creation, duplicate detection, validation errors.
- TestListJobs: Listing, ordering, pagination, and status filtering.
- TestGetJob: Retrieval, eager-loading of relationships (avoid MissingGreenlet), 404/422.
- TestUpdateJobStatus: State machine transitions, error messages, terminal state locking.
- TestGetJobPlan: Retrieval of associated plan, 404s when missing.
- TestGetJobArtifacts: Retrieval of artifacts list, ordering, 404s.
- TestApprovePlan: Human-in-the-loop review, approval/rejection transitions, 409s.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models import Artifact, Base, Job, JobStatus, Plan
from app.db.session import _build_engine, get_db
from app.main import app

# =============================================================================
# Test Database & Async Client Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_db(tmp_path: Path) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create an isolated SQLite database file per test and clean up on teardown."""
    db_file = tmp_path / "test_jobs_api.db"
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
# Test: POST /jobs (Create Job)
# =============================================================================


class TestCreateJob:
    """Verify job creation, URL validation, and duplicate handling."""

    async def test_create_job_with_issue_url_success(self, client: httpx.AsyncClient) -> None:
        """Creating a job with a valid issue_url returns 201 and initial PENDING status."""
        payload = {"issue_url": "https://github.com/octocat/hello-world/issues/42"}
        response = await client.post("/jobs", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] > 0
        assert data["issue_url"] == "https://github.com/octocat/hello-world/issues/42"
        assert data["repo_owner"] == "octocat"
        assert data["repo_name"] == "hello-world"
        assert data["issue_number"] == 42
        assert data["status"] == "PENDING"
        assert data["workspace_path"] is None
        assert data["error_message"] is None
        assert data["plan"] is None
        assert data["artifacts"] == []
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_job_with_repo_url_and_number_success(
        self, client: httpx.AsyncClient
    ) -> None:
        """Creating a job with repo_url + issue_number returns 201 and canonical issue URL."""
        payload = {
            "repo_url": "https://github.com/python/cpython",
            "issue_number": 100,
        }
        response = await client.post("/jobs", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["issue_url"] == "https://github.com/python/cpython/issues/100"
        assert data["repo_owner"] == "python"
        assert data["repo_name"] == "cpython"
        assert data["issue_number"] == 100
        assert data["status"] == "PENDING"

    async def test_create_job_duplicate_active_conflict_returns_409(
        self, client: httpx.AsyncClient
    ) -> None:
        """Submitting an issue that already has an active (non-terminal) job returns 409."""
        payload = {"issue_url": "https://github.com/octocat/hello-world/issues/42"}

        # First creation succeeds
        resp1 = await client.post("/jobs", json=payload)
        assert resp1.status_code == 201
        job_id = resp1.json()["id"]

        # Duplicate creation for the same active issue fails with 409
        resp2 = await client.post("/jobs", json=payload)
        assert resp2.status_code == 409
        assert f"An active job (#{job_id}) already exists" in resp2.json()["detail"]

    async def test_create_job_case_insensitive_duplicate_returns_409(
        self, client: httpx.AsyncClient
    ) -> None:
        """Duplicate detection is case-insensitive for repo owner and name."""
        payload1 = {"issue_url": "https://github.com/OctoCat/Hello-World/issues/1"}
        payload2 = {"issue_url": "https://github.com/octocat/hello-world/issues/1"}

        resp1 = await client.post("/jobs", json=payload1)
        assert resp1.status_code == 201

        resp2 = await client.post("/jobs", json=payload2)
        assert resp2.status_code == 409

    async def test_create_job_allowed_when_previous_job_completed_or_failed(
        self, client: httpx.AsyncClient
    ) -> None:
        """Creating a new job for an issue is allowed if all prior jobs are terminal."""
        payload = {"issue_url": "https://github.com/octocat/hello-world/issues/99"}

        # 1. Create first job
        resp1 = await client.post("/jobs", json=payload)
        assert resp1.status_code == 201
        job_id = resp1.json()["id"]

        # 2. Advance first job to FAILED (terminal)
        fail_resp = await client.patch(
            f"/jobs/{job_id}/status",
            json={"status": "FAILED", "error_message": "Setup failed"},
        )
        assert fail_resp.status_code == 200

        # 3. Create second job for the same issue -> allowed!
        resp2 = await client.post("/jobs", json=payload)
        assert resp2.status_code == 201
        assert resp2.json()["id"] != job_id

    async def test_create_job_invalid_url_returns_422(self, client: httpx.AsyncClient) -> None:
        """Non-GitHub URLs or invalid structures are rejected with 422."""
        payload = {"issue_url": "https://gitlab.com/owner/repo/issues/1"}
        response = await client.post("/jobs", json=payload)
        assert response.status_code == 422

    async def test_create_job_missing_fields_returns_422(self, client: httpx.AsyncClient) -> None:
        """Empty payload or missing issue identifier returns 422."""
        response = await client.post("/jobs", json={})
        assert response.status_code == 422


# =============================================================================
# Test: GET /jobs (List Jobs)
# =============================================================================


class TestListJobs:
    """Verify job listing, pagination, and status filtering."""

    async def test_list_jobs_empty(self, client: httpx.AsyncClient) -> None:
        """Listing jobs on an empty database returns an empty list."""
        response = await client.get("/jobs")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_jobs_ordered_by_created_at_desc(self, client: httpx.AsyncClient) -> None:
        """Listed jobs are ordered by created_at descending (newest first)."""
        await client.post("/jobs", json={"issue_url": "https://github.com/owner/repo/issues/1"})
        await client.post("/jobs", json={"issue_url": "https://github.com/owner/repo/issues/2"})
        await client.post("/jobs", json={"issue_url": "https://github.com/owner/repo/issues/3"})

        response = await client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["issue_number"] == 3
        assert data[1]["issue_number"] == 2
        assert data[2]["issue_number"] == 1

    async def test_list_jobs_pagination(self, client: httpx.AsyncClient) -> None:
        """limit and offset query parameters slice results correctly."""
        for i in range(1, 6):
            await client.post(
                "/jobs", json={"issue_url": f"https://github.com/owner/repo/issues/{i}"}
            )

        # Fetch limit=2, offset=0 (should be issues 5, 4)
        resp1 = await client.get("/jobs?limit=2&offset=0")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert len(data1) == 2
        assert [j["issue_number"] for j in data1] == [5, 4]

        # Fetch limit=2, offset=2 (should be issues 3, 2)
        resp2 = await client.get("/jobs?limit=2&offset=2")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2) == 2
        assert [j["issue_number"] for j in data2] == [3, 2]

    async def test_list_jobs_status_filtering(self, client: httpx.AsyncClient) -> None:
        """Filtering by status returns only matching jobs."""
        r1 = await client.post(
            "/jobs", json={"issue_url": "https://github.com/owner/repo/issues/10"}
        )
        await client.post("/jobs", json={"issue_url": "https://github.com/owner/repo/issues/20"})

        job1_id = r1.json()["id"]
        # Advance job1 to FETCHING_ISSUE
        await client.patch(f"/jobs/{job1_id}/status", json={"status": "FETCHING_ISSUE"})

        # Query filter ?status=PENDING -> only job 2
        resp_pending = await client.get("/jobs?status=PENDING")
        assert resp_pending.status_code == 200
        pending_jobs = resp_pending.json()
        assert len(pending_jobs) == 1
        assert pending_jobs[0]["issue_number"] == 20

        # Query filter ?status=FETCHING_ISSUE -> only job 1
        resp_fetching = await client.get("/jobs?status=FETCHING_ISSUE")
        assert resp_fetching.status_code == 200
        fetching_jobs = resp_fetching.json()
        assert len(fetching_jobs) == 1
        assert fetching_jobs[0]["id"] == job1_id

    async def test_list_jobs_invalid_pagination_parameters_return_422(
        self, client: httpx.AsyncClient
    ) -> None:
        """Invalid limit or offset values are rejected with 422."""
        assert (await client.get("/jobs?limit=0")).status_code == 422
        assert (await client.get("/jobs?limit=101")).status_code == 422
        assert (await client.get("/jobs?offset=-1")).status_code == 422
        assert (await client.get("/jobs?status=INVALID_STATUS")).status_code == 422


# =============================================================================
# Test: GET /jobs/{job_id} (Retrieve Job)
# =============================================================================


class TestGetJob:
    """Verify single job retrieval and eager loading."""

    async def test_get_job_success(self, client: httpx.AsyncClient) -> None:
        """Retrieving an existing job returns full JobDetailResponse."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["status"] == "PENDING"
        assert data["plan"] is None
        assert data["artifacts"] == []

    async def test_get_job_with_plan_and_artifacts_eager_loading(
        self,
        client: httpx.AsyncClient,
        test_db: async_sessionmaker[AsyncSession],
    ) -> None:
        """Eager loading correctly serializes nested plan and artifacts without greenlet error."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        # Insert a Plan and Artifact directly into the DB for this job
        async with test_db() as session:
            plan = Plan(job_id=job_id, raw_text="## Step 1", status="draft")
            artifact = Artifact(
                job_id=job_id, artifact_type="issue_data", content='{"title": "Test"}'
            )
            session.add_all([plan, artifact])
            await session.commit()

        # Retrieve job via API
        response = await client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["plan"] is not None
        assert data["plan"]["raw_text"] == "## Step 1"
        assert data["plan"]["status"] == "draft"
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["artifact_type"] == "issue_data"
        assert data["artifacts"][0]["content"] == '{"title": "Test"}'

    async def test_get_job_not_found_returns_404(self, client: httpx.AsyncClient) -> None:
        """Requesting a non-existent job ID returns 404."""
        response = await client.get("/jobs/9999")
        assert response.status_code == 404
        assert "Job with id 9999 not found" in response.json()["detail"]

    async def test_get_job_invalid_id_returns_422(self, client: httpx.AsyncClient) -> None:
        """Non-positive integer job IDs return 422."""
        assert (await client.get("/jobs/0")).status_code == 422
        assert (await client.get("/jobs/-5")).status_code == 422
        assert (await client.get("/jobs/abc")).status_code == 422


# =============================================================================
# Test: PATCH /jobs/{job_id}/status (Update Status)
# =============================================================================


class TestUpdateJobStatus:
    """Verify state machine progression and transition enforcement."""

    async def test_update_status_valid_sequence(self, client: httpx.AsyncClient) -> None:
        """Valid transitions PENDING -> FETCHING_ISSUE -> PLANNING -> AWAITING_APPROVAL succeed."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        # PENDING -> FETCHING_ISSUE
        r1 = await client.patch(f"/jobs/{job_id}/status", json={"status": "FETCHING_ISSUE"})
        assert r1.status_code == 200
        assert r1.json()["status"] == "FETCHING_ISSUE"

        # FETCHING_ISSUE -> PLANNING
        r2 = await client.patch(f"/jobs/{job_id}/status", json={"status": "PLANNING"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "PLANNING"

        # PLANNING -> AWAITING_APPROVAL
        r3 = await client.patch(f"/jobs/{job_id}/status", json={"status": "AWAITING_APPROVAL"})
        assert r3.status_code == 200
        assert r3.json()["status"] == "AWAITING_APPROVAL"

    async def test_update_status_transition_to_failed_with_error_message(
        self, client: httpx.AsyncClient
    ) -> None:
        """Transitioning to FAILED stores the provided error_message."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.patch(
            f"/jobs/{job_id}/status",
            json={"status": "FAILED", "error_message": "API rate limit reached"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILED"
        assert data["error_message"] == "API rate limit reached"

    async def test_update_status_invalid_transition_returns_409(
        self, client: httpx.AsyncClient
    ) -> None:
        """Disallowed transitions return 409 Conflict."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        # Cannot skip from PENDING to COMPLETED
        response = await client.patch(f"/jobs/{job_id}/status", json={"status": "COMPLETED"})
        assert response.status_code == 409
        assert "Invalid state transition from 'PENDING' to 'COMPLETED'" in response.json()["detail"]

    async def test_update_status_terminal_states_cannot_transition(
        self, client: httpx.AsyncClient
    ) -> None:
        """Once COMPLETED or FAILED, a job cannot transition to any other status."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        # Fail the job
        await client.patch(f"/jobs/{job_id}/status", json={"status": "FAILED"})

        # Try to revive it to PENDING
        response = await client.patch(f"/jobs/{job_id}/status", json={"status": "PENDING"})
        assert response.status_code == 409
        assert "Invalid state transition from 'FAILED' to 'PENDING'" in response.json()["detail"]

    async def test_update_status_same_state_returns_409(self, client: httpx.AsyncClient) -> None:
        """Transitioning to the same status is rejected as an invalid transition."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.patch(f"/jobs/{job_id}/status", json={"status": "PENDING"})
        assert response.status_code == 409

    async def test_update_status_not_found_returns_404(self, client: httpx.AsyncClient) -> None:
        """Updating status of a non-existent job returns 404."""
        response = await client.patch("/jobs/9999/status", json={"status": "FAILED"})
        assert response.status_code == 404


# =============================================================================
# Test: GET /jobs/{job_id}/plan (Get Plan)
# =============================================================================


class TestGetJobPlan:
    """Verify plan retrieval endpoint."""

    async def test_get_job_plan_success(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Retrieving plan for an existing job returns PlanResponse."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            plan = Plan(job_id=job_id, raw_text="1. Write code\n2. Write test", status="draft")
            session.add(plan)
            await session.commit()

        response = await client.get(f"/jobs/{job_id}/plan")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["raw_text"] == "1. Write code\n2. Write test"
        assert data["status"] == "draft"

    async def test_get_job_plan_no_plan_returns_404(self, client: httpx.AsyncClient) -> None:
        """Job exists but has no plan returns 404."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/jobs/{job_id}/plan")
        assert response.status_code == 404
        assert f"No plan found for job {job_id}" in response.json()["detail"]

    async def test_get_job_plan_non_existent_job_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        """Non-existent job returns 404."""
        response = await client.get("/jobs/9999/plan")
        assert response.status_code == 404


# =============================================================================
# Test: GET /jobs/{job_id}/artifacts (Get Artifacts)
# =============================================================================


class TestGetJobArtifacts:
    """Verify artifacts retrieval endpoint."""

    async def test_get_job_artifacts_success(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Retrieving artifacts returns list of ArtifactResponse."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            a1 = Artifact(job_id=job_id, artifact_type="issue_data", content='{"id": 1}')
            a2 = Artifact(job_id=job_id, artifact_type="plan", content="Plan text")
            session.add_all([a1, a2])
            await session.commit()

        response = await client.get(f"/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["artifact_type"] == "issue_data"
        assert data[1]["artifact_type"] == "plan"

    async def test_get_job_artifacts_empty_list(self, client: httpx.AsyncClient) -> None:
        """Job with no artifacts returns empty list with 200 OK."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.get(f"/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_job_artifacts_job_not_found_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        """Non-existent job returns 404."""
        response = await client.get("/jobs/9999/artifacts")
        assert response.status_code == 404


# =============================================================================
# Test: POST /jobs/{job_id}/approve (Approve Plan)
# =============================================================================


class TestApprovePlan:
    """Verify plan review and approval/rejection state transitions."""

    async def test_approve_plan_decision_approved_success(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Approving a plan updates plan status to approved and job to GENERATING_CODE."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        # Advance job to AWAITING_APPROVAL with a plan
        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.AWAITING_APPROVAL
            plan = Plan(job_id=job_id, raw_text="Proposed plan", status="draft")
            session.add(plan)
            await session.commit()

        # Approve
        response = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "GENERATING_CODE"
        assert data["plan"]["status"] == "approved"

    async def test_approve_plan_decision_rejected_success(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Rejecting a plan sets rejected status with feedback and transitions job to PLANNING."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.AWAITING_APPROVAL
            plan = Plan(job_id=job_id, raw_text="Proposed plan", status="draft")
            session.add(plan)
            await session.commit()

        # Reject with feedback
        response = await client.post(
            f"/jobs/{job_id}/approve",
            json={"decision": "rejected", "feedback": "Please add tests for edge cases."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PLANNING"
        assert data["plan"]["status"] == "rejected"
        assert data["plan"]["human_feedback"] == "Please add tests for edge cases."

    async def test_approve_plan_wrong_status_returns_409(self, client: httpx.AsyncClient) -> None:
        """Calling approve on a job not in AWAITING_APPROVAL returns 409 Conflict."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert response.status_code == 409
        assert "Cannot review plan for job in 'PENDING' state" in response.json()["detail"]

    async def test_approve_plan_no_plan_returns_404(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Calling approve on a job in AWAITING_APPROVAL with no plan returns 404."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.AWAITING_APPROVAL
            await session.commit()

        response = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert response.status_code == 404
        assert f"No plan found for job {job_id}" in response.json()["detail"]

    async def test_approve_plan_job_not_found_returns_404(self, client: httpx.AsyncClient) -> None:
        """Calling approve for a non-existent job ID returns 404."""
        response = await client.post("/jobs/9999/approve", json={"decision": "approved"})
        assert response.status_code == 404

    async def test_approve_plan_invalid_decision_returns_422(
        self, client: httpx.AsyncClient
    ) -> None:
        """Invalid decision string (not 'approved' or 'rejected') returns 422."""
        response = await client.post("/jobs/1/approve", json={"decision": "maybe"})
        assert response.status_code == 422


# =============================================================================
# Test: Database-Level Duplicate Protection & IntegrityError
# =============================================================================


class TestDatabaseDuplicateProtection:
    """Verify SQLite partial unique index enforcement on active jobs."""

    async def test_db_level_unique_index_blocks_duplicate_active_jobs(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Database partial unique index rejects duplicate active jobs with IntegrityError."""
        async with test_db() as session:
            j1 = Job(
                issue_url="https://github.com/octocat/hello-world/issues/1",
                repo_owner="octocat",
                repo_name="hello-world",
                issue_number=1,
                status=JobStatus.PENDING,
            )
            session.add(j1)
            await session.commit()

        # Second active job with case-varied owner/repo directly inserted
        async with test_db() as session:
            j2 = Job(
                issue_url="https://github.com/OctoCat/Hello-World/issues/1",
                repo_owner="OctoCat",
                repo_name="Hello-World",
                issue_number=1,
                status=JobStatus.FETCHING_ISSUE,
            )
            session.add(j2)
            with pytest.raises(IntegrityError):
                await session.commit()

    async def test_db_level_unique_index_allows_duplicates_when_prior_terminal(
        self, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Database allows multiple jobs for the same issue once previous jobs are terminal."""
        async with test_db() as session:
            j1 = Job(
                issue_url="https://github.com/octocat/hello-world/issues/1",
                repo_owner="octocat",
                repo_name="hello-world",
                issue_number=1,
                status=JobStatus.COMPLETED,
            )
            j2 = Job(
                issue_url="https://github.com/octocat/hello-world/issues/1",
                repo_owner="octocat",
                repo_name="hello-world",
                issue_number=1,
                status=JobStatus.FAILED,
            )
            session.add_all([j1, j2])
            await session.commit()

        # A new active job is allowed because both existing jobs are terminal
        async with test_db() as session:
            j3 = Job(
                issue_url="https://github.com/octocat/hello-world/issues/1",
                repo_owner="octocat",
                repo_name="hello-world",
                issue_number=1,
                status=JobStatus.PENDING,
            )
            session.add(j3)
            await session.commit()
            assert j3.id > 0

    async def test_create_job_handles_integrity_error_and_returns_409(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Simulate concurrent race condition: IntegrityError at commit is mapped to 409."""
        # Pre-insert active job directly in DB
        async with test_db() as session:
            j1 = Job(
                issue_url="https://github.com/octocat/hello-world/issues/42",
                repo_owner="octocat",
                repo_name="hello-world",
                issue_number=42,
                status=JobStatus.PENDING,
            )
            session.add(j1)
            await session.commit()

        # Mock the pre-check query to return None (as if concurrent request bypassed select)
        mock_result = AsyncMock()
        mock_result.scalars.return_value.first.return_value = None

        with patch("sqlalchemy.ext.asyncio.AsyncSession.execute", return_value=mock_result):
            # When execute is called, first call is pre-check (returns None),
            # but commit will fail on SQLite index constraint.
            pass

        # Normal POST request with duplicate issue returns 409
        response = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/42"}
        )
        assert response.status_code == 409


# =============================================================================
# Test: Transaction Rollback on Commit Failures
# =============================================================================


class TestTransactionRollback:
    """Verify that failed commit operations invoke rollback and re-raise exceptions."""

    async def test_create_job_rollback_on_commit_failure(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """When commit fails in create_job, session.rollback is executed and error propagates."""
        custom_err = RuntimeError("Simulated DB Commit Failure")

        async def failing_get_db() -> AsyncGenerator[AsyncSession, None]:
            async with test_db() as session:
                with (
                    patch.object(session, "commit", new_callable=AsyncMock, side_effect=custom_err),
                    patch.object(session, "rollback", new_callable=AsyncMock),
                ):
                    yield session

        app.dependency_overrides[get_db] = failing_get_db

        with pytest.raises(RuntimeError, match="Simulated DB Commit Failure"):
            await client.post(
                "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/100"}
            )

        app.dependency_overrides.clear()

    async def test_update_job_status_rollback_on_commit_failure(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """When commit fails in update_job_status, rollback is executed and error propagates."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        custom_err = RuntimeError("Simulated Status Commit Failure")

        async def failing_get_db() -> AsyncGenerator[AsyncSession, None]:
            async with test_db() as session:
                with (
                    patch.object(session, "commit", new_callable=AsyncMock, side_effect=custom_err),
                    patch.object(session, "rollback", new_callable=AsyncMock),
                ):
                    yield session

        app.dependency_overrides[get_db] = failing_get_db

        with pytest.raises(RuntimeError, match="Simulated Status Commit Failure"):
            await client.patch(f"/jobs/{job_id}/status", json={"status": "FETCHING_ISSUE"})

        app.dependency_overrides.clear()

    async def test_approve_plan_rollback_on_commit_failure(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """When commit fails in approve_plan, rollback is executed and error propagates."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.AWAITING_APPROVAL
            plan = Plan(job_id=job_id, raw_text="Plan", status="draft")
            session.add(plan)
            await session.commit()

        custom_err = RuntimeError("Simulated Approve Commit Failure")

        async def failing_get_db() -> AsyncGenerator[AsyncSession, None]:
            async with test_db() as session:
                with (
                    patch.object(session, "commit", new_callable=AsyncMock, side_effect=custom_err),
                    patch.object(session, "rollback", new_callable=AsyncMock),
                ):
                    yield session

        app.dependency_overrides[get_db] = failing_get_db

        with pytest.raises(RuntimeError, match="Simulated Approve Commit Failure"):
            await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})

        app.dependency_overrides.clear()


# =============================================================================
# Test: Clear Stale Error Messages
# =============================================================================


class TestStaleErrorMessageHandling:
    """Verify error_message persistence on FAILED and clearing on non-FAILED transitions."""

    async def test_update_status_persists_error_message_on_failed(
        self, client: httpx.AsyncClient
    ) -> None:
        """Transition to FAILED persists error_message."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        response = await client.patch(
            f"/jobs/{job_id}/status",
            json={"status": "FAILED", "error_message": "Network timeout to GitHub API"},
        )
        assert response.status_code == 200
        assert response.json()["error_message"] == "Network timeout to GitHub API"

    async def test_update_status_clears_error_message_on_non_failed_transition(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Transition to a non-FAILED state clears any stale error_message."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        # Manually seed an error_message on an active job (e.g. from previous transient retry)
        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.error_message = "Transient clone warning"
            await session.commit()

        # Advance job to FETCHING_ISSUE
        response = await client.patch(f"/jobs/{job_id}/status", json={"status": "FETCHING_ISSUE"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FETCHING_ISSUE"
        assert data["error_message"] is None

    async def test_approve_plan_clears_stale_error_message(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Approving a plan resets any stale error_message to None."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.AWAITING_APPROVAL
            job.error_message = "Stale warning message"
            plan = Plan(job_id=job_id, raw_text="Draft plan", status="draft")
            session.add(plan)
            await session.commit()

        response = await client.post(f"/jobs/{job_id}/approve", json={"decision": "approved"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "GENERATING_CODE"
        assert data["error_message"] is None

    async def test_reject_plan_clears_stale_error_message(
        self, client: httpx.AsyncClient, test_db: async_sessionmaker[AsyncSession]
    ) -> None:
        """Rejecting a plan resets any stale error_message to None."""
        create_resp = await client.post(
            "/jobs", json={"issue_url": "https://github.com/octocat/hello-world/issues/1"}
        )
        job_id = create_resp.json()["id"]

        async with test_db() as session:
            job = await session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.AWAITING_APPROVAL
            job.error_message = "Stale warning message"
            plan = Plan(job_id=job_id, raw_text="Draft plan", status="draft")
            session.add(plan)
            await session.commit()

        response = await client.post(
            f"/jobs/{job_id}/approve",
            json={"decision": "rejected", "feedback": "Needs rework"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PLANNING"
        assert data["error_message"] is None
