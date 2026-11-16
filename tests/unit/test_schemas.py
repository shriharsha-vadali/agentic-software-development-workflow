"""
tests/unit/test_schemas.py
--------------------------
Unit tests for Pydantic v2 API schemas defined in app/api/schemas.py.

Covers:
- CreateJobRequest input validation (issue_url, repo_url + issue_number, edge cases)
- GitHub URL syntax & host restrictions (credentials, ports, query strings,
  fragments, path traversal)
- PlanStep, PlanResponse, and ApprovePlanRequest validation (step numbers,
  duplicates, difficulty, tools)
- ArtifactResponse validation and ORM conversion
- JobResponse and JobDetailResponse ORM validation (from_attributes=True)
- ErrorResponse standard format, timezone awareness, and instance independence
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ApprovePlanRequest,
    ArtifactResponse,
    CreateJobRequest,
    ErrorResponse,
    JobDetailResponse,
    JobResponse,
    PlanResponse,
    PlanStep,
)
from app.db.models import Artifact, Job, JobStatus, Plan

# =============================================================================
# Tests — CreateJobRequest Validation
# =============================================================================


class TestCreateJobRequest:
    """Verify input validation, host checks, and normalisation in CreateJobRequest."""

    def test_valid_with_issue_url(self) -> None:
        """Providing a valid GitHub issue_url extracts all components properly."""
        req = CreateJobRequest(issue_url="https://github.com/octocat/hello-world/issues/42")

        assert req.repo_owner == "octocat"
        assert req.repo_name == "hello-world"
        assert req.resolved_issue_number == 42
        assert req.canonical_issue_url == "https://github.com/octocat/hello-world/issues/42"

    def test_valid_with_repo_url_and_issue_number(self) -> None:
        """Providing repo_url and issue_number synthesizes canonical_issue_url correctly."""
        req = CreateJobRequest(
            repo_url="https://github.com/octocat/hello-world",
            issue_number=101,
        )

        assert req.repo_owner == "octocat"
        assert req.repo_name == "hello-world"
        assert req.resolved_issue_number == 101
        assert req.canonical_issue_url == "https://github.com/octocat/hello-world/issues/101"

    def test_valid_with_matching_both_inputs(self) -> None:
        """Supplying both issue_url and repo_url + issue_number succeeds when they match."""
        req = CreateJobRequest(
            issue_url="https://github.com/fastapi/fastapi/issues/123",
            repo_url="https://github.com/fastapi/fastapi",
            issue_number=123,
        )

        assert req.repo_owner == "fastapi"
        assert req.repo_name == "fastapi"
        assert req.resolved_issue_number == 123

    def test_strips_dot_git_suffix(self) -> None:
        """Repos with .git extension are properly normalized."""
        req = CreateJobRequest(
            repo_url="https://github.com/pydantic/pydantic.git",
            issue_number=5,
        )

        assert req.repo_name == "pydantic"
        assert req.canonical_issue_url == "https://github.com/pydantic/pydantic/issues/5"

    def test_strips_dot_git_suffix_from_issue_url(self) -> None:
        """issue_url containing .git before /issues is normalized."""
        req = CreateJobRequest(issue_url="https://github.com/pydantic/pydantic.git/issues/5")

        assert req.repo_name == "pydantic"
        assert req.canonical_issue_url == "https://github.com/pydantic/pydantic/issues/5"

    def test_handles_www_github_com(self) -> None:
        """www.github.com is accepted as valid hostname."""
        req = CreateJobRequest(issue_url="https://www.github.com/octocat/hello-world/issues/1")

        assert req.repo_owner == "octocat"
        assert req.resolved_issue_number == 1

    def test_missing_all_fields_raises_validation_error(self) -> None:
        """Fails when neither issue_url nor repo_url + issue_number are provided."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest()

        assert "Either 'issue_url' OR both 'repo_url' and 'issue_number'" in str(exc.value)

    def test_repo_url_without_issue_number_raises_validation_error(self) -> None:
        """repo_url alone without issue_number is rejected with helpful guidance."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo")

        assert "When 'repo_url' is provided, 'issue_number' is also required." in str(exc.value)

    def test_issue_number_without_repo_url_raises_validation_error(self) -> None:
        """issue_number alone without repo_url is rejected with helpful guidance."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_number=42)

        assert "When 'issue_number' is provided, 'repo_url' is also required." in str(exc.value)

    def test_mismatched_inputs_raises_validation_error(self) -> None:
        """Fails if issue_url and repo_url disagree."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(
                issue_url="https://github.com/owner1/repo1/issues/42",
                repo_url="https://github.com/owner2/repo2",
                issue_number=42,
            )

        assert "Mismatch between 'issue_url'" in str(exc.value)

    def test_mismatched_issue_number_raises_validation_error(self) -> None:
        """Fails if issue_url issue number disagrees with issue_number argument."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(
                issue_url="https://github.com/owner/repo/issues/42",
                repo_url="https://github.com/owner/repo",
                issue_number=99,
            )

        assert "Mismatch between 'issue_url'" in str(exc.value)

    def test_non_github_host_rejected(self) -> None:
        """Rejects URLs from non-github hosts (e.g. gitlab or bitbucket)."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://gitlab.com/owner/repo/issues/42")

        assert "Only github.com URLs are currently supported" in str(exc.value)

    def test_invalid_issue_url_path_format(self) -> None:
        """Rejects paths that do not conform to /<owner>/<repo>/issues/<num>."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/pull/42")

        assert "Invalid GitHub issue URL path format" in str(exc.value)

    def test_non_integer_issue_number_in_url(self) -> None:
        """Rejects non-numeric issue numbers in URL path."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/not-a-number")

        assert "Invalid issue number in URL" in str(exc.value)

    def test_zero_or_negative_issue_number_rejected(self) -> None:
        """Issue numbers <= 0 are rejected."""
        with pytest.raises(ValidationError):
            CreateJobRequest(repo_url="https://github.com/owner/repo", issue_number=0)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/0")
        assert "Must be a positive integer" in str(exc.value)

    def test_invalid_owner_format_rejected(self) -> None:
        """Owner names starting with hyphens or invalid characters are rejected."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/-invalid-owner/repo", issue_number=1)

        assert "Invalid GitHub owner/organization name" in str(exc.value)

    def test_invalid_repo_format_rejected(self) -> None:
        """Empty or invalid repo names are rejected."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo$bad", issue_number=1)

        assert "Invalid GitHub repository name" in str(exc.value)

    # --- Edge Cases Review: Credentials, Ports, Queries, Fragments, Traversals ---

    def test_reject_embedded_credentials(self) -> None:
        """Rejects URLs containing user credentials (e.g. https://user:pass@github.com)."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://user:pass@github.com/owner/repo/issues/1")
        assert "must not contain embedded user credentials" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://user:pass@github.com/owner/repo", issue_number=1)
        assert "must not contain embedded user credentials" in str(exc.value)

    def test_valid_explicit_default_ports(self) -> None:
        """Explicit standard ports for their respective schemes are accepted."""
        req_http = CreateJobRequest(issue_url="http://github.com:80/octocat/hello-world/issues/1")
        assert req_http.resolved_issue_number == 1

        req_https = CreateJobRequest(
            issue_url="https://github.com:443/octocat/hello-world/issues/1"
        )
        assert req_https.resolved_issue_number == 1

    def test_reject_unexpected_port(self) -> None:
        """Rejects URLs containing non-standard ports (e.g. :8080)."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com:8080/owner/repo/issues/1")
        assert "Invalid port 8080 for HTTPS scheme" in str(exc.value)

    def test_reject_mismatched_scheme_port(self) -> None:
        """Rejects ports that do not match the scheme's standard default."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="http://github.com:443/owner/repo/issues/1")
        assert "Invalid port 443 for HTTP scheme" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com:80/owner/repo/issues/1")
        assert "Invalid port 80 for HTTPS scheme" in str(exc.value)

    def test_reject_malformed_ports(self) -> None:
        """Rejects non-numeric or out-of-range ports."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com:abc/owner/repo/issues/1")
        assert "Malformed port in URL" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com:999999/owner/repo/issues/1")
        assert "Malformed port in URL" in str(exc.value)

    def test_reject_empty_query_marker(self) -> None:
        """Rejects URLs with empty query string marker '?'."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/1?")
        assert "must not contain query parameters or '?' marker" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo?", issue_number=1)
        assert "must not contain query parameters or '?' marker" in str(exc.value)

    def test_reject_empty_fragment_marker(self) -> None:
        """Rejects URLs with empty fragment marker '#'."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/1#")
        assert "must not contain URL fragments or '#' marker" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo#", issue_number=1)
        assert "must not contain URL fragments or '#' marker" in str(exc.value)

    def test_reject_query_string(self) -> None:
        """Rejects URLs with query parameters."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/1?sort=created")
        assert "must not contain query parameters" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo?tab=readme", issue_number=1)
        assert "must not contain query parameters" in str(exc.value)

    def test_reject_url_fragment(self) -> None:
        """Rejects URLs with fragment identifiers."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/1#issuecomment-42")
        assert "must not contain URL fragments" in str(exc.value)

    def test_reject_percent_encoded_path(self) -> None:
        """Rejects URLs with percent-encoded characters in path."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner%20name/repo/issues/1")
        assert "must not contain percent-encoded characters" in str(exc.value)

    def test_reject_relative_navigation_path(self) -> None:
        """Rejects URLs with directory traversal navigation segments ('.' or '..')."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/../repo/issues/1")
        assert "must not contain relative navigation segments" in str(exc.value)

    def test_reject_consecutive_slashes(self) -> None:
        """Rejects URLs with malformed consecutive slashes in path."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner//repo/issues/1")
        assert "must not contain consecutive slashes" in str(exc.value)

    def test_reject_signed_or_malformed_issue_number(self) -> None:
        """Rejects issue numbers with explicit signs or trailing characters."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/+42")
        assert "Must be a positive integer" in str(exc.value)

    def test_reject_unicode_digits_in_issue_url(self) -> None:
        """Rejects non-ASCII/Unicode digits in issue URL."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(issue_url="https://github.com/owner/repo/issues/١٢٣")
        assert "Must be a positive integer composed of ASCII digits" in str(exc.value)

    def test_reject_boolean_issue_number(self) -> None:
        """Rejects booleans passed as issue_number."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo", issue_number=True)
        assert "Issue number must be an integer, not a boolean" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(repo_url="https://github.com/owner/repo", issue_number=False)
        assert "Issue number must be an integer, not a boolean" in str(exc.value)

    def test_reject_non_integer_issue_number(self) -> None:
        """Rejects floats and strings passed as issue_number."""
        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(
                repo_url="https://github.com/owner/repo",
                issue_number=12.5,  # type: ignore[arg-type]
            )
        assert "Issue number must be an integer" in str(exc.value)

        with pytest.raises(ValidationError) as exc:
            CreateJobRequest(
                repo_url="https://github.com/owner/repo",
                issue_number="42",  # type: ignore[arg-type]
            )
        assert "Issue number must be an integer" in str(exc.value)


# =============================================================================
# Tests — Plan Schemas
# =============================================================================


class TestPlanSchemas:
    """Verify PlanStep, PlanResponse, and ApprovePlanRequest."""

    def test_plan_step_valid(self) -> None:
        """PlanStep model validates fields correctly."""
        step = PlanStep(
            step_number=1,
            description="Create API routes",
            file_paths=["app/api/jobs.py"],
            difficulty="medium",
            required_tools=["git", "pytest"],
        )

        assert step.step_number == 1
        assert step.description == "Create API routes"
        assert step.file_paths == ["app/api/jobs.py"]
        assert step.difficulty == "medium"
        assert step.required_tools == ["git", "pytest"]

    def test_plan_step_default_fields(self) -> None:
        """file_paths, difficulty, and required_tools have sensible defaults."""
        step = PlanStep(step_number=2, description="Write tests")
        assert step.file_paths == []
        assert step.difficulty is None
        assert step.required_tools == []

    def test_plan_step_invalid_number_raises(self) -> None:
        """step_number <= 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            PlanStep(step_number=0, description="Step 0")

        with pytest.raises(ValidationError):
            PlanStep(step_number=-1, description="Negative step")

    def test_plan_step_empty_description_raises(self) -> None:
        """Empty or whitespace-only description is rejected."""
        with pytest.raises(ValidationError):
            PlanStep(step_number=1, description="")

        with pytest.raises(ValidationError) as exc:
            PlanStep(step_number=1, description="   ")
        assert "Description must not be empty or whitespace only" in str(exc.value)

    def test_plan_step_difficulty_normalization(self) -> None:
        """Difficulty strings are normalized to lowercase."""
        step = PlanStep(step_number=1, description="Test step", difficulty="Medium")
        assert step.difficulty == "medium"

        step2 = PlanStep(step_number=1, description="Test step", difficulty="  HARD  ")
        assert step2.difficulty == "hard"

    def test_plan_step_invalid_difficulty_raises(self) -> None:
        """Unrecognized difficulty values are rejected."""
        with pytest.raises(ValidationError):
            PlanStep(step_number=1, description="Test step", difficulty="impossible")

    def test_plan_step_list_sanitization_and_deduplication(self) -> None:
        """file_paths and required_tools are trimmed, deduplicated, and empty items removed."""
        step = PlanStep(
            step_number=1,
            description="Step",
            file_paths=["  app/models.py  ", "", "app/models.py", "  ", "tests/test.py"],
            required_tools=["git", " pytest ", "git", ""],
        )
        assert step.file_paths == ["app/models.py", "tests/test.py"]
        assert step.required_tools == ["git", "pytest"]

    def test_plan_response_from_orm(self) -> None:
        """
        PlanResponse serializes from an in-memory SQLAlchemy Plan instance
        (pure attribute mapping without database I/O).
        """
        orm_plan = Plan(
            id=1,
            job_id=42,
            raw_text="1. Add models\n2. Add routes",
            steps=[{"step_number": 1, "description": "Add models", "file_paths": []}],
            status="draft",
            human_feedback=None,
            created_at=datetime.now(UTC),
        )

        schema = PlanResponse.model_validate(orm_plan)
        assert schema.id == 1
        assert schema.job_id == 42
        assert schema.raw_text == orm_plan.raw_text
        assert schema.status == "draft"
        assert schema.steps is not None
        assert len(schema.steps) == 1
        assert schema.steps[0].step_number == 1

    def test_plan_response_rejects_duplicate_step_numbers(self) -> None:
        """PlanResponse rejects a plan containing duplicate step_number values."""
        steps = [
            PlanStep(step_number=1, description="First step"),
            PlanStep(step_number=1, description="Conflicting step 1"),
        ]
        with pytest.raises(ValidationError) as exc:
            PlanResponse(
                id=1,
                job_id=42,
                raw_text="Plan",
                steps=steps,
                status="draft",
                created_at=datetime.now(UTC),
            )
        assert "Duplicate step_number found: 1" in str(exc.value)

    def test_approve_plan_request_approved(self) -> None:
        """ApprovePlanRequest accepts 'approved' decision."""
        req = ApprovePlanRequest(decision="approved", feedback="  Looks great!  ")
        assert req.decision == "approved"
        # Feedback is stripped of surrounding whitespace
        assert req.feedback == "Looks great!"

    def test_approve_plan_request_rejected_with_feedback(self) -> None:
        """ApprovePlanRequest accepts 'rejected' with reviewer feedback."""
        req = ApprovePlanRequest(decision="rejected", feedback="Please add error handling.")
        assert req.decision == "rejected"
        assert req.feedback == "Please add error handling."

    def test_approve_plan_request_empty_feedback_becomes_none(self) -> None:
        """Blank feedback string normalizes to None."""
        req = ApprovePlanRequest(decision="approved", feedback="   ")
        assert req.feedback is None

    def test_approve_plan_request_invalid_decision_raises(self) -> None:
        """Decisions other than 'approved' or 'rejected' are rejected."""
        with pytest.raises(ValidationError):
            ApprovePlanRequest(decision="maybe")  # type: ignore[arg-type]


# =============================================================================
# Tests — Artifact Schemas
# =============================================================================


class TestArtifactSchemas:
    """Verify ArtifactResponse model and ORM integration."""

    def test_artifact_response_from_dict(self) -> None:
        """ArtifactResponse can be created from raw dict."""
        now = datetime.now(UTC)
        artifact = ArtifactResponse(
            id=1,
            job_id=42,
            artifact_type="issue_data",
            content='{"number": 42}',
            created_at=now,
        )

        assert artifact.id == 1
        assert artifact.artifact_type == "issue_data"
        assert artifact.content == '{"number": 42}'
        assert artifact.created_at == now

    def test_artifact_response_from_orm(self) -> None:
        """
        ArtifactResponse serializes from an in-memory SQLAlchemy Artifact instance
        (pure attribute mapping without database I/O).
        """
        orm_artifact = Artifact(
            id=5,
            job_id=12,
            artifact_type="diff",
            content="--- a/file.py\n+++ b/file.py",
            created_at=datetime.now(UTC),
        )

        schema = ArtifactResponse.model_validate(orm_artifact)
        assert schema.id == 5
        assert schema.job_id == 12
        assert schema.artifact_type == "diff"
        assert schema.content == orm_artifact.content


# =============================================================================
# Tests — Job Response Schemas
# =============================================================================


class TestJobResponses:
    """Verify JobResponse and JobDetailResponse ORM mapping."""

    def test_job_response_from_orm(self) -> None:
        """JobResponse serializes scalar summary attributes from an in-memory ORM Job."""
        now = datetime.now(UTC)
        orm_job = Job(
            id=10,
            issue_url="https://github.com/octocat/hello-world/issues/1",
            repo_owner="octocat",
            repo_name="hello-world",
            issue_number=1,
            status=JobStatus.FETCHING_ISSUE,
            created_at=now,
            updated_at=now,
        )

        resp = JobResponse.model_validate(orm_job)
        assert resp.id == 10
        assert resp.repo_owner == "octocat"
        assert resp.repo_name == "hello-world"
        assert resp.issue_number == 1
        assert resp.status == JobStatus.FETCHING_ISSUE
        assert resp.created_at == now
        assert resp.updated_at == now

    def test_job_detail_response_with_plan_and_artifacts_from_orm(self) -> None:
        """
        JobDetailResponse serializes pre-populated relationship attributes on an
        in-memory ORM Job without triggering database queries.
        """
        now = datetime.now(UTC)
        orm_job = Job(
            id=1,
            issue_url="https://github.com/octocat/hello-world/issues/1",
            repo_owner="octocat",
            repo_name="hello-world",
            issue_number=1,
            workspace_path="./workspaces/octocat__hello-world__1",
            status=JobStatus.AWAITING_APPROVAL,
            error_message=None,
            created_at=now,
            updated_at=now,
        )

        orm_plan = Plan(
            id=100,
            job_id=1,
            raw_text="Plan content",
            steps=None,
            status="draft",
            human_feedback=None,
            created_at=now,
        )
        orm_job.plan = orm_plan

        orm_artifact = Artifact(
            id=200,
            job_id=1,
            artifact_type="issue_data",
            content="{}",
            created_at=now,
        )
        orm_job.artifacts = [orm_artifact]

        detail = JobDetailResponse.model_validate(orm_job)
        assert detail.id == 1
        assert detail.workspace_path == "./workspaces/octocat__hello-world__1"
        assert detail.status == JobStatus.AWAITING_APPROVAL
        assert detail.plan is not None
        assert detail.plan.id == 100
        assert detail.plan.raw_text == "Plan content"
        assert len(detail.artifacts) == 1
        assert detail.artifacts[0].id == 200

    def test_job_detail_response_null_plan_and_empty_artifacts(self) -> None:
        """
        JobDetailResponse handles None plan and empty artifacts list on an
        in-memory ORM Job without database I/O.
        """
        now = datetime.now(UTC)
        orm_job = Job(
            id=2,
            issue_url="https://github.com/octocat/hello-world/issues/2",
            repo_owner="octocat",
            repo_name="hello-world",
            issue_number=2,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        orm_job.plan = None
        orm_job.artifacts = []

        detail = JobDetailResponse.model_validate(orm_job)
        assert detail.plan is None
        assert detail.artifacts == []
        assert detail.error_message is None


# =============================================================================
# Tests — ErrorResponse
# =============================================================================


class TestErrorResponse:
    """Verify standard ErrorResponse schema."""

    def test_error_response_basic(self) -> None:
        """Basic error response with detail string."""
        err = ErrorResponse(detail="Resource not found")
        assert err.detail == "Resource not found"
        assert err.error_code is None
        assert isinstance(err.timestamp, datetime)
        assert err.timestamp.tzinfo is not None

    def test_error_response_with_code_and_custom_timestamp(self) -> None:
        """Error response with error code and specific timestamp."""
        ts = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)
        err = ErrorResponse(
            detail="Job in invalid state",
            error_code="INVALID_STATE",
            timestamp=ts,
        )

        assert err.detail == "Job in invalid state"
        assert err.error_code == "INVALID_STATE"
        assert err.timestamp == ts

    def test_error_response_independent_instances_and_utc(self) -> None:
        """Default timestamps are independently evaluated and timezone-aware UTC."""
        err1 = ErrorResponse(detail="First error")
        time.sleep(0.002)
        err2 = ErrorResponse(detail="Second error")

        assert err1.timestamp != err2.timestamp
        assert err1.timestamp.tzinfo == UTC
        assert err2.timestamp.tzinfo == UTC

    def test_error_response_naive_timestamp_converted_to_utc(self) -> None:
        """Supplied naive timestamp is normalized to timezone-aware UTC."""
        naive_ts = datetime(2026, 11, 16, 10, 30, 0)
        err = ErrorResponse(detail="Error", timestamp=naive_ts)
        assert err.timestamp.tzinfo == UTC
        assert err.timestamp.year == 2026
        assert err.timestamp.hour == 10

    def test_error_response_converts_non_utc_timezone_to_utc(self) -> None:
        """
        Timezone-aware timestamps with non-UTC offset are converted to UTC,
        preserving the exact same instant.
        """
        non_utc_tz = timezone(timedelta(hours=5, minutes=30))
        local_ts = datetime(2026, 11, 16, 15, 30, 0, tzinfo=non_utc_tz)
        err = ErrorResponse(detail="Offset test", timestamp=local_ts)

        assert err.timestamp.tzinfo == UTC
        assert err.timestamp == local_ts
        assert err.timestamp.hour == 10
        assert err.timestamp.minute == 0
