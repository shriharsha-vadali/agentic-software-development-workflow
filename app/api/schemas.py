"""
app/api/schemas.py
------------------
Pydantic v2 schemas for API request validation, response serialization,
and data transfer across the Agentic Software Development Workflow.

Architecture decisions
----------------------
Schema vs. Model separation
    SQLAlchemy ORM models (app/db/models.py) define the persistent database
    layer. Pydantic schemas defined here represent the public HTTP API contract.
    Keeping them separate ensures database implementation details (e.g. internal
    foreign keys, cascade rules) do not leak into API responses, and allows
    the API and database to evolve independently.

Pydantic v2 conventions
    - Uses `model_config = ConfigDict(from_attributes=True)` so schemas can
      serialize directly from SQLAlchemy ORM instances (replacing v1 `orm_mode = True`).
    - Uses `@field_validator` and `@model_validator(mode="after")` for input sanitization
      and cross-field consistency.
    - Uses typed `Field` annotations with constraints (`gt=0`, `min_length=1`).

Validation strategy for GitHub resources
    - `CreateJobRequest` supports flexible and strict inputs: a developer can
      pass either a `repo_url` and `issue_number`, or a full `issue_url`.
    - Both formats validate that the target repository belongs to `github.com`
      and follows GitHub's naming conventions for owner and repository.
    - Strictly rejects embedded credentials, unexpected ports, query parameters,
      URL fragments, percent-encoding, and path traversal sequences.
    - Standardizes URLs by stripping trailing `.git` and slashes.

Async SQLAlchemy ORM serialization
    - Schemas do not execute database queries.
    - When serializing entities with relationships (e.g. `JobDetailResponse`),
      API routes must eagerly load relationships (e.g. `selectinload(Job.plan)`)
      to avoid `sqlalchemy.exc.MissingGreenlet` errors outside an async session.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from app.db.models import JobStatus

# Regex pattern matching valid GitHub usernames and organization names:
# Alphanumeric characters or single hyphens; cannot start or end with a hyphen; max 39 chars.
_GITHUB_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")

# Regex pattern matching valid GitHub repository names:
# Alphanumeric characters, periods, hyphens, and underscores; max 100 chars.
_GITHUB_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,100}$")

# Regex pattern matching positive integer issue numbers containing only ASCII digits:
_ASCII_DIGITS_PATTERN = re.compile(r"^[0-9]+$")


# =============================================================================
# URL Helper Function
# =============================================================================


def _validate_github_url(
    url: HttpUrl,
    *,
    expected_segments: int,
    url_name: str,
) -> tuple[str, str, int | None]:
    """
    Strictly parse and validate a github.com URL.

    Checks:
    - Rejects embedded user credentials (username/password).
    - Rejects explicit non-standard ports (anything other than None, 80, 443).
    - Rejects query parameters and URL fragments.
    - Enforces hostname to be github.com or www.github.com.
    - Rejects percent-encoded characters (%xx), relative traversals, or consecutive slashes.
    - Enforces expected path format:
        For repo_url:  /<owner>/<repo> (expected_segments=2)
        For issue_url: /<owner>/<repo>/issues/<issue_number> (expected_segments=4)
    - Enforces GitHub owner and repository naming syntax rules.
    - Normalizes .git suffix removal.

    Returns:
        tuple of (owner, repo, issue_number_or_none)
    """
    raw_str = str(url)
    parsed = urlparse(raw_str)

    # 1. Embedded credentials check
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{url_name} must not contain embedded user credentials.")

    # 2. Scheme and port checks
    try:
        port = parsed.port
    except ValueError as err:
        raise ValueError(f"Malformed port in {url_name}: {err}.") from None

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme '{scheme}'. Only 'http' and 'https' are allowed.")

    if port is not None:
        if scheme == "http" and port != 80:
            raise ValueError(
                f"Invalid port {port} for HTTP scheme in {url_name}. "
                "Only default port 80 is allowed."
            )
        if scheme == "https" and port != 443:
            raise ValueError(
                f"Invalid port {port} for HTTPS scheme in {url_name}. "
                "Only default port 443 is allowed."
            )

    # 3. Query string, fragment, and empty marker checks
    if "?" in raw_str:
        raise ValueError(f"{url_name} must not contain query parameters or '?' marker.")
    if "#" in raw_str:
        raise ValueError(f"{url_name} must not contain URL fragments or '#' marker.")

    # 4. Hostname check
    hostname = (parsed.hostname or "").lower()
    if hostname not in ("github.com", "www.github.com"):
        raise ValueError(
            f"Only github.com URLs are currently supported. "
            f"Received host for {url_name}: '{hostname}'."
        )

    # 5. Path hygiene checks
    if "%" in parsed.path:
        raise ValueError(f"{url_name} path must not contain percent-encoded characters.")
    if "//" in parsed.path:
        raise ValueError(f"{url_name} path must not contain consecutive slashes.")

    stripped_path = parsed.path.strip("/")
    if not stripped_path:
        raise ValueError(f"{url_name} path cannot be empty.")

    segments = stripped_path.split("/")
    if any(seg in (".", "..") for seg in segments):
        raise ValueError(f"{url_name} path must not contain relative navigation segments.")

    issue_number: int | None = None

    # 6. Segment structure validation
    if expected_segments == 4:
        # Expected: <owner>/<repo>/issues/<issue_number>
        if len(segments) != 4 or segments[2] != "issues":
            raise ValueError(
                f"Invalid GitHub issue URL path format: '{parsed.path}'. "
                "Expected format: https://github.com/<owner>/<repo>/issues/<issue_number>"
            )

        raw_num = segments[3]
        if not _ASCII_DIGITS_PATTERN.fullmatch(raw_num) or int(raw_num) <= 0:
            raise ValueError(
                f"Invalid issue number in URL: '{raw_num}'. "
                "Must be a positive integer composed of ASCII digits."
            )
        issue_number = int(raw_num)

    elif expected_segments == 2:
        # Expected: <owner>/<repo>
        if len(segments) != 2:
            raise ValueError(
                f"Invalid GitHub repository URL path: '{parsed.path}'. "
                "Expected format: https://github.com/<owner>/<repo>"
            )
    else:
        raise ValueError(f"Unsupported segment expectation: {expected_segments}")

    owner = segments[0]
    raw_repo = segments[1]
    repo = raw_repo[:-4] if raw_repo.lower().endswith(".git") else raw_repo

    # 7. Owner and repo syntax checks
    if not _GITHUB_OWNER_PATTERN.match(owner):
        raise ValueError(
            f"Invalid GitHub owner/organization name: '{owner}'. "
            "Must be alphanumeric with optional hyphens and cannot begin/end with a hyphen."
        )
    if not _GITHUB_REPO_PATTERN.match(repo):
        raise ValueError(
            f"Invalid GitHub repository name: '{repo}'. "
            "Must be 1-100 characters containing alphanumeric characters, "
            "underscores, hyphens, or periods."
        )

    return owner, repo, issue_number


# =============================================================================
# Plan Schemas
# =============================================================================


class PlanStep(BaseModel):
    """
    Structured step representation within an implementation plan.

    Used when parsing raw plan text into structured execution units
    for Phase 2 code generation.
    """

    model_config = ConfigDict(from_attributes=True)

    step_number: int = Field(
        ...,
        gt=0,
        description="Sequential index of the execution step (1-indexed, must be positive).",
        examples=[1],
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of the action to be taken in this step.",
        examples=["Create database models in app/db/models.py"],
    )
    file_paths: list[str] = Field(
        default_factory=list,
        description="Files targeted for creation, modification, or inspection in this step.",
        examples=[["app/db/models.py", "tests/unit/test_models.py"]],
    )
    difficulty: Literal["easy", "medium", "hard", "low", "high"] | None = Field(
        default=None,
        description="Estimated difficulty level ('easy', 'medium', 'hard', 'low', 'high').",
        examples=["medium"],
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tools, CLI utilities, or agents required for this step (e.g. ['git', 'pytest'])."
        ),
        examples=[["git", "pytest"]],
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, val: str) -> str:
        """Ensure description is not empty or whitespace only."""
        cleaned = val.strip()
        if not cleaned:
            raise ValueError("Description must not be empty or whitespace only.")
        return cleaned

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, val: str | None) -> str | None:
        """Normalize difficulty strings to lowercase and strip whitespace."""
        if val is None:
            return None
        if isinstance(val, str):
            cleaned = val.strip().lower()
            return cleaned if cleaned else None
        return val

    @field_validator("file_paths", "required_tools")
    @classmethod
    def sanitize_string_list(cls, items: list[str]) -> list[str]:
        """Strip whitespace, discard empty entries, and remove duplicates."""
        cleaned: list[str] = []
        for item in items:
            s = item.strip()
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned


class PlanResponse(BaseModel):
    """
    Response schema for an LLM-generated implementation plan.

    Directly maps to the SQLAlchemy `Plan` model via `from_attributes=True`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(
        ...,
        description="Unique identifier of the plan.",
        examples=[1],
    )
    job_id: int = Field(
        ...,
        description="Identifier of the associated job.",
        examples=[42],
    )
    raw_text: str = Field(
        ...,
        description="Verbatim markdown text of the implementation plan generated by the LLM.",
        examples=["## Implementation Plan\n1. Define ORM models\n2. Add test suite"],
    )
    steps: list[PlanStep] | None = Field(
        default=None,
        description="Parsed structured steps if available (Phase 2), or None if unparsed.",
    )
    status: str = Field(
        ...,
        description="Review state of the plan: 'draft', 'approved', or 'rejected'.",
        examples=["draft"],
    )
    human_feedback: str | None = Field(
        default=None,
        description="Reviewer feedback or rejection reason, if provided.",
        examples=["Please include validation for empty strings"],
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the plan was generated.",
    )

    @field_validator("steps")
    @classmethod
    def validate_unique_steps(cls, steps: list[PlanStep] | None) -> list[PlanStep] | None:
        """Reject plans with duplicate step numbers."""
        if steps is not None:
            seen_numbers: set[int] = set()
            for step in steps:
                if step.step_number in seen_numbers:
                    raise ValueError(f"Duplicate step_number found: {step.step_number}")
                seen_numbers.add(step.step_number)
        return steps


class ApprovePlanRequest(BaseModel):
    """
    Request schema for approving or rejecting a generated plan.

    Sent to `POST /jobs/{id}/approve` when the agent is in AWAITING_APPROVAL state.
    """

    decision: Literal["approved", "rejected"] = Field(
        ...,
        description="Human reviewer decision: 'approved' to proceed or 'rejected' to revise.",
        examples=["approved"],
    )
    feedback: str | None = Field(
        default=None,
        description="Optional feedback, suggestions, or rejection reason passed back to the agent.",
        examples=["Looks good, proceed to implementation."],
    )

    @field_validator("feedback")
    @classmethod
    def sanitize_feedback(cls, value: str | None) -> str | None:
        """Strip surrounding whitespace if feedback is provided."""
        if value is not None:
            cleaned = value.strip()
            return cleaned if cleaned else None
        return None


# =============================================================================
# Artifact Schemas
# =============================================================================


class ArtifactResponse(BaseModel):
    """
    Response schema representing an artifact generated during pipeline execution.

    Directly maps to the SQLAlchemy `Artifact` model via `from_attributes=True`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(
        ...,
        description="Unique identifier of the artifact.",
        examples=[10],
    )
    job_id: int = Field(
        ...,
        description="Identifier of the associated job.",
        examples=[42],
    )
    artifact_type: str = Field(
        ...,
        description=(
            "Kind of artifact produced "
            "(e.g. 'issue_data', 'plan', 'diff', 'test_output', 'pr_url')."
        ),
        examples=["issue_data"],
    )
    content: str = Field(
        ...,
        description="The artifact payload (JSON string or plain text).",
        examples=['{"title": "Add user authentication"}'],
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the artifact was stored.",
    )


# =============================================================================
# Job Request Schemas
# =============================================================================


class CreateJobRequest(BaseModel):
    """
    Request payload to trigger an agentic workflow for a GitHub issue.

    Accepted input formats:
      1. Full issue URL:
         `issue_url="https://github.com/owner/repo/issues/42"`
      2. Repository URL + issue number:
         `repo_url="https://github.com/owner/repo"`, `issue_number=42`

    If both are supplied, they must point to the same repository and issue number.
    """

    issue_url: HttpUrl | None = Field(
        default=None,
        description="Full URL of the target GitHub issue (e.g. https://github.com/owner/repo/issues/42).",
        examples=["https://github.com/octocat/hello-world/issues/42"],
    )
    repo_url: HttpUrl | None = Field(
        default=None,
        description="URL of the GitHub repository (e.g. https://github.com/owner/repo).",
        examples=["https://github.com/octocat/hello-world"],
    )
    issue_number: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="GitHub issue number (> 0, positive integer).",
        examples=[42],
    )

    # Derived attributes stored after validation
    _repo_owner: str = ""
    _repo_name: str = ""
    _resolved_issue_number: int = 0
    _canonical_issue_url: str = ""

    @field_validator("issue_number", mode="before")
    @classmethod
    def validate_issue_number_type(cls, v: object) -> object:
        """Reject booleans and non-integer types for issue_number."""
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("Issue number must be an integer, not a boolean.")
        if not isinstance(v, int):
            raise ValueError(f"Issue number must be an integer, received {type(v).__name__}.")
        return v

    @field_validator("issue_url", "repo_url", mode="before")
    @classmethod
    def validate_raw_github_url(cls, v: object) -> object:
        """
        Pre-parse check on incoming URL strings before Pydantic normalization.

        Ensures that credentials, query strings, fragments, unexpected ports,
        percent-encoded paths, directory traversals ('..' or '.'), and consecutive
        slashes are rejected cleanly.
        """
        if v is None:
            return None
        raw_str = str(v)
        parsed = urlparse(raw_str)

        # 1. Embedded credentials check
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URL must not contain embedded user credentials.")

        # 2. Scheme and port checks
        try:
            port = parsed.port
        except ValueError as err:
            raise ValueError(f"Malformed port in URL: {err}.") from None

        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Unsupported URL scheme '{scheme}'. Only 'http' and 'https' are allowed."
            )

        if port is not None:
            if scheme == "http" and port != 80:
                raise ValueError(
                    f"Invalid port {port} for HTTP scheme. Only default port 80 is allowed."
                )
            if scheme == "https" and port != 443:
                raise ValueError(
                    f"Invalid port {port} for HTTPS scheme. Only default port 443 is allowed."
                )

        # 3. Query string, fragment, and empty marker checks
        if "?" in raw_str:
            raise ValueError("URL must not contain query parameters or '?' marker.")
        if "#" in raw_str:
            raise ValueError("URL must not contain URL fragments or '#' marker.")

        # 4. Hostname check
        hostname = (parsed.hostname or "").lower()
        if hostname not in ("github.com", "www.github.com"):
            raise ValueError(
                f"Only github.com URLs are currently supported. Received host: '{hostname}'."
            )

        # 5. Path hygiene checks
        if "%" in parsed.path:
            raise ValueError("URL path must not contain percent-encoded characters.")
        if "//" in parsed.path:
            raise ValueError("URL path must not contain consecutive slashes.")

        stripped_path = parsed.path.strip("/")
        if not stripped_path:
            raise ValueError("URL path cannot be empty.")

        segments = stripped_path.split("/")
        if any(seg in (".", "..") for seg in segments):
            raise ValueError("URL path must not contain relative navigation segments.")

        # Require ASCII digits in URL issue-number segments
        if len(segments) == 4 and segments[2] == "issues":
            raw_num = segments[3]
            if not _ASCII_DIGITS_PATTERN.fullmatch(raw_num) or int(raw_num) <= 0:
                raise ValueError(
                    f"Invalid issue number in URL: '{raw_num}'. "
                    "Must be a positive integer composed of ASCII digits."
                )

        return v

    @property
    def repo_owner(self) -> str:
        """Extracted GitHub organization or user account name."""
        return self._repo_owner

    @property
    def repo_name(self) -> str:
        """Extracted GitHub repository name."""
        return self._repo_name

    @property
    def resolved_issue_number(self) -> int:
        """Resolved positive integer issue number."""
        return self._resolved_issue_number

    @property
    def canonical_issue_url(self) -> str:
        """Normalized canonical GitHub issue URL (e.g. https://github.com/owner/repo/issues/42)."""
        return self._canonical_issue_url

    @model_validator(mode="after")
    def validate_and_resolve_targets(self) -> CreateJobRequest:
        """
        Validate GitHub URLs, enforce host restrictions, extract components,
        and ensure inputs are mutually consistent.
        """
        has_issue_url = self.issue_url is not None
        has_repo_and_number = self.repo_url is not None and self.issue_number is not None

        if not has_issue_url and not has_repo_and_number:
            if self.repo_url is not None and self.issue_number is None:
                raise ValueError("When 'repo_url' is provided, 'issue_number' is also required.")
            if self.issue_number is not None and self.repo_url is None:
                raise ValueError("When 'issue_number' is provided, 'repo_url' is also required.")
            raise ValueError(
                "Either 'issue_url' OR both 'repo_url' and 'issue_number' must be provided."
            )

        owner: str = ""
        repo: str = ""
        number: int = 0

        # Scenario A: Full issue_url is provided
        if self.issue_url is not None:
            owner, repo, extracted_number = _validate_github_url(
                self.issue_url,
                expected_segments=4,
                url_name="issue_url",
            )
            assert extracted_number is not None
            number = extracted_number

        # Scenario B: repo_url + issue_number are provided
        if self.repo_url is not None and self.issue_number is not None:
            b_owner, b_repo, _ = _validate_github_url(
                self.repo_url,
                expected_segments=2,
                url_name="repo_url",
            )
            b_number = self.issue_number

            # If both were provided, verify they agree on owner, repo, and issue number
            if has_issue_url:
                if (owner.lower(), repo.lower(), number) != (
                    b_owner.lower(),
                    b_repo.lower(),
                    b_number,
                ):
                    raise ValueError(
                        f"Mismatch between 'issue_url' ({owner}/{repo}#{number}) "
                        f"and 'repo_url'+'issue_number' ({b_owner}/{b_repo}#{b_number})."
                    )
            else:
                owner = b_owner
                repo = b_repo
                number = b_number

        # Store resolved, normalized target information
        self._repo_owner = owner
        self._repo_name = repo
        self._resolved_issue_number = number
        self._canonical_issue_url = f"https://github.com/{owner}/{repo}/issues/{number}"

        return self


class UpdateJobStatusRequest(BaseModel):
    """
    Request schema to transition a job's status.

    Sent to `PATCH /jobs/{id}/status` to advance or fail the workflow state machine.
    """

    model_config = ConfigDict(from_attributes=True)

    status: JobStatus = Field(
        ...,
        description="Target state machine status to transition to.",
        examples=[JobStatus.FETCHING_ISSUE],
    )
    error_message: str | None = Field(
        default=None,
        description=(
            "Optional error message explaining reason for failure if transitioning to FAILED."
        ),
        examples=["GitHub repository not accessible."],
    )

    @field_validator("error_message")
    @classmethod
    def sanitize_error_message(cls, val: str | None) -> str | None:
        """Strip surrounding whitespace if error_message is provided."""
        if val is not None:
            cleaned = val.strip()
            return cleaned if cleaned else None
        return None


# =============================================================================
# Job Response Schemas
# =============================================================================


class JobResponse(BaseModel):
    """
    Standard summary response for a single agent workflow job.

    Used when listing jobs or returning initial creation acknowledgement.
    Maps directly to the SQLAlchemy `Job` model via `from_attributes=True`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(
        ...,
        description="Unique identifier of the workflow job.",
        examples=[1],
    )
    issue_url: str = Field(
        ...,
        description="Full URL of the target GitHub issue.",
        examples=["https://github.com/octocat/hello-world/issues/42"],
    )
    repo_owner: str = Field(
        ...,
        description="GitHub organization or user owning the repository.",
        examples=["octocat"],
    )
    repo_name: str = Field(
        ...,
        description="GitHub repository name.",
        examples=["hello-world"],
    )
    issue_number: int = Field(
        ...,
        description="Target GitHub issue number.",
        examples=[42],
    )
    status: JobStatus = Field(
        ...,
        description="Current state machine position of the agent workflow.",
        examples=[JobStatus.PENDING],
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the job was created.",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp when the job status or records were last updated.",
    )


class JobDetailResponse(JobResponse):
    """
    Comprehensive job details including error status, associated plan, and artifacts.

    Used for `GET /jobs/{id}` and state inspection.

    ORM Serialization & Async SQLAlchemy Note:
    -----------------------------------------
    This schema defines `model_config = ConfigDict(from_attributes=True)`.
    When serializing from an asynchronous SQLAlchemy `Job` instance outside an
    active database session, relationship attributes (`job.plan` and `job.artifacts`)
    MUST be eagerly loaded by the repository/route query using `selectinload`:
        stmt = (
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.plan), selectinload(Job.artifacts))
        )
    Attempting to serialize an ORM instance where relationships are lazy-loaded
    outside greenlet context will raise `sqlalchemy.exc.MissingGreenlet`.
    To prevent this, Pydantic schemas never trigger database I/O, and API route
    handlers are responsible for eager loading required relationships.
    """

    workspace_path: str | None = Field(
        default=None,
        description="Local disk path of the cloned repository workspace, if initialized.",
        examples=["./workspaces/octocat__hello-world__42"],
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if the job transitioned to the FAILED state.",
        examples=["Repository not found or access denied."],
    )
    plan: PlanResponse | None = Field(
        default=None,
        description="The implementation plan generated for this job, if available.",
    )
    artifacts: list[ArtifactResponse] = Field(
        default_factory=list,
        description="List of all artifacts generated during this job's lifecycle.",
    )


# =============================================================================
# Standard Error Response
# =============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error envelope returned on 4xx/5xx HTTP responses.

    Provides a consistent format for API consumers across all endpoints.
    """

    model_config = ConfigDict(from_attributes=True)

    detail: str = Field(
        ...,
        description="Human-readable explanation of the error.",
        examples=["Job with id 999 not found."],
    )
    error_code: str | None = Field(
        default=None,
        description=(
            "Machine-readable error classification code (e.g. 'JOB_NOT_FOUND', 'INVALID_STATE')."
        ),
        examples=["JOB_NOT_FOUND"],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="ISO 8601 UTC timestamp indicating when the error occurred.",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, val: datetime) -> datetime:
        """
        Guarantee that timestamp is always timezone-aware UTC.

        Assigns UTC if naive; converts other timezone offsets to UTC.
        """
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
