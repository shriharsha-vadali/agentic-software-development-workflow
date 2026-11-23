"""
app/adapters/github.py
----------------------
Asynchronous GitHub API adapter for the Agentic Software Development Workflow.

Responsibilities:
- Encapsulates all external communication with the GitHub REST API v3.
- Parses and strictly types GitHub repository, issue, label, and comment payloads.
- Handles HTTP transport, authentication, rate limits, timeouts, and pagination.
- Guarantees token safety (never logs or leaks SecretStr credentials).
- Provides FastAPI dependency injection and async context management.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# GitHub REST API Version Header
GITHUB_API_VERSION = "2022-11-28"

# Regex for parsing the GitHub pagination Link header
_LINK_REL_NEXT_PATTERN = re.compile(r'<([^>]+)>;\s*rel="next"')


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class GitHubAdapterError(Exception):
    """Base exception for all GitHub API client and adapter errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GitHubAuthError(GitHubAdapterError):
    """Raised when authentication fails (HTTP 401: bad credentials, expired PAT)."""


class GitHubPermissionError(GitHubAdapterError):
    """Raised when access is forbidden (HTTP 403: missing scopes, enterprise restriction)."""


class GitHubResourceNotFoundError(GitHubAdapterError):
    """Raised when target repository, issue, or comment is not found (HTTP 404)."""


class GitHubRateLimitError(GitHubAdapterError):
    """Raised when GitHub rate limits or secondary rate limits are exceeded (HTTP 403 / 429)."""

    def __init__(
        self,
        message: str,
        status_code: int = 403,
        reset_at: datetime | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.reset_at = reset_at
        self.retry_after = retry_after


class GitHubValidationError(GitHubAdapterError):
    """Raised when GitHub rejects query or input data (HTTP 422 Unprocessable Content)."""


class GitHubNetworkError(GitHubAdapterError):
    """Raised on connection failure, timeout, or DNS resolution issues."""


class GitHubAPIResponseError(GitHubAdapterError):
    """Raised on 5xx server errors, invalid JSON, or schema parsing mismatches."""


# =============================================================================
# Typed GitHub API Data Models
# =============================================================================


class GitHubUser(BaseModel):
    """Typed model for a GitHub user / author account."""

    model_config = ConfigDict(extra="ignore")

    login: str = Field(..., description="GitHub username.")
    id: int = Field(..., description="Unique GitHub user identifier.")
    html_url: str = Field(default="", description="Web profile URL.")
    type: str = Field(default="User", description="Account type ('User', 'Bot', 'Organization').")
    site_admin: bool = Field(default=False, description="Whether the user is a site administrator.")


class GitHubLabel(BaseModel):
    """Typed model for an issue label."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = Field(default=None, description="Unique label identifier.")
    name: str = Field(..., description="Label text name.")
    color: str = Field(default="", description="Hex color code without '#'.")
    description: str | None = Field(default=None, description="Optional label description.")


class GitHubComment(BaseModel):
    """Typed model for an issue comment."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(..., description="Unique comment identifier.")
    body: str = Field(..., description="Comment markdown text body.")
    user: GitHubUser = Field(..., description="Author of the comment.")
    html_url: str = Field(default="", description="Web permalink for the comment.")
    created_at: datetime = Field(..., description="Timestamp when the comment was created.")
    updated_at: datetime = Field(..., description="Timestamp when the comment was last edited.")


class GitHubIssue(BaseModel):
    """Typed model for a GitHub issue."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(..., description="Unique issue identifier.")
    number: int = Field(..., description="Issue number within the repository.")
    title: str = Field(..., description="Issue title.")
    body: str | None = Field(default=None, description="Issue description body in markdown.")
    state: str = Field(..., description="State of the issue ('open' or 'closed').")
    user: GitHubUser = Field(..., description="Author of the issue.")
    labels: list[GitHubLabel] = Field(default_factory=list, description="Associated labels.")
    html_url: str = Field(default="", description="Web permalink for the issue.")
    comments_count: int = Field(default=0, alias="comments", description="Number of comments.")
    created_at: datetime = Field(..., description="Timestamp when the issue was opened.")
    updated_at: datetime = Field(..., description="Timestamp when the issue was last updated.")
    closed_at: datetime | None = Field(default=None, description="Timestamp when closed, if any.")
    pull_request: dict[str, Any] | None = Field(
        default=None, description="Present if this issue is a pull request."
    )


class GitHubBranch(BaseModel):
    """Typed model for a GitHub branch."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Branch name.")
    commit: dict[str, Any] = Field(default_factory=dict, description="Latest commit metadata.")

    @property
    def sha(self) -> str:
        """Convenience property to access commit SHA."""
        return str(self.commit.get("sha", ""))


class GitHubPullRequest(BaseModel):
    """Typed model for a GitHub Pull Request."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(..., description="Unique PR identifier.")
    number: int = Field(..., description="Pull request number.")
    title: str = Field(..., description="Pull request title.")
    body: str | None = Field(default=None, description="Pull request description.")
    state: str = Field(default="open", description="State ('open' or 'closed').")
    html_url: str = Field(default="", description="Web permalink for the PR.")
    head_branch: str = Field(default="", description="Source branch name.")
    base_branch: str = Field(default="main", description="Target base branch name.")


class GitHubRepository(BaseModel):
    """Typed model for a GitHub repository."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(..., description="Unique repository identifier.")
    name: str = Field(..., description="Repository name.")
    full_name: str = Field(..., description="Full repository path in 'owner/name' format.")
    owner: GitHubUser = Field(..., description="Repository owner account.")
    private: bool = Field(default=False, description="Whether the repository is private.")
    html_url: str = Field(default="", description="Web URL for the repository.")
    description: str | None = Field(default=None, description="Repository description.")
    default_branch: str = Field(default="main", description="Default branch name.")
    open_issues_count: int = Field(default=0, description="Total count of open issues.")
    permissions: dict[str, bool] | None = Field(
        default=None, description="Permissions for the authenticated user on this repository."
    )


_MAX_OWNER_LENGTH: int = 39
_MAX_REPO_LENGTH: int = 100
_RESERVED_NAMES: frozenset[str] = frozenset({".", ".."})

# GitHub usernames/orgs: alphanumeric with single hyphens (no consecutive or
# leading/trailing hyphens), 1-39 chars
_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*$")
# GitHub repository names: alphanumeric, hyphens, underscores, dots, 1-100 chars
_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def validate_repo_params(owner: str, repo: str) -> None:
    """
    Validate repository owner and name parameters against GitHub specifications.

    Rejects:
    - Empty, whitespace, or non-string values.
    - Exact '.' or '..' relative directory paths.
    - Path traversal sequences (../, ..\\).
    - URL-encoded characters (e.g. %2e, %2f, %00).
    - Control characters and whitespace.
    - Length exceeding GitHub limits (owner > 39 chars, repo > 100 chars).
    - Malformed names not conforming to permitted character sets.
    """
    if not isinstance(owner, str) or not isinstance(repo, str):
        raise GitHubValidationError("Repository owner and name must be strings")

    # Reject leading/trailing whitespace or empty strings
    if not owner or owner != owner.strip():
        raise GitHubValidationError(
            f"Invalid repository owner (empty or contains whitespace): {owner!r}"
        )
    if not repo or repo != repo.strip():
        raise GitHubValidationError(
            f"Invalid repository name (empty or contains whitespace): {repo!r}"
        )

    # Reject exact reserved directory references
    if owner in _RESERVED_NAMES:
        raise GitHubValidationError(
            f"Repository owner cannot be reserved directory reference: {owner!r}"
        )
    if repo in _RESERVED_NAMES:
        raise GitHubValidationError(
            f"Repository name cannot be reserved directory reference: {repo!r}"
        )

    # Reject URL encoding attempts and control characters
    if "%" in owner or unquote(owner) != owner or any(ord(c) < 32 or ord(c) == 127 for c in owner):
        raise GitHubValidationError(
            f"Repository owner contains invalid or encoded characters: {owner!r}"
        )
    if "%" in repo or unquote(repo) != repo or any(ord(c) < 32 or ord(c) == 127 for c in repo):
        raise GitHubValidationError(
            f"Repository name contains invalid or encoded characters: {repo!r}"
        )

    # Length constraints
    if len(owner) > _MAX_OWNER_LENGTH:
        raise GitHubValidationError(
            f"Repository owner exceeds maximum length of {_MAX_OWNER_LENGTH} "
            f"characters (got {len(owner)})"
        )
    if len(repo) > _MAX_REPO_LENGTH:
        raise GitHubValidationError(
            f"Repository name exceeds maximum length of {_MAX_REPO_LENGTH} "
            f"characters (got {len(repo)})"
        )

    # Character set matching
    if not _OWNER_PATTERN.match(owner):
        raise GitHubValidationError(f"Repository owner does not match permitted pattern: {owner!r}")
    if not _REPO_PATTERN.match(repo):
        raise GitHubValidationError(f"Repository name does not match permitted pattern: {repo!r}")


# =============================================================================
# GitHub Client Protocol
# =============================================================================


@runtime_checkable
class GitHubClientProtocol(Protocol):
    """
    Asynchronous protocol defining the contract for GitHub operations.
    Enables pluggable implementations including Personal Access Token (PAT) REST clients,
    GitHub App installation token clients, and deterministic in-memory mock clients.
    """

    async def start(self) -> None:
        """Initialize the client session and underlying resources."""
        ...

    async def close(self) -> None:
        """Release underlying resources and close the client session."""
        ...

    async def __aenter__(self) -> GitHubClientProtocol: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Fetch repository metadata by owner and name."""
        ...

    async def list_authorized_repositories(
        self,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubRepository]:
        """
        List repositories accessible to the authenticated credentials.

        Authentication Provider Compatibility:
        - Personal Access Token (PAT):
          Queries 'GET /user/repos', returning all repositories the token user has
          explicit read or write access to as a direct JSON list.
        - GitHub App Installation Tokens (Future GitHubAppClient):
          Must query 'GET /installation/repositories' (HTTP header:
          'Authorization: Bearer <installation_access_token>'), which returns an
          envelope: '{"total_count": int, "repositories": [...]}'. The client unwraps
          the 'repositories' list and returns 'list[GitHubRepository]'.
        Both implementations conform to this exact method signature and return type.
        """
        ...

    async def verify_repository_access(self, owner: str, repo: str) -> bool:
        """Verify whether credentials grant access to the specified repository."""
        ...

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """Fetch an individual issue by repository and issue number."""
        ...

    async def get_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubComment]:
        """Fetch a single page of issue comments."""
        ...

    async def get_all_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        per_page: int = 100,
        max_comments: int | None = None,
    ) -> list[GitHubComment]:
        """Fetch all comments for an issue, following pagination safely."""
        ...

    async def list_repository_issues(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubIssue]:
        """List issues for a repository, excluding pull requests."""
        ...

    async def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        """Fetch branch metadata."""
        ...

    async def create_branch(
        self,
        owner: str,
        repo: str,
        new_branch: str,
        from_sha: str,
    ) -> dict[str, Any]:
        """Create a new git reference / branch."""
        ...

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> GitHubPullRequest:
        """Create a new pull request."""
        ...

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
        """Fetch an existing pull request."""
        ...


# =============================================================================
# GitHub API Client
# =============================================================================


class GitHubClient:
    """
    Asynchronous client for interacting with the GitHub REST API v3.

    Handles connection pooling, request authentication, header formatting,
    rate-limit extraction, pagination, and error dispatching.
    """

    def __init__(
        self,
        token: str | SecretStr | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Initialize the GitHub API client.

        Parameters
        ----------
        token:
            GitHub Personal Access Token (PAT). If None, reads from
            get_settings().github_token.
        base_url:
            Root URL for GitHub REST API calls. If None, reads from
            get_settings().github_api_base_url.
        timeout:
            Request timeout in seconds. If None, reads from get_settings().github_timeout_seconds.
        http_client:
            Optional pre-configured httpx.AsyncClient (e.g. for testing with MockTransport).
            When provided, client lifecycle (start/close) is managed by the caller.
        """
        settings = get_settings()

        # Token resolution and secret protection
        if token is None:
            self._token: SecretStr = settings.github_token
        elif isinstance(token, str):
            self._token = SecretStr(token)
        else:
            self._token = token

        # Base URL resolution (strip trailing slashes)
        resolved_base = base_url if base_url is not None else settings.github_api_base_url
        self.base_url: str = resolved_base.rstrip("/")

        # Timeout resolution
        self.timeout: float = timeout if timeout is not None else settings.github_timeout_seconds

        # HTTP Client management
        self._external_client: bool = http_client is not None
        self._client: httpx.AsyncClient | None = http_client

    def _build_headers(self) -> dict[str, str]:
        """Construct standard GitHub REST API request headers."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "Agentic-Software-Development-Workflow/0.1.0",
        }
        token_val = self._token.get_secret_value().strip()
        if token_val:
            headers["Authorization"] = f"Bearer {token_val}"
        return headers

    async def start(self) -> None:
        """Initialize the underlying HTTP client if not externally injected."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._build_headers(),
            )
            logger.debug(
                "Initialized internal httpx.AsyncClient for GitHub API at %s", self.base_url
            )

    async def close(self) -> None:
        """Close the underlying HTTP client if internally owned."""
        if not self._external_client and self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed internal httpx.AsyncClient for GitHub API")

    async def __aenter__(self) -> GitHubClient:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"<GitHubClient base_url={self.base_url!r}>"

    async def _get_active_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client, lazily initializing if required."""
        if self._client is None:
            await self.start()
        assert self._client is not None
        return self._client

    # -------------------------------------------------------------------------
    # Transport & Error Handling
    # -------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_data: Any | None = None,
    ) -> httpx.Response:
        """
        Execute an HTTP request against the GitHub REST API with error handling.

        Raises
        ------
        GitHubAuthError
            On 401 Unauthorized.
        GitHubRateLimitError
            On 403 (with rate-limit headers) or 429 Too Many Requests.
        GitHubPermissionError
            On 403 Forbidden without rate limit headers.
        GitHubResourceNotFoundError
            On 404 Not Found.
        GitHubValidationError
            On 422 Unprocessable Content.
        GitHubAPIResponseError
            On 5xx Server Errors or unexpected status codes.
        GitHubNetworkError
            On connection errors or timeouts.
        """
        client = await self._get_active_client()
        clean_path = path if path.startswith("/") else f"/{path}"
        url = (
            path
            if path.startswith("http://") or path.startswith("https://")
            else f"{self.base_url}{clean_path}"
        )
        headers = self._build_headers()

        try:
            response = await client.request(
                method, url, params=params, json=json_data, headers=headers
            )
        except httpx.TimeoutException as exc:
            logger.warning("GitHub API request timed out: %s %s", method, clean_path)
            raise GitHubNetworkError(
                f"GitHub API request timed out after {self.timeout}s: {method} {clean_path}"
            ) from exc
        except httpx.NetworkError as exc:
            logger.warning("GitHub API network error: %s %s (%s)", method, clean_path, exc)
            raise GitHubNetworkError(
                f"GitHub API network connection failed for {method} {clean_path}"
            ) from exc

        if response.is_success:
            return response

        # Extract rate-limit headers
        remaining = response.headers.get("x-ratelimit-remaining")
        reset_header = response.headers.get("x-ratelimit-reset")
        retry_after_header = response.headers.get("retry-after")

        reset_at: datetime | None = None
        if reset_header and reset_header.isdigit():
            reset_at = datetime.fromtimestamp(int(reset_header), tz=UTC)

        retry_after: int | None = None
        if retry_after_header and retry_after_header.isdigit():
            retry_after = int(retry_after_header)

        # Parse error message from body if available
        detail = ""
        try:
            body_json = response.json()
            if isinstance(body_json, dict):
                detail = body_json.get("message", "")
        except Exception:
            detail = response.text[:200]

        # Dispatch typed exceptions
        if response.status_code == 401:
            raise GitHubAuthError(
                f"GitHub authentication failed (401): {detail or 'Bad credentials'}",
                status_code=401,
            )

        if response.status_code == 403:
            # Check if this 403 is specifically a rate-limit event
            is_rate_limit = (remaining == "0") or ("rate limit" in detail.lower())
            if is_rate_limit:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded: {detail}",
                    status_code=403,
                    reset_at=reset_at,
                    retry_after=retry_after,
                )
            raise GitHubPermissionError(
                f"GitHub access forbidden (403): {detail or 'Insufficient permissions'}",
                status_code=403,
            )

        if response.status_code == 404:
            raise GitHubResourceNotFoundError(
                f"GitHub resource not found (404) at {clean_path}: {detail or 'Not Found'}",
                status_code=404,
            )

        if response.status_code == 422:
            raise GitHubValidationError(
                f"GitHub validation error (422): {detail or 'Unprocessable Entity'}",
                status_code=422,
            )

        if response.status_code == 429:
            raise GitHubRateLimitError(
                f"GitHub rate limit exceeded (429): {detail or 'Too Many Requests'}",
                status_code=429,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            raise GitHubAPIResponseError(
                f"GitHub internal server error ({response.status_code}): {detail}",
                status_code=response.status_code,
            )

        raise GitHubAPIResponseError(
            f"Unexpected GitHub API response ({response.status_code}): {detail}",
            status_code=response.status_code,
        )

    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """
        Fetch repository metadata for the specified owner and repository name.

        Endpoint: GET /repos/{owner}/{repo}
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}"
        response = await self._request("GET", path)

        try:
            return GitHubRepository.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            logger.error("Failed to parse GitHub repository payload: %s", exc)
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema for repository '{owner}/{repo}'"
            ) from exc

    async def list_authorized_repositories(
        self,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubRepository]:
        """
        Enumerate repositories accessible to the configured authenticated credentials.
        Uses GET /user/repos with pagination.

        Parameters
        ----------
        per_page:
            Number of repositories per page (1 to 100).
        page:
            Page number (1-based).
        """
        page_size = min(max(per_page, 1), 100)
        page_num = max(page, 1)
        path = "/user/repos"
        params = {
            "per_page": page_size,
            "page": page_num,
            "sort": "updated",
            "direction": "desc",
        }
        response = await self._request("GET", path, params=params)

        try:
            raw_items = response.json()
            if not isinstance(raw_items, list):
                raise ValueError("Expected JSON array of repositories")
            return [GitHubRepository.model_validate(item) for item in raw_items]
        except (ValueError, ValidationError) as exc:
            logger.error("Failed to parse GitHub repositories payload: %s", exc)
            raise GitHubAPIResponseError(
                "Invalid JSON response schema for authorized repositories"
            ) from exc

    async def verify_repository_access(self, owner: str, repo: str) -> bool:
        """
        Verify whether the configured credentials can access the specified repository.

        Behavior:
        - Returns True if credentials grant access and repository exists.
        - Returns False if access is genuinely denied:
          - Repository not found (HTTP 404)
          - Insufficient permissions / Forbidden (HTTP 403 without rate limit)
        - Re-raises on non-permission operational and security failures so that
          the service boundary can distinguish temporary or configuration errors:
          - GitHubAuthError (HTTP 401: bad/expired token)
          - GitHubRateLimitError (HTTP 403 rate limit / 429)
          - GitHubNetworkError (connection error, timeout)
          - GitHubAPIResponseError (HTTP 5xx server errors, invalid schemas)
          - GitHubValidationError (malformed owner/repo parameters)

        Access is never granted on failure.
        """
        validate_repo_params(owner, repo)
        try:
            repo_data = await self.get_repository(owner, repo)
            return repo_data is not None
        except (GitHubResourceNotFoundError, GitHubPermissionError) as exc:
            logger.info("Repository access denied for '%s/%s': %s", owner, repo, exc)
            return False

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """
        Fetch a single repository issue by owner, repo, and issue number.

        Endpoint: GET /repos/{owner}/{repo}/issues/{issue_number}
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}/issues/{issue_number}"
        response = await self._request("GET", path)

        try:
            return GitHubIssue.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            logger.error("Failed to parse GitHub issue payload: %s", exc)
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema for issue '{owner}/{repo}#{issue_number}'"
            ) from exc

    async def get_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubComment]:
        """
        Fetch a single page of issue comments.

        Endpoint: GET /repos/{owner}/{repo}/issues/{issue_number}/comments
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}/issues/{issue_number}/comments"
        params = {"per_page": min(max(per_page, 1), 100), "page": max(page, 1)}
        response = await self._request("GET", path, params=params)

        try:
            raw_items = response.json()
            if not isinstance(raw_items, list):
                raise ValueError("Expected JSON array of comments")
            return [GitHubComment.model_validate(item) for item in raw_items]
        except (ValueError, ValidationError) as exc:
            logger.error("Failed to parse GitHub comments page payload: %s", exc)
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema for comments on '{owner}/{repo}#{issue_number}'"
            ) from exc

    async def get_all_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        per_page: int = 100,
        max_comments: int | None = None,
    ) -> list[GitHubComment]:
        """
        Fetch all comments for an issue, transparently following GitHub API pagination.

        Parameters
        ----------
        owner:
            Repository owner or organization name.
        repo:
            Repository name.
        issue_number:
            Issue number.
        per_page:
            Number of comments to request per page (max 100).
        max_comments:
            Optional ceiling on total comments to retrieve. None retrieves all.
        """
        validate_repo_params(owner, repo)
        if max_comments is not None and max_comments <= 0:
            return []

        all_comments: list[GitHubComment] = []
        page = 1
        page_size = min(max(per_page, 1), 100)
        max_pages = 1000  # Guard against infinite pagination loops

        while page <= max_pages:
            path = f"/repos/{owner}/{repo}/issues/{issue_number}/comments"
            params = {"per_page": page_size, "page": page}
            response = await self._request("GET", path, params=params)

            try:
                raw_items = response.json()
                if not isinstance(raw_items, list):
                    raise ValueError("Expected JSON array of comments")
                page_comments = [GitHubComment.model_validate(item) for item in raw_items]
            except (ValueError, ValidationError) as exc:
                raise GitHubAPIResponseError(
                    f"Invalid JSON response schema while paginating comments for "
                    f"'{owner}/{repo}#{issue_number}'"
                ) from exc

            if not page_comments:
                break

            all_comments.extend(page_comments)

            if max_comments is not None and len(all_comments) >= max_comments:
                return all_comments[:max_comments]

            # Check Link header for rel="next"
            link_header = response.headers.get("link", "")
            match = _LINK_REL_NEXT_PATTERN.search(link_header)
            if not match:
                # If no next link and returned items were less than page size,
                # we are on the final page
                if len(page_comments) < page_size:
                    break
                # Alternatively if no Link header but exactly page_size returned, advance page
                page += 1
            else:
                next_url = match.group(1)
                parsed_next = urlparse(next_url)
                query_params = parse_qs(parsed_next.query)
                if "page" in query_params and query_params["page"]:
                    page = int(query_params["page"][0])
                else:
                    page += 1

        return all_comments

    async def list_repository_issues(
        self,
        owner: str,
        repo: str,
        *,
        state: str = "open",
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubIssue]:
        """
        List issues for the specified repository with pagination.
        Filters out pull requests (which GitHub's issue endpoint includes by default).

        Parameters
        ----------
        owner:
            Repository owner or organization name.
        repo:
            Repository name.
        state:
            Issue state filter: 'open', 'closed', or 'all'.
        per_page:
            Results per page (1 to 100).
        page:
            Page number (1-based).
        """
        validate_repo_params(owner, repo)
        valid_states = {"open", "closed", "all"}
        clean_state = state.lower() if state.lower() in valid_states else "open"
        page_size = min(max(per_page, 1), 100)
        page_num = max(page, 1)

        path = f"/repos/{owner}/{repo}/issues"
        params = {
            "state": clean_state,
            "per_page": page_size,
            "page": page_num,
        }
        response = await self._request("GET", path, params=params)

        try:
            raw_items = response.json()
            if not isinstance(raw_items, list):
                raise ValueError("Expected JSON array of issues")

            parsed_issues: list[GitHubIssue] = []
            for item in raw_items:
                if isinstance(item, dict) and "pull_request" not in item:
                    parsed_issues.append(GitHubIssue.model_validate(item))
            return parsed_issues
        except (ValueError, ValidationError) as exc:
            logger.error("Failed to parse GitHub repository issues payload: %s", exc)
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema for issues on '{owner}/{repo}'"
            ) from exc

    async def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        """
        Fetch branch information for a repository.

        Endpoint: GET /repos/{owner}/{repo}/branches/{branch}
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}/branches/{branch}"
        response = await self._request("GET", path)
        try:
            return GitHubBranch.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema for branch '{branch}' in '{owner}/{repo}'"
            ) from exc

    async def create_branch(
        self, owner: str, repo: str, new_branch: str, from_sha: str
    ) -> dict[str, Any]:
        """
        Create a new git reference / branch in the repository.

        Endpoint: POST /repos/{owner}/{repo}/git/refs
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}/git/refs"
        payload = {"ref": f"refs/heads/{new_branch}", "sha": from_sha}
        response = await self._request("POST", path, json_data=payload)
        data = response.json()
        if not isinstance(data, dict):
            raise GitHubAPIResponseError(f"Unexpected response creating branch '{new_branch}'")
        return data

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> GitHubPullRequest:
        """
        Create a new Pull Request.

        Endpoint: POST /repos/{owner}/{repo}/pulls
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}/pulls"
        payload = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
        }
        response = await self._request("POST", path, json_data=payload)
        try:
            return GitHubPullRequest.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema creating PR in '{owner}/{repo}'"
            ) from exc

    async def get_pull_request(self, owner: str, repo: str, pull_number: int) -> GitHubPullRequest:
        """
        Fetch details of an existing Pull Request.

        Endpoint: GET /repos/{owner}/{repo}/pulls/{pull_number}
        """
        validate_repo_params(owner, repo)
        path = f"/repos/{owner}/{repo}/pulls/{pull_number}"
        response = await self._request("GET", path)
        try:
            return GitHubPullRequest.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise GitHubAPIResponseError(
                f"Invalid JSON response schema for PR #{pull_number} in '{owner}/{repo}'"
            ) from exc


# =============================================================================
# FastAPI Dependency Injection
# =============================================================================


async def get_github_client(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[GitHubClient, None]:
    """
    FastAPI dependency yielding an active GitHubClient session per request.

    Automatically closes the client session when request handling completes.
    """
    async with GitHubClient(
        token=settings.github_token,
        base_url=settings.github_api_base_url,
        timeout=settings.github_timeout_seconds,
    ) as client:
        yield client
