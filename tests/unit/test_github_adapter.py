"""
tests/unit/test_github_adapter.py
---------------------------------
Comprehensive unit tests for the asynchronous GitHub API adapter (app/adapters/github.py).

Test categories:
- TestGitHubClientLifecycle: Context manager lifecycle, external client injection.
- TestGetRepository: Repository retrieval, schema validation, 404 handling.
- TestGetIssue: Issue retrieval, labels, author, timestamps, 404 handling.
- TestGetComments: Single page comments, schema validation.
- TestPagination: Multi-page comment traversal via Link header and page boundaries.
- TestErrorHandling: 401 Auth, 403 Permission, 403/429 Rate Limit, 404, 422, 500, Timeout.
- TestSecretSafety: Verification that credentials are not leaked in exceptions, logs, or reprs.
- TestDependencyInjection: FastAPI get_github_client dependency lifecycle.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.adapters.github import (
    GitHubAPIResponseError,
    GitHubAuthError,
    GitHubClient,
    GitHubClientProtocol,
    GitHubComment,
    GitHubIssue,
    GitHubNetworkError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubRepository,
    GitHubResourceNotFoundError,
    GitHubValidationError,
    get_github_client,
    validate_repo_params,
)
from app.core.config import Settings
from tests.mocks.mock_github import MockGitHubClient

# =============================================================================
# Mock Payloads & Helpers
# =============================================================================

MOCK_REPO_PAYLOAD: dict[str, Any] = {
    "id": 1296269,
    "name": "Hello-World",
    "full_name": "octocat/Hello-World",
    "private": False,
    "owner": {
        "login": "octocat",
        "id": 1,
        "html_url": "https://github.com/octocat",
        "type": "User",
        "site_admin": False,
    },
    "html_url": "https://github.com/octocat/Hello-World",
    "description": "This your first repo!",
    "default_branch": "master",
    "open_issues_count": 2,
}

MOCK_ISSUE_PAYLOAD: dict[str, Any] = {
    "id": 1347,
    "number": 42,
    "title": "Found a bug in worker loop",
    "body": "Detailed description of the issue in markdown.",
    "state": "open",
    "user": {
        "login": "octocat",
        "id": 1,
        "html_url": "https://github.com/octocat",
        "type": "User",
        "site_admin": False,
    },
    "labels": [
        {
            "id": 208045946,
            "name": "bug",
            "color": "f29513",
            "description": "Something isn't working",
        },
        {
            "id": 208045947,
            "name": "backend",
            "color": "0052cc",
            "description": "Backend component",
        },
    ],
    "html_url": "https://github.com/octocat/Hello-World/issues/42",
    "comments": 3,
    "created_at": "2026-11-10T12:00:00Z",
    "updated_at": "2026-11-10T13:30:00Z",
    "closed_at": None,
}

MOCK_COMMENTS_PAGE_1: list[dict[str, Any]] = [
    {
        "id": 101,
        "body": "First comment from reviewer.",
        "user": {"login": "mona", "id": 2, "html_url": "https://github.com/mona"},
        "html_url": "https://github.com/octocat/Hello-World/issues/42#issuecomment-101",
        "created_at": "2026-11-10T12:15:00Z",
        "updated_at": "2026-11-10T12:15:00Z",
    },
    {
        "id": 102,
        "body": "Second comment with reproduction steps.",
        "user": {"login": "octocat", "id": 1, "html_url": "https://github.com/octocat"},
        "html_url": "https://github.com/octocat/Hello-World/issues/42#issuecomment-102",
        "created_at": "2026-11-10T12:20:00Z",
        "updated_at": "2026-11-10T12:20:00Z",
    },
]

MOCK_COMMENTS_PAGE_2: list[dict[str, Any]] = [
    {
        "id": 103,
        "body": "Third comment: patch proposed.",
        "user": {"login": "dev", "id": 3, "html_url": "https://github.com/dev"},
        "html_url": "https://github.com/octocat/Hello-World/issues/42#issuecomment-103",
        "created_at": "2026-11-10T13:00:00Z",
        "updated_at": "2026-11-10T13:00:00Z",
    }
]


# =============================================================================
# Test: GitHubClient Lifecycle & Headers
# =============================================================================


class TestGitHubClientLifecycle:
    """Verify HTTP client lifecycle, context manager entry/exit, and header security."""

    async def test_client_context_manager_lifecycle(self) -> None:
        """Client starts internal httpx.AsyncClient on enter and closes it on exit."""
        client = GitHubClient(
            token="ghp_test_token_123",
            base_url="https://api.github.com",
            timeout=10.0,
        )
        assert client._client is None

        async with client as active_client:
            assert active_client._client is not None
            assert not active_client._client.is_closed

        assert client._client is None

    async def test_client_external_http_client_not_closed(self) -> None:
        """When an external client is injected, GitHubClient does not close it upon exit."""
        external_client = httpx.AsyncClient()
        client = GitHubClient(token="ghp_test", http_client=external_client)

        async with client:
            assert client._client is external_client

        # External client should remain open
        assert not external_client.is_closed
        await external_client.aclose()

    async def test_client_headers_contain_version_and_auth(self) -> None:
        """Headers must include Bearer token, api version, user agent, and accept json."""
        client = GitHubClient(token="ghp_secret_token")
        headers = client._build_headers()

        assert headers["Accept"] == "application/vnd.github+json"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"
        assert headers["Authorization"] == "Bearer ghp_secret_token"
        assert "Agentic-Software-Development-Workflow" in headers["User-Agent"]

    async def test_client_headers_empty_token_omits_authorization(self) -> None:
        """When token is empty string, Authorization header is omitted."""
        client = GitHubClient(token="")
        headers = client._build_headers()
        assert "Authorization" not in headers

    async def test_base_url_strips_trailing_slash(self) -> None:
        """Trailing slashes are cleanly stripped from the configured base_url."""
        client = GitHubClient(token="token", base_url="https://github.enterprise.local/api/v3/")
        assert client.base_url == "https://github.enterprise.local/api/v3"

    async def test_client_lazy_start_without_context_manager(self) -> None:
        """Calling an API method without entering context manager lazily initializes client."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=MOCK_REPO_PAYLOAD)

        # Build mock transport and inject client
        mock_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = GitHubClient(token="ghp_test", http_client=mock_http)
        repo = await client.get_repository("octocat", "Hello-World")
        assert repo.name == "Hello-World"
        await mock_http.aclose()

    def test_client_default_settings_resolution(self) -> None:
        """Client without explicit parameters resolves from get_settings()."""
        client = GitHubClient()
        assert client.base_url == "https://api.github.com"
        assert client.timeout == 30.0
        assert isinstance(client._token, SecretStr)


# =============================================================================
# Test: Repository Retrieval
# =============================================================================


class TestGetRepository:
    """Verify GET /repos/{owner}/{repo}."""

    async def test_get_repository_success(self) -> None:
        """Successful repo retrieval maps JSON payload into GitHubRepository."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World"
            assert request.headers["Authorization"] == "Bearer ghp_test"
            return httpx.Response(200, json=MOCK_REPO_PAYLOAD)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            repo = await client.get_repository("octocat", "Hello-World")

        assert isinstance(repo, GitHubRepository)
        assert repo.name == "Hello-World"
        assert repo.full_name == "octocat/Hello-World"
        assert repo.owner.login == "octocat"
        assert repo.default_branch == "master"
        assert repo.open_issues_count == 2
        assert repo.private is False

    async def test_get_repository_404_raises_not_found(self) -> None:
        """404 Not Found response raises GitHubResourceNotFoundError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "Not Found"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubResourceNotFoundError) as exc_info:
                await client.get_repository("octocat", "nonexistent")

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value).lower()

    async def test_get_repository_schema_mismatch_raises_api_response_error(self) -> None:
        """Malformed repository payload raises GitHubAPIResponseError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"incomplete": "data"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubAPIResponseError, match="Invalid JSON response schema"):
                await client.get_repository("octocat", "Hello-World")


# =============================================================================
# Test: Issue Retrieval
# =============================================================================


class TestGetIssue:
    """Verify GET /repos/{owner}/{repo}/issues/{issue_number}."""

    async def test_get_issue_success(self) -> None:
        """Successful issue retrieval parses title, body, author, labels, and timestamps."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/issues/42"
            return httpx.Response(200, json=MOCK_ISSUE_PAYLOAD)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            issue = await client.get_issue("octocat", "Hello-World", 42)

        assert isinstance(issue, GitHubIssue)
        assert issue.number == 42
        assert issue.title == "Found a bug in worker loop"
        assert issue.body == "Detailed description of the issue in markdown."
        assert issue.state == "open"
        assert issue.user.login == "octocat"
        assert len(issue.labels) == 2
        assert issue.labels[0].name == "bug"
        assert issue.labels[1].name == "backend"
        assert issue.comments_count == 3
        assert issue.created_at == datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)

    async def test_get_issue_schema_mismatch_raises_api_response_error(self) -> None:
        """Invalid schema (e.g. missing required number field) raises GitHubAPIResponseError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"title": "Missing number and author"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubAPIResponseError, match="Invalid JSON response schema"):
                await client.get_issue("octocat", "Hello-World", 42)

    async def test_get_issue_404_raises_not_found(self) -> None:
        """404 on issue fetch raises GitHubResourceNotFoundError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "Not Found"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubResourceNotFoundError) as exc_info:
                await client.get_issue("octocat", "Hello-World", 9999)

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value).lower()


# =============================================================================
# Test: Issue Comments & Pagination
# =============================================================================


class TestGetComments:
    """Verify issue comments retrieval and pagination."""

    async def test_get_issue_comments_single_page(self) -> None:
        """Fetching a single page of comments returns list of GitHubComment objects."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/issues/42/comments"
            assert request.url.params["page"] == "1"
            assert request.url.params["per_page"] == "30"
            return httpx.Response(200, json=MOCK_COMMENTS_PAGE_1)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            comments = await client.get_issue_comments("octocat", "Hello-World", 42)

        assert len(comments) == 2
        assert isinstance(comments[0], GitHubComment)
        assert comments[0].id == 101
        assert comments[0].user.login == "mona"
        assert comments[1].id == 102
        assert comments[1].user.login == "octocat"

    async def test_get_issue_comments_non_list_raises_api_response_error(self) -> None:
        """Unexpected object instead of array raises GitHubAPIResponseError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "not a list"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubAPIResponseError, match="Invalid JSON response schema"):
                await client.get_issue_comments("octocat", "Hello-World", 42)

    async def test_get_all_issue_comments_pagination_no_link_header(self) -> None:
        """When Link header is absent, page increments until empty list or short page."""

        def handler(request: httpx.Request) -> httpx.Response:
            page = request.url.params.get("page", "1")
            per_page = int(request.url.params.get("per_page", "2"))
            if page == "1":
                # Returns exactly per_page items without Link header
                return httpx.Response(200, json=MOCK_COMMENTS_PAGE_1[:per_page])
            if page == "2":
                # Returns 1 item (< per_page), indicating last page
                return httpx.Response(200, json=MOCK_COMMENTS_PAGE_2)
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            all_comments = await client.get_all_issue_comments(
                "octocat", "Hello-World", 42, per_page=2
            )

        assert len(all_comments) == 3

    async def test_get_all_issue_comments_pagination_via_link_header(self) -> None:
        """Pagination iterates through multiple pages using the Link header."""

        def handler(request: httpx.Request) -> httpx.Response:
            page = request.url.params.get("page", "1")
            if page == "1":
                headers = {
                    "link": (
                        "<https://api.github.com/repos/octocat/Hello-World/issues/42/comments"
                        '?page=2&per_page=2>; rel="next"'
                    )
                }
                return httpx.Response(200, headers=headers, json=MOCK_COMMENTS_PAGE_1)
            if page == "2":
                return httpx.Response(200, json=MOCK_COMMENTS_PAGE_2)
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            all_comments = await client.get_all_issue_comments(
                "octocat", "Hello-World", 42, per_page=2
            )

        assert len(all_comments) == 3
        assert [c.id for c in all_comments] == [101, 102, 103]

    async def test_get_all_issue_comments_respects_max_comments(self) -> None:
        """max_comments caps the total retrieved comments even if more pages exist."""

        def handler(request: httpx.Request) -> httpx.Response:
            headers = {
                "link": (
                    "<https://api.github.com/repos/octocat/Hello-World/issues/42/comments"
                    '?page=2>; rel="next"'
                )
            }
            return httpx.Response(200, headers=headers, json=MOCK_COMMENTS_PAGE_1)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            comments = await client.get_all_issue_comments(
                "octocat", "Hello-World", 42, max_comments=1
            )

        assert len(comments) == 1
        assert comments[0].id == 101

    async def test_get_all_issue_comments_zero_max_comments_returns_empty(self) -> None:
        """max_comments <= 0 returns empty list without issuing HTTP requests."""
        client = GitHubClient(token="ghp_test")
        comments = await client.get_all_issue_comments("octocat", "Hello-World", 42, max_comments=0)
        assert comments == []

        negative_comments = await client.get_all_issue_comments(
            "octocat", "Hello-World", 42, max_comments=-5
        )
        assert negative_comments == []


# =============================================================================
# Test: Error Handling & Status Codes
# =============================================================================


class TestErrorHandling:
    """Verify mapping of HTTP errors and network conditions to typed exceptions."""

    async def test_auth_failure_401_raises_github_auth_error(self) -> None:
        """HTTP 401 raises GitHubAuthError with status_code=401."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Bad credentials"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="invalid_pat", http_client=mock_http)
            with pytest.raises(GitHubAuthError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        assert exc_info.value.status_code == 401
        assert "authentication failed" in str(exc_info.value).lower()

    async def test_permission_failure_403_raises_github_permission_error(self) -> None:
        """HTTP 403 (when not rate limited) raises GitHubPermissionError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "50"},
                json={"message": "Resource not accessible by personal access token"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubPermissionError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        assert exc_info.value.status_code == 403
        assert "forbidden" in str(exc_info.value).lower()

    async def test_rate_limit_403_with_remaining_zero_raises_rate_limit_error(self) -> None:
        """HTTP 403 with x-ratelimit-remaining: 0 raises GitHubRateLimitError with reset_at."""
        reset_epoch = 1731672000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(reset_epoch),
                },
                json={"message": "API rate limit exceeded for user"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubRateLimitError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        assert exc_info.value.status_code == 403
        assert exc_info.value.reset_at == datetime.fromtimestamp(reset_epoch, tz=UTC)
        assert "rate limit exceeded" in str(exc_info.value).lower()

    async def test_rate_limit_429_with_retry_after(self) -> None:
        """HTTP 429 raises GitHubRateLimitError with parsed retry_after."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"retry-after": "60"},
                json={"message": "Too many requests. Please wait."},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubRateLimitError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 60

    async def test_validation_failure_422_raises_validation_error(self) -> None:
        """HTTP 422 raises GitHubValidationError with status_code=422."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"message": "Validation Failed", "errors": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubValidationError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        assert exc_info.value.status_code == 422

    async def test_server_error_500_raises_api_response_error(self) -> None:
        """HTTP 500 raises GitHubAPIResponseError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubAPIResponseError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        assert exc_info.value.status_code == 500

    async def test_timeout_raises_network_error(self) -> None:
        """httpx.TimeoutException maps to GitHubNetworkError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timeout occurred", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http, timeout=5.0)
            with pytest.raises(GitHubNetworkError, match="timed out"):
                await client.get_issue("owner", "repo", 1)

    async def test_connect_error_raises_network_error(self) -> None:
        """httpx.NetworkError maps to GitHubNetworkError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused by peer", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubNetworkError, match="network connection failed"):
                await client.get_issue("owner", "repo", 1)


# =============================================================================
# Test: Secret Safety
# =============================================================================


class TestSecretSafety:
    """Verify that credentials are never exposed through logs, reprs, or exceptions."""

    def test_token_stored_as_secret_str(self) -> None:
        """Token attribute on client is SecretStr."""
        client = GitHubClient(token="super_secret_pat_999")
        assert isinstance(client._token, SecretStr)
        assert client._token.get_secret_value() == "super_secret_pat_999"
        assert "super_secret_pat_999" not in str(client._token)
        assert "super_secret_pat_999" not in repr(client._token)

    async def test_token_not_leaked_in_exception_messages(self) -> None:
        """Exceptions raised by the adapter do not contain the raw token string."""
        secret = "super_secret_pat_888"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Bad credentials"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token=secret, http_client=mock_http)
            with pytest.raises(GitHubAuthError) as exc_info:
                await client.get_issue("owner", "repo", 1)

        err_msg = str(exc_info.value)
        assert secret not in err_msg


# =============================================================================
# Test: FastAPI Dependency Injection
# =============================================================================


class TestDependencyInjection:
    """Verify get_github_client dependency factory."""

    async def test_get_github_client_lifecycle(self) -> None:
        """get_github_client yields an initialized client and cleans it up upon exit."""
        settings = Settings(
            github_token=SecretStr("ghp_di_token_123"),
            llm_api_key=SecretStr("sk_test"),
            github_api_base_url="https://api.github.com",
            github_timeout_seconds=25.0,
        )

        gen = get_github_client(settings=settings)
        client = await anext(gen)

        assert isinstance(client, GitHubClient)
        assert client._token.get_secret_value() == "ghp_di_token_123"
        assert client.base_url == "https://api.github.com"
        assert client.timeout == 25.0

        # Complete generator to test teardown
        with pytest.raises(StopAsyncIteration):
            await anext(gen)


# =============================================================================
# Test: Branches & Pull Requests
# =============================================================================


class TestBranchesAndPRs:
    """Verify branch and pull request endpoints."""

    async def test_get_branch_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/branches/main"
            return httpx.Response(
                200,
                json={
                    "name": "main",
                    "commit": {"sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e"},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            branch = await client.get_branch("octocat", "Hello-World", "main")

        assert branch.name == "main"
        assert branch.sha == "6dcb09b5b57875f334f61aebed695e2e4193db5e"

    async def test_create_branch_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/git/refs"
            return httpx.Response(
                201, json={"ref": "refs/heads/feature-1", "object": {"sha": "abc"}}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            res = await client.create_branch("octocat", "Hello-World", "feature-1", "abc")

        assert res["ref"] == "refs/heads/feature-1"

    async def test_create_and_get_pull_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                assert request.url.path == "/repos/octocat/Hello-World/pulls"
                return httpx.Response(
                    201,
                    json={
                        "id": 1001,
                        "number": 99,
                        "title": "Fix bug",
                        "body": "Fixes #42",
                        "state": "open",
                        "html_url": "https://github.com/octocat/Hello-World/pull/99",
                        "head_branch": "agent/issue-42",
                        "base_branch": "main",
                    },
                )
            if request.method == "GET":
                assert request.url.path == "/repos/octocat/Hello-World/pulls/99"
                return httpx.Response(
                    200,
                    json={
                        "id": 1001,
                        "number": 99,
                        "title": "Fix bug",
                        "body": "Fixes #42",
                        "state": "open",
                        "html_url": "https://github.com/octocat/Hello-World/pull/99",
                        "head_branch": "agent/issue-42",
                        "base_branch": "main",
                    },
                )
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            pr = await client.create_pull_request(
                "octocat", "Hello-World", "Fix bug", "Fixes #42", "agent/issue-42", "main"
            )
            assert pr.number == 99
            assert pr.state == "open"

            pr_fetched = await client.get_pull_request("octocat", "Hello-World", 99)
            assert pr_fetched.id == 1001


# =============================================================================
# Milestone 1: Multi-Repository, Protocol & Discovery Tests
# =============================================================================


class TestGitHubClientProtocol:
    """Verify runtime protocol conformance, static typing, and exact signature parity."""

    def test_github_client_implements_protocol(self) -> None:
        client = GitHubClient(token="ghp_test")
        assert isinstance(client, GitHubClientProtocol)

    def test_mock_github_client_implements_protocol(self) -> None:
        mock = MockGitHubClient()
        assert isinstance(mock, GitHubClientProtocol)

    def test_static_protocol_conformance(self) -> None:
        """Verify static type compatibility with GitHubClientProtocol."""
        client: GitHubClientProtocol = GitHubClient(token="ghp_test")
        mock: GitHubClientProtocol = MockGitHubClient()
        assert client is not None
        assert mock is not None

    def test_protocol_signature_parity_client_and_mock(self) -> None:
        """
        Verify that every method declared in GitHubClientProtocol exists on both
        GitHubClient and MockGitHubClient with identical parameter names, kinds,
        defaults, and coroutine qualifiers.
        """
        protocol_members = {
            name: getattr(GitHubClientProtocol, name)
            for name in dir(GitHubClientProtocol)
            if not name.startswith("_") or name in ("__aenter__", "__aexit__")
        }

        # Expected 15 protocol methods:
        # __aenter__, __aexit__, start, close,
        # get_repository, list_authorized_repositories, verify_repository_access,
        # get_issue, get_issue_comments, get_all_issue_comments, list_repository_issues,
        # get_branch, create_branch, create_pull_request, get_pull_request
        assert len(protocol_members) == 15, (
            f"Unexpected protocol member count: {len(protocol_members)}"
        )

        for cls in (GitHubClient, MockGitHubClient):
            for name, proto_member in protocol_members.items():
                assert hasattr(cls, name), f"{cls.__name__} is missing protocol method {name!r}"
                cls_member = getattr(cls, name)

                # Check coroutine parity
                assert inspect.iscoroutinefunction(proto_member) == inspect.iscoroutinefunction(
                    cls_member
                ), f"{cls.__name__}.{name} coroutine qualifier mismatch"

                proto_sig = inspect.signature(proto_member)
                cls_sig = inspect.signature(cls_member)

                proto_params = list(proto_sig.parameters.items())
                cls_params = list(cls_sig.parameters.items())

                assert len(proto_params) == len(cls_params), (
                    f"{cls.__name__}.{name} param count: "
                    f"expected {len(proto_params)}, got {len(cls_params)}"
                )

                for (p_name, p_param), (c_name, c_param) in zip(
                    proto_params, cls_params, strict=True
                ):
                    assert p_name == c_name, (
                        f"{cls.__name__}.{name} parameter name mismatch: "
                        f"expected {p_name}, got {c_name}"
                    )
                    assert p_param.kind == c_param.kind, (
                        f"{cls.__name__}.{name} parameter {p_name} kind mismatch"
                    )
                    assert p_param.default == c_param.default, (
                        f"{cls.__name__}.{name} parameter {p_name} default mismatch"
                    )


class TestRepositoryDiscovery:
    """Verify listing authorized repositories."""

    async def test_list_authorized_repositories_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/user/repos"
            assert request.url.params["per_page"] == "30"
            assert request.url.params["page"] == "1"
            return httpx.Response(
                200,
                json=[
                    MOCK_REPO_PAYLOAD,
                    {
                        "id": 8888,
                        "name": "Another-Repo",
                        "full_name": "octocat/Another-Repo",
                        "private": True,
                        "owner": {
                            "login": "octocat",
                            "id": 1,
                            "html_url": "https://github.com/octocat",
                        },
                        "html_url": "https://github.com/octocat/Another-Repo",
                        "description": "Private repository",
                        "default_branch": "main",
                        "open_issues_count": 0,
                        "permissions": {"admin": True, "push": True, "pull": True},
                    },
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            repos = await client.list_authorized_repositories()

        assert len(repos) == 2
        assert repos[0].name == "Hello-World"
        assert repos[1].name == "Another-Repo"
        assert repos[1].private is True
        assert repos[1].permissions == {"admin": True, "push": True, "pull": True}

    async def test_list_authorized_repositories_pagination(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/user/repos"
            assert request.url.params["per_page"] == "10"
            assert request.url.params["page"] == "2"
            return httpx.Response(200, json=[MOCK_REPO_PAYLOAD])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            repos = await client.list_authorized_repositories(per_page=10, page=2)

        assert len(repos) == 1
        assert repos[0].id == 1296269


class TestRepositoryAccessVerification:
    """Verify repository access checking and fail-closed behaviors."""

    async def test_verify_repository_access_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World"
            return httpx.Response(200, json=MOCK_REPO_PAYLOAD)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            has_access = await client.verify_repository_access("octocat", "Hello-World")

        assert has_access is True

    async def test_verify_repository_access_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "Not Found"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            has_access = await client.verify_repository_access("octocat", "Nonexistent-Repo")

        assert has_access is False

    async def test_verify_repository_access_forbidden(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"message": "Must have push access"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            has_access = await client.verify_repository_access("octocat", "Secret-Repo")

        assert has_access is False

    async def test_verify_repository_access_unauthorized_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Bad credentials"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_invalid", http_client=mock_http)
            with pytest.raises(GitHubAuthError):
                await client.verify_repository_access("octocat", "Hello-World")

    async def test_verify_repository_access_rate_limited_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"message": "Too Many Requests"},
                headers={"retry-after": "60"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubRateLimitError):
                await client.verify_repository_access("octocat", "Hello-World")

    async def test_verify_repository_access_network_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Failed to resolve host api.github.com")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubNetworkError):
                await client.verify_repository_access("octocat", "Hello-World")

    async def test_verify_repository_access_server_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            with pytest.raises(GitHubAPIResponseError):
                await client.verify_repository_access("octocat", "Hello-World")

    async def test_verify_repository_access_invalid_params_raises(self) -> None:
        client = GitHubClient(token="ghp_test")
        with pytest.raises(GitHubValidationError):
            await client.verify_repository_access("../etc", "passwd")


class TestRepositoryIssueDiscovery:
    """Verify issue enumeration and PR filtering."""

    async def test_list_repository_issues_filters_out_pull_requests(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/issues"
            assert request.url.params["state"] == "open"
            return httpx.Response(
                200,
                json=[
                    MOCK_ISSUE_PAYLOAD,  # Normal issue (pull_request is None)
                    {
                        "id": 9999,
                        "number": 50,
                        "title": "PR that GitHub includes in issues",
                        "state": "open",
                        "user": {"login": "contributor", "id": 10},
                        "html_url": "https://github.com/octocat/Hello-World/pull/50",
                        "pull_request": {
                            "url": "https://api.github.com/repos/octocat/Hello-World/pulls/50"
                        },
                        "created_at": "2026-11-10T14:00:00Z",
                        "updated_at": "2026-11-10T14:00:00Z",
                    },
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            issues = await client.list_repository_issues("octocat", "Hello-World")

        assert len(issues) == 1
        assert issues[0].number == 42
        assert issues[0].title == "Found a bug in worker loop"

    async def test_list_repository_issues_state_filtering_and_pagination(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/issues"
            assert request.url.params["state"] == "closed"
            assert request.url.params["per_page"] == "15"
            assert request.url.params["page"] == "3"
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            issues = await client.list_repository_issues(
                "octocat", "Hello-World", state="closed", per_page=15, page=3
            )

        assert issues == []


class TestParameterValidation:
    """Verify validation of owner and repository parameters."""

    def test_valid_names_pass(self) -> None:
        validate_repo_params("octocat", "Hello-World")
        validate_repo_params("my-org-123", "service.api_core")
        validate_repo_params("valid-user", "repo-with-dashes-and_underscores.v1")

    @pytest.mark.parametrize(
        ("owner", "repo"),
        [
            ("", "valid-repo"),
            ("valid-owner", ""),
            ("  ", "valid-repo"),
            ("valid-owner", "  "),
            (".", "valid-repo"),
            ("valid-owner", "."),
            ("..", "valid-repo"),
            ("valid-owner", ".."),
            ("../evil", "valid-repo"),
            ("valid-owner", "../../etc/passwd"),
            ("%2e%2e", "valid-repo"),
            ("valid-owner", "%2f"),
            ("valid-owner", "%00"),
            ("a" * 40, "valid-repo"),  # Exceeds max 39 chars
            ("valid-owner", "b" * 101),  # Exceeds max 100 chars
            ("owner with spaces", "valid-repo"),
            ("valid-owner", "repo with spaces"),
            ("owner\nnewline", "valid-repo"),
            ("valid-owner", "repo;injection"),
            ("octo--cat", "valid-repo"),  # Consecutive hyphens in owner
            ("octocat-", "valid-repo"),  # Trailing hyphen in owner
            ("-octocat", "valid-repo"),  # Leading hyphen in owner
            ("owner_with_underscore", "valid-repo"),  # Underscore invalid in GitHub username
            ("owner.with.dot", "valid-repo"),  # Dot invalid in GitHub username
        ],
    )
    def test_invalid_names_raise_validation_error(self, owner: str, repo: str) -> None:
        with pytest.raises(GitHubValidationError):
            validate_repo_params(owner, repo)


class TestMockGitHubClient:
    """Verify in-memory MockGitHubClient behavior and error injection."""

    async def test_mock_multi_repository_and_issues(self) -> None:
        mock = MockGitHubClient()
        mock.add_repository("octocat", "Repo-A", private=False)
        mock.add_repository("octocat", "Repo-B", private=True)

        mock.add_issue("octocat", "Repo-A", "Issue 1 in A", body="Body 1")
        mock.add_issue("octocat", "Repo-A", "Issue 2 in A", is_pull_request=True)
        mock.add_issue("octocat", "Repo-B", "Issue 1 in B")

        # Verify access
        assert await mock.verify_repository_access("octocat", "Repo-A") is True
        assert await mock.verify_repository_access("octocat", "Repo-B") is True
        assert await mock.verify_repository_access("octocat", "Nonexistent") is False

        # List authorized repositories
        repos = await mock.list_authorized_repositories()
        assert len(repos) == 2

        # List repository issues (excludes PRs)
        issues_a = await mock.list_repository_issues("octocat", "Repo-A")
        assert len(issues_a) == 1
        assert issues_a[0].title == "Issue 1 in A"

    async def test_mock_error_simulations(self) -> None:
        mock = MockGitHubClient()
        mock.add_repository("octocat", "Repo-A")

        # Simulate auth failure raises
        mock.simulate_auth_error = True
        with pytest.raises(GitHubAuthError):
            await mock.get_repository("octocat", "Repo-A")
        with pytest.raises(GitHubAuthError):
            await mock.verify_repository_access("octocat", "Repo-A")
        mock.simulate_auth_error = False

        # Simulate rate limit raises
        mock.simulate_rate_limit = True
        with pytest.raises(GitHubRateLimitError):
            await mock.get_repository("octocat", "Repo-A")
        with pytest.raises(GitHubRateLimitError):
            await mock.verify_repository_access("octocat", "Repo-A")
        mock.simulate_rate_limit = False

        # Simulate network error raises
        mock.simulate_network_error = True
        with pytest.raises(GitHubNetworkError):
            await mock.get_repository("octocat", "Repo-A")
        with pytest.raises(GitHubNetworkError):
            await mock.verify_repository_access("octocat", "Repo-A")
        mock.simulate_network_error = False

        # Forbidden repo returns False (genuine permission failure)
        mock.forbidden_repositories.add("octocat/repo-a")
        with pytest.raises(GitHubPermissionError):
            await mock.get_repository("octocat", "Repo-A")
        assert await mock.verify_repository_access("octocat", "Repo-A") is False

    async def test_mock_branches_and_pull_requests(self) -> None:
        mock = MockGitHubClient()
        mock.add_repository("octocat", "Repo-A", default_branch="main")

        # Default branch exists
        branch = await mock.get_branch("octocat", "Repo-A", "main")
        assert branch.name == "main"

        # Create branch
        await mock.create_branch("octocat", "Repo-A", "feature-x", "abc123sha")
        feat_branch = await mock.get_branch("octocat", "Repo-A", "feature-x")
        assert feat_branch.sha == "abc123sha"

        # Create PR
        pr = await mock.create_pull_request(
            "octocat", "Repo-A", "Title", "Body", "feature-x", "main"
        )
        assert pr.number == 1

        # Fetch PR
        fetched_pr = await mock.get_pull_request("octocat", "Repo-A", 1)
        assert fetched_pr.number == 1
        assert fetched_pr.head_branch == "feature-x"

    async def test_mock_get_all_issue_comments_limits(self) -> None:
        mock = MockGitHubClient()
        mock.add_repository("octocat", "Repo-A")
        mock.add_issue("octocat", "Repo-A", "Issue 1", issue_number=1)
        mock.add_comment("octocat", "Repo-A", 1, "Comment 1")
        mock.add_comment("octocat", "Repo-A", 1, "Comment 2")
        mock.add_comment("octocat", "Repo-A", 1, "Comment 3")

        # Zero or negative max_comments returns empty list
        assert await mock.get_all_issue_comments("octocat", "Repo-A", 1, max_comments=0) == []
        assert await mock.get_all_issue_comments("octocat", "Repo-A", 1, max_comments=-1) == []

        # Capped max_comments
        comments = await mock.get_all_issue_comments("octocat", "Repo-A", 1, max_comments=2)
        assert len(comments) == 2

        # All comments
        all_c = await mock.get_all_issue_comments("octocat", "Repo-A", 1)
        assert len(all_c) == 3
