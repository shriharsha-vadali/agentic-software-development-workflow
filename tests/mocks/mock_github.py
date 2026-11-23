"""
tests/mocks/mock_github.py
--------------------------
Deterministic in-memory mock GitHub API client implementing GitHubClientProtocol.
Provides controllable fixtures for repositories, issues, comments, branches, and PRs
without requiring live network connections or GitHub credentials.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.adapters.github import (
    GitHubAPIResponseError,
    GitHubAuthError,
    GitHubBranch,
    GitHubClientProtocol,
    GitHubComment,
    GitHubIssue,
    GitHubLabel,
    GitHubNetworkError,
    GitHubPermissionError,
    GitHubPullRequest,
    GitHubRateLimitError,
    GitHubRepository,
    GitHubResourceNotFoundError,
    GitHubUser,
    validate_repo_params,
)

logger = logging.getLogger(__name__)


class MockGitHubClient(GitHubClientProtocol):
    """
    Deterministic in-memory implementation of GitHubClientProtocol.
    Allows unit and integration tests to configure repositories, issues,
    branches, pull requests, and error conditions without external I/O.
    """

    def __init__(self) -> None:
        self._repositories: dict[str, GitHubRepository] = {}
        self._issues: dict[str, list[GitHubIssue]] = {}
        self._comments: dict[str, list[GitHubComment]] = {}
        self._branches: dict[str, dict[str, GitHubBranch]] = {}
        self._pull_requests: dict[str, list[GitHubPullRequest]] = {}
        self._next_pr_number: int = 1

        # Lifecycle tracking
        self.is_started: bool = False
        self.is_closed: bool = False

        # Configurable error simulation flags
        self.simulate_auth_error: bool = False
        self.simulate_rate_limit: bool = False
        self.simulate_network_error: bool = False
        self.simulate_server_error: bool = False
        self.forbidden_repositories: set[str] = set()

    def _repo_key(self, owner: str, repo: str) -> str:
        return f"{owner.strip().lower()}/{repo.strip().lower()}"

    def _issue_key(self, owner: str, repo: str, issue_number: int) -> str:
        return f"{self._repo_key(owner, repo)}#{issue_number}"

    def _check_simulated_errors(self, owner: str | None = None, repo: str | None = None) -> None:
        """Check and raise configured simulation errors."""
        if self.simulate_auth_error:
            raise GitHubAuthError("Simulated 401 Unauthorized: Bad credentials", status_code=401)
        if self.simulate_rate_limit:
            raise GitHubRateLimitError("Simulated 429 Too Many Requests", status_code=429)
        if self.simulate_network_error:
            raise GitHubNetworkError("Simulated connection timeout / network error")
        if self.simulate_server_error:
            raise GitHubAPIResponseError("Simulated 500 Internal Server Error", status_code=500)
        if owner and repo:
            key = self._repo_key(owner, repo)
            if key in self.forbidden_repositories:
                raise GitHubPermissionError(
                    f"Simulated 403 Forbidden: Insufficient permissions for '{owner}/{repo}'",
                    status_code=403,
                )

    # -------------------------------------------------------------------------
    # Test Setup / Fixture Population Helpers
    # -------------------------------------------------------------------------

    def add_repository(
        self,
        owner: str,
        repo: str,
        *,
        private: bool = False,
        default_branch: str = "main",
        description: str | None = None,
        permissions: dict[str, bool] | None = None,
        open_issues_count: int = 0,
    ) -> GitHubRepository:
        """Register a mock repository."""
        validate_repo_params(owner, repo)
        key = self._repo_key(owner, repo)
        repo_obj = GitHubRepository(
            id=len(self._repositories) + 1000,
            name=repo,
            full_name=f"{owner}/{repo}",
            owner=GitHubUser(login=owner, id=len(self._repositories) + 1),
            private=private,
            html_url=f"https://github.com/{owner}/{repo}",
            description=description,
            default_branch=default_branch,
            open_issues_count=open_issues_count,
            permissions=permissions or {"admin": True, "push": True, "pull": True},
        )
        self._repositories[key] = repo_obj
        if key not in self._issues:
            self._issues[key] = []
        if key not in self._branches:
            self._branches[key] = {}
        if key not in self._pull_requests:
            self._pull_requests[key] = []

        # Ensure default branch exists
        self.add_branch(owner, repo, default_branch)
        return repo_obj

    def add_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        *,
        issue_number: int | None = None,
        state: str = "open",
        author: str = "octocat",
        labels: list[str] | None = None,
        is_pull_request: bool = False,
    ) -> GitHubIssue:
        """Register a mock issue (or PR representation) in the repository."""
        validate_repo_params(owner, repo)
        key = self._repo_key(owner, repo)
        if key not in self._repositories:
            self.add_repository(owner, repo)

        existing = self._issues.setdefault(key, [])
        num = issue_number if issue_number is not None else len(existing) + 1

        label_objs = [
            GitHubLabel(id=i + 1, name=lbl, color="ededed") for i, lbl in enumerate(labels or [])
        ]
        issue_obj = GitHubIssue(
            id=len(existing) + 10000,
            number=num,
            title=title,
            body=body,
            state=state,
            user=GitHubUser(login=author, id=555),
            labels=label_objs,
            html_url=f"https://github.com/{owner}/{repo}/issues/{num}",
            comments_count=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            pull_request={"url": f"https://api.github.com/repos/{owner}/{repo}/pulls/{num}"}
            if is_pull_request
            else None,
        )
        existing.append(issue_obj)
        self._repositories[key].open_issues_count = len(
            [i for i in existing if i.state == "open" and i.pull_request is None]
        )
        return issue_obj

    def add_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str,
        *,
        author: str = "octocat",
    ) -> GitHubComment:
        """Add a comment to an existing issue."""
        validate_repo_params(owner, repo)
        issue_key = self._issue_key(owner, repo, issue_number)
        comments = self._comments.setdefault(issue_key, [])
        comment_id = len(comments) + 20000
        comment_url = (
            f"https://github.com/{owner}/{repo}/issues/{issue_number}"
            f"#issuecomment-{len(comments) + 1}"
        )
        comment_obj = GitHubComment(
            id=comment_id,
            body=body,
            user=GitHubUser(login=author, id=555),
            html_url=comment_url,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        comments.append(comment_obj)

        # Update comments_count on issue
        repo_key = self._repo_key(owner, repo)
        for iss in self._issues.get(repo_key, []):
            if iss.number == issue_number:
                iss.comments_count = len(comments)
                break

        return comment_obj

    def add_branch(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        sha: str = "0123456789abcdef0123456789abcdef01234567",
    ) -> GitHubBranch:
        """Register a branch with commit SHA in the repository."""
        validate_repo_params(owner, repo)
        key = self._repo_key(owner, repo)
        if key not in self._branches:
            self._branches[key] = {}
        branch_obj = GitHubBranch(name=branch_name, commit={"sha": sha})
        self._branches[key][branch_name] = branch_obj
        return branch_obj

    # -------------------------------------------------------------------------
    # GitHubClientProtocol Implementation
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        self.is_started = True

    async def close(self) -> None:
        self.is_closed = True

    async def __aenter__(self) -> MockGitHubClient:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        key = self._repo_key(owner, repo)
        if key not in self._repositories:
            raise GitHubResourceNotFoundError(
                f"GitHub resource not found (404) at /repos/{owner}/{repo}",
                status_code=404,
            )
        return self._repositories[key]

    async def list_authorized_repositories(
        self,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubRepository]:
        self._check_simulated_errors()
        page_size = min(max(per_page, 1), 100)
        page_num = max(page, 1)

        all_repos = list(self._repositories.values())
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        return all_repos[start_idx:end_idx]

    async def verify_repository_access(self, owner: str, repo: str) -> bool:
        """
        Verify whether mock credentials grant access to the specified repository.
        Follows identical error distinction contract as GitHubClient:
        - 404 / 403 Forbidden returns False.
        - Network, Rate Limit, Auth, Server, and Validation errors are re-raised.
        """
        validate_repo_params(owner, repo)
        try:
            self._check_simulated_errors(owner, repo)
        except (GitHubResourceNotFoundError, GitHubPermissionError):
            return False
        key = self._repo_key(owner, repo)
        return key in self._repositories

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        key = self._repo_key(owner, repo)
        for iss in self._issues.get(key, []):
            if iss.number == issue_number:
                return iss
        raise GitHubResourceNotFoundError(
            f"GitHub issue #{issue_number} not found in {owner}/{repo}",
            status_code=404,
        )

    async def get_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubComment]:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        # Ensure issue exists
        await self.get_issue(owner, repo, issue_number)

        issue_key = self._issue_key(owner, repo, issue_number)
        all_comments = self._comments.get(issue_key, [])
        page_size = min(max(per_page, 1), 100)
        page_num = max(page, 1)

        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        return all_comments[start_idx:end_idx]

    async def get_all_issue_comments(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        per_page: int = 100,
        max_comments: int | None = None,
    ) -> list[GitHubComment]:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        await self.get_issue(owner, repo, issue_number)

        if max_comments is not None and max_comments <= 0:
            return []

        issue_key = self._issue_key(owner, repo, issue_number)
        all_comments = list(self._comments.get(issue_key, []))
        if max_comments is not None:
            return all_comments[:max_comments]
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
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        # Ensure repository exists
        await self.get_repository(owner, repo)

        key = self._repo_key(owner, repo)
        issues = self._issues.get(key, [])

        # Filter out pull requests and apply state filter
        clean_state = state.lower()
        filtered = [
            i
            for i in issues
            if i.pull_request is None and (clean_state == "all" or i.state.lower() == clean_state)
        ]

        page_size = min(max(per_page, 1), 100)
        page_num = max(page, 1)
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        return filtered[start_idx:end_idx]

    async def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        key = self._repo_key(owner, repo)
        branches = self._branches.get(key, {})
        if branch not in branches:
            raise GitHubResourceNotFoundError(
                f"Branch '{branch}' not found in '{owner}/{repo}'",
                status_code=404,
            )
        return branches[branch]

    async def create_branch(
        self,
        owner: str,
        repo: str,
        new_branch: str,
        from_sha: str,
    ) -> dict[str, Any]:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        key = self._repo_key(owner, repo)
        if key not in self._repositories:
            raise GitHubResourceNotFoundError(
                f"Repository '{owner}/{repo}' not found", status_code=404
            )

        branches = self._branches.setdefault(key, {})
        if new_branch in branches:
            raise GitHubAPIResponseError(
                f"Reference 'refs/heads/{new_branch}' already exists", status_code=422
            )

        branches[new_branch] = GitHubBranch(name=new_branch, commit={"sha": from_sha})
        return {"ref": f"refs/heads/{new_branch}", "object": {"sha": from_sha}}

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> GitHubPullRequest:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        key = self._repo_key(owner, repo)
        if key not in self._repositories:
            raise GitHubResourceNotFoundError(
                f"Repository '{owner}/{repo}' not found", status_code=404
            )

        prs = self._pull_requests.setdefault(key, [])
        pr_number = self._next_pr_number
        self._next_pr_number += 1

        pr_obj = GitHubPullRequest(
            id=pr_number + 5000,
            number=pr_number,
            title=title,
            body=body,
            state="open",
            html_url=f"https://github.com/{owner}/{repo}/pull/{pr_number}",
            head_branch=head,
            base_branch=base,
        )
        prs.append(pr_obj)
        return pr_obj

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pull_number: int,
    ) -> GitHubPullRequest:
        validate_repo_params(owner, repo)
        self._check_simulated_errors(owner, repo)
        key = self._repo_key(owner, repo)
        for pr in self._pull_requests.get(key, []):
            if pr.number == pull_number:
                return pr
        raise GitHubResourceNotFoundError(
            f"Pull request #{pull_number} not found in '{owner}/{repo}'",
            status_code=404,
        )
