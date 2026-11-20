"""
app/services/git_service.py
---------------------------
Git operations within an isolated workspace.
Handles branch creation, committing approved changes, diff generation,
and repository inspection without touching the developer's main branch.
Enforces security boundaries by disabling repository hooks for all Git operations.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.execution.runner import LocalSubprocessBackend, SandboxRunner

logger = logging.getLogger(__name__)


class GitServiceError(Exception):
    """Base exception for workspace git operations."""


class GitWorkspaceService:
    """Manages Git version control actions inside an isolated job workspace."""

    def __init__(
        self,
        runner: SandboxRunner | None = None,
        empty_hooks_dir: Path | None = None,
    ) -> None:
        self.runner = runner or SandboxRunner(
            backend=LocalSubprocessBackend(allow_host_execution=True)
        )
        if empty_hooks_dir is None:
            try:
                raw_dir = tempfile.mkdtemp(prefix="agentic_empty_hooks_")
                self.empty_hooks_dir = Path(raw_dir).resolve()
            except Exception as exc:
                raise GitServiceError(
                    f"Failed to create secure empty git hooks directory: {exc}"
                ) from exc
        else:
            empty_hooks_dir.mkdir(parents=True, exist_ok=True)
            self.empty_hooks_dir = empty_hooks_dir

    def _build_git_cmd(self, git_args: list[str]) -> list[str]:
        """
        Construct a Git command vector with security overrides.
        Disables repository-controlled hooks by redirecting core.hooksPath
        to a dedicated, controlled empty hooks directory.
        """
        return [
            "git",
            "-c",
            f"core.hooksPath={self.empty_hooks_dir.as_posix()}",
            *git_args,
        ]

    async def init_or_verify_repo(self, workspace_path: Path) -> None:
        """Initialize a local git repository in workspace if one does not exist."""
        git_dir = workspace_path / ".git"
        if not git_dir.exists():
            res = await self.runner.run(self._build_git_cmd(["init"]), cwd=workspace_path)
            if not res.success:
                raise GitServiceError(f"Failed to git init workspace: {res.stderr}")
            # Set local user config so commits succeed in automated environments
            await self.runner.run(
                self._build_git_cmd(["config", "user.name", "Agentic Developer Bot"]),
                cwd=workspace_path,
            )
            await self.runner.run(
                self._build_git_cmd(["config", "user.email", "agent@developer.local"]),
                cwd=workspace_path,
            )

    async def create_feature_branch(
        self,
        workspace_path: Path,
        branch_name: str,
    ) -> str:
        """Create and checkout a dedicated feature branch for the job."""
        await self.init_or_verify_repo(workspace_path)

        res = await self.runner.run(
            self._build_git_cmd(["checkout", "-b", branch_name]),
            cwd=workspace_path,
        )
        if not res.success:
            # If branch already exists, try checking it out
            res_checkout = await self.runner.run(
                self._build_git_cmd(["checkout", branch_name]),
                cwd=workspace_path,
            )
            if not res_checkout.success:
                raise GitServiceError(
                    f"Failed to create/checkout feature branch '{branch_name}': {res.stderr}"
                )

        logger.info("Checked out feature branch '%s' in %s", branch_name, workspace_path)
        return branch_name

    async def stage_and_commit(
        self,
        workspace_path: Path,
        commit_message: str,
    ) -> str:
        """Stage all workspace changes and commit with the specified message."""
        # git add -A
        add_res = await self.runner.run(
            self._build_git_cmd(["add", "-A"]),
            cwd=workspace_path,
        )
        if not add_res.success:
            raise GitServiceError(f"git add failed: {add_res.stderr}")

        # git commit -m
        commit_res = await self.runner.run(
            self._build_git_cmd(["commit", "-m", commit_message]),
            cwd=workspace_path,
        )
        if not commit_res.success and "nothing to commit" not in commit_res.stdout:
            raise GitServiceError(f"git commit failed: {commit_res.stderr}")

        # Get commit SHA
        sha_res = await self.runner.run(
            self._build_git_cmd(["rev-parse", "HEAD"]),
            cwd=workspace_path,
        )
        sha = sha_res.stdout.strip()
        logger.info("Committed changes in %s (SHA: %s)", workspace_path, sha)
        return sha

    async def get_diff(self, workspace_path: Path, base_ref: str = "HEAD~1") -> str:
        """Return the unified diff of the latest commit."""
        res = await self.runner.run(
            self._build_git_cmd(["diff", base_ref]),
            cwd=workspace_path,
        )
        if res.success:
            return res.stdout
        # Fallback to diff of working directory against HEAD
        fallback_res = await self.runner.run(
            self._build_git_cmd(["diff", "HEAD"]),
            cwd=workspace_path,
        )
        return fallback_res.stdout if fallback_res.success else ""
