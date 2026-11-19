"""
app/intelligence/workspace.py
-----------------------------
Workspace management for isolated job execution.
Handles directory isolation, safe path resolution, file discovery,
and language/framework detection.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Directory names and file patterns to ignore during repository scanning
IGNORE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    "site-packages",
    ".tox",
    ".coverage",
}

IGNORE_EXTENSIONS: set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".bin",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}

MAX_FILE_SIZE_BYTES: int = 500 * 1024  # 500 KB limit for source indexing


class WorkspaceManager:
    """Manages isolated disk workspaces for individual jobs."""

    def __init__(self, base_workspaces_dir: str | Path = "workspaces") -> None:
        self.base_dir = Path(base_workspaces_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_job_workspace(self, job_id: int) -> Path:
        """Return the absolute path of the isolated workspace for a given job."""
        return (self.base_dir / f"job_{job_id}").resolve()

    def create_workspace(self, job_id: int, clean_if_exists: bool = False) -> Path:
        """Create and return the workspace directory for the specified job."""
        path = self.get_job_workspace(job_id)
        if clean_if_exists and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_workspace(self, job_id: int) -> None:
        """Remove the workspace directory for the specified job."""
        path = self.get_job_workspace(job_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Cleaned up workspace for job %d at %s", job_id, path)

    def resolve_safe_path(self, job_id: int, relative_path: str | Path) -> Path:
        """
        Resolve a relative file path inside the job's workspace.
        Raises ValueError if path traversal attempts to escape the workspace boundary.
        """
        workspace = self.get_job_workspace(job_id)
        target = (workspace / relative_path).resolve()
        try:
            target.relative_to(workspace)
        except ValueError as err:
            raise ValueError(
                f"Path traversal detected: '{relative_path}' resolves outside workspace"
            ) from err
        return target

    def discover_files(self, workspace_path: Path) -> list[Path]:
        """
        Recursively discover all indexable text/source files in the workspace,
        filtering out ignored directories, binary extensions, and oversized files.
        """
        files: list[Path] = []
        if not workspace_path.exists():
            return files

        for root, dirs, filenames in os.walk(workspace_path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

            root_path = Path(root)
            for fname in filenames:
                file_path = root_path / fname
                ext = file_path.suffix.lower()

                if ext in IGNORE_EXTENSIONS:
                    continue

                try:
                    stat = file_path.stat()
                    if stat.st_size > MAX_FILE_SIZE_BYTES or stat.st_size == 0:
                        continue
                except OSError:
                    continue

                files.append(file_path)

        return sorted(files)

    def detect_project_type(self, workspace_path: Path) -> dict[str, Any]:
        """
        Detect languages, frameworks, test runners, and key configuration files.
        """
        configs: list[str] = []
        languages: set[str] = set()
        test_frameworks: set[str] = set()

        check_configs: dict[str, str] = {
            "pyproject.toml": "python",
            "setup.py": "python",
            "requirements.txt": "python",
            "package.json": "javascript",
            "tsconfig.json": "typescript",
            "Cargo.toml": "rust",
            "go.mod": "go",
            "pom.xml": "java",
            "build.gradle": "java",
        }

        for fname, lang in check_configs.items():
            if (workspace_path / fname).exists():
                configs.append(fname)
                languages.add(lang)

        # Detect tests
        test_dirs = ["tests", "test", "__tests__"]
        has_tests = any((workspace_path / d).exists() for d in test_dirs)
        if "python" in languages:
            test_frameworks.add("pytest")

        return {
            "primary_language": "python"
            if "python" in languages
            else (next(iter(languages), "unknown")),
            "languages": sorted(languages),
            "config_files": sorted(configs),
            "has_tests": has_tests,
            "test_frameworks": sorted(test_frameworks),
        }
