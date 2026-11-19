"""
app/execution/runner.py
-----------------------
Sandboxed subprocess execution runner.
Executes test commands in an isolated workspace with command whitelisting,
timeout enforcement, secret scrubbing, and output capture.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Permitted command executable prefixes
ALLOWED_COMMAND_BINARIES: set[str] = {
    "pytest",
    "python",
    "python.exe",
    "python3",
    "ruff",
    "mypy",
    "git",
}

# Sensitive environment variables to strip from execution environment
SENSITIVE_ENV_VARS: set[str] = {
    "GITHUB_TOKEN",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "SECRET_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_CLIENT_SECRET",
}

DEFAULT_EXECUTION_TIMEOUT: float = 60.0


@dataclass
class ExecutionResult:
    """Result of an isolated command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class SandboxExecutionError(Exception):
    """Base exception for execution runner errors."""


class CommandNotAllowedError(SandboxExecutionError):
    """Raised when an unwhitelisted or potentially dangerous command is executed."""


class ExecutionTimeoutError(SandboxExecutionError):
    """Raised when subprocess execution exceeds the configured timeout."""


class SandboxRunner:
    """Executes permitted test and analysis commands inside a job workspace."""

    def __init__(self, default_timeout: float = DEFAULT_EXECUTION_TIMEOUT) -> None:
        self.default_timeout = default_timeout

    def validate_command(self, cmd_args: list[str]) -> None:
        """Verify that the command binary is in the allowed whitelist."""
        if not cmd_args:
            raise CommandNotAllowedError("Empty command is not allowed.")

        raw_binary = cmd_args[0].replace("\\", "/").split("/")[-1].lower()
        if raw_binary not in ALLOWED_COMMAND_BINARIES:
            raise CommandNotAllowedError(
                f"Binary '{raw_binary}' is not permitted in sandboxed execution. "
                f"Allowed: {sorted(ALLOWED_COMMAND_BINARIES)}"
            )

    def sanitize_environment(self) -> dict[str, str]:
        """Produce a sanitized copy of environment variables with secrets stripped."""
        env = os.environ.copy()
        for secret_key in SENSITIVE_ENV_VARS:
            env.pop(secret_key, None)
        # Ensure python writes UTF-8 output
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    async def run(
        self,
        command: str | list[str],
        cwd: Path,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Execute command inside workspace `cwd` with timeout and sanitized environment.

        Parameters
        ----------
        command : str | list[str]
            Command string or argument list.
        cwd : Path
            Working directory (must exist).
        timeout : float | None
            Execution timeout in seconds.
        """
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        if isinstance(command, str):
            cmd_args = shlex.split(command, posix=(os.name != "nt"))
        else:
            cmd_args = list(command)

        self.validate_command(cmd_args)
        exec_timeout = timeout or self.default_timeout
        env = self.sanitize_environment()

        logger.info("Executing sandboxed command in %s: %s", cwd, " ".join(cmd_args))
        start_time = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                cmd_args[0],
                *cmd_args[1:],
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=exec_timeout
            )
            duration = time.monotonic() - start_time
            exit_code = proc.returncode if proc.returncode is not None else -1

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            return ExecutionResult(
                command=" ".join(cmd_args),
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=round(duration, 3),
            )

        except TimeoutError as exc:
            duration = time.monotonic() - start_time
            logger.warning("Command timed out after %.1fs: %s", exec_timeout, " ".join(cmd_args))
            try:
                proc.kill()
            except Exception:
                pass
            raise ExecutionTimeoutError(
                f"Execution timed out after {exec_timeout} seconds"
            ) from exc

    async def run_pytest(
        self,
        cwd: Path,
        test_target: str = "tests",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Convenience method to execute pytest using the current Python interpreter."""
        cmd = [sys.executable, "-m", "pytest", test_target, "-v"]
        return await self.run(cmd, cwd=cwd, timeout=timeout)
