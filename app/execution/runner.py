"""
app/execution/runner.py
-----------------------
Sandboxed execution runner with container isolation.
Executes test and analysis commands in disposable Docker containers or an
explicitly permitted local subprocess backend. Enforces command whitelisting,
resource limits, network disabling, non-root users, timeouts, and credential scrubbing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

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
    "API_KEY",
}

DEFAULT_EXECUTION_TIMEOUT: float = 60.0

# Strict regex for Docker image naming (alphanumeric start, safe chars, optional tag)
DOCKER_IMAGE_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*(?:/[a-zA-Z0-9][a-zA-Z0-9_.-]*)*(?::[a-zA-Z0-9_.-]+)?$"
)


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
    """Raised when execution exceeds the configured timeout."""


class DockerUnavailableError(SandboxExecutionError):
    """Raised when Docker backend is requested but Docker CLI/daemon is unavailable."""


class HostExecutionNotAllowedError(SandboxExecutionError):
    """Raised when local host execution is attempted without explicit authorization."""


class InvalidDockerImageError(SandboxExecutionError):
    """Raised when a Docker image name contains invalid characters or shell metacharacters."""


def validate_docker_image(image_name: str) -> None:
    """
    Validate Docker image name strictly against permitted regex pattern.
    Rejects spaces, command injection characters, flags, and malformed tags.
    """
    if not image_name or len(image_name) > 255 or not DOCKER_IMAGE_PATTERN.match(image_name):
        raise InvalidDockerImageError(
            f"Invalid or unsafe Docker image name: '{image_name}'. "
            "Must be alphanumeric with optional repository and tag (e.g. 'python:3.11-slim')."
        )


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    async def run(
        self,
        cmd_args: list[str],
        cwd: Path,
        timeout: float,
        env: dict[str, str],
    ) -> ExecutionResult:
        """Execute command with timeout and environment."""


class DockerExecutionBackend(ExecutionBackend):
    """
    Docker-based disposable container execution backend.
    Enforces non-root user, network disabled, read-only rootfs, dropped capabilities,
    and memory/CPU/PIDs resource constraints.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        network_disabled: bool = True,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        pids_limit: int = 100,
        read_only_rootfs: bool = True,
        user: str = "1000:1000",
        docker_binary: str = "docker",
    ) -> None:
        validate_docker_image(image)
        self.image = image
        self.network_disabled = network_disabled
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.read_only_rootfs = read_only_rootfs
        self.user = user
        self.docker_binary = docker_binary

    async def is_docker_available(self) -> bool:
        """Check if Docker CLI is callable and daemon is responsive."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=3.0)
            return proc.returncode == 0
        except Exception:
            return False

    def build_command(
        self,
        cmd_args: list[str],
        cwd: Path,
        container_name: str,
    ) -> list[str]:
        """Construct isolated 'docker run' argument vector."""
        resolved_workspace = cwd.resolve()

        docker_args = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
        ]

        if self.network_disabled:
            docker_args.extend(["--network", "none"])

        if self.user:
            docker_args.extend(["--user", self.user])

        # Resource limits
        docker_args.extend(["--memory", self.memory_limit])
        docker_args.extend(["--cpus", str(self.cpu_limit)])
        docker_args.extend(["--pids-limit", str(self.pids_limit)])

        # Security privileges & capabilities
        docker_args.extend(["--cap-drop", "ALL"])
        docker_args.append("--security-opt=no-new-privileges")

        # Filesystem restrictions
        if self.read_only_rootfs:
            docker_args.append("--read-only")
            docker_args.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        # Workspace mount (ONLY the specific job workspace)
        # Normalize to POSIX style (forward slashes) for Docker Desktop on Windows & Linux
        workspace_mount = f"{resolved_workspace.as_posix()}:/workspace:rw"
        docker_args.extend(["-v", workspace_mount])
        docker_args.extend(["-w", "/workspace"])

        # Safe, unprivileged environment variables (never pass secrets)
        docker_args.extend(["-e", "PYTHONUNBUFFERED=1"])
        docker_args.extend(["-e", "PYTHONIOENCODING=utf-8"])
        docker_args.extend(["-e", "PYTHONDONTWRITEBYTECODE=1"])
        docker_args.extend(["-e", "HOME=/tmp"])

        # Image
        docker_args.append(self.image)

        # Container command: map host python path to standard container binary
        mapped_cmd = list(cmd_args)
        raw_bin = mapped_cmd[0].replace("\\", "/").split("/")[-1].lower()
        if raw_bin in ("python.exe", "python"):
            mapped_cmd[0] = "python"

        docker_args.extend(mapped_cmd)
        return docker_args

    DEFAULT_CLEANUP_TIMEOUT: float = 10.0

    async def _cleanup_container(
        self, container_name: str, timeout: float = DEFAULT_CLEANUP_TIMEOUT
    ) -> None:
        """Forcefully remove a specific container with a bounded timeout, ignoring failures."""
        cleanup_proc: asyncio.subprocess.Process | None = None
        try:
            cleanup_proc = await asyncio.create_subprocess_exec(
                self.docker_binary,
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(cleanup_proc.communicate(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Container cleanup for %s timed out after %.1fs. "
                "Forcefully killing cleanup process.",
                container_name,
                timeout,
            )
            if cleanup_proc is not None:
                try:
                    cleanup_proc.kill()
                except Exception:
                    pass
                try:
                    await cleanup_proc.wait()
                except Exception:
                    pass
        except Exception as err:
            logger.warning("Failed to clean up container %s: %s", container_name, err)

    async def run(
        self,
        cmd_args: list[str],
        cwd: Path,
        timeout: float,
        env: dict[str, str],
    ) -> ExecutionResult:
        if not await self.is_docker_available():
            raise DockerUnavailableError(
                "Docker execution backend is active, but Docker is unavailable or not running. "
                "Please start Docker Desktop/daemon, or for container-less development/testing "
                "explicitly configure EXECUTION_BACKEND=local_subprocess and "
                "ALLOW_HOST_EXECUTION=true. Host execution will never be silently used."
            )

        container_name = f"agentic-exec-{uuid.uuid4().hex[:12]}"
        docker_cmd = self.build_command(cmd_args, cwd, container_name)
        logger.info(
            "Executing containerized command in %s (container: %s)",
            cwd,
            container_name,
        )
        start_time = time.monotonic()
        proc: asyncio.subprocess.Process | None = None
        needs_cleanup = False

        try:
            proc = await asyncio.create_subprocess_exec(
                docker_cmd[0],
                *docker_cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            needs_cleanup = True
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.monotonic() - start_time
            exit_code = proc.returncode if proc.returncode is not None else -1

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            # Normal exit completed; container was removed by --rm
            needs_cleanup = False

            return ExecutionResult(
                command=" ".join(cmd_args),
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_seconds=round(duration, 3),
            )

        except TimeoutError as exc:
            duration = time.monotonic() - start_time
            logger.warning(
                "Container execution timed out after %.1fs: %s. Cleaning up container %s.",
                timeout,
                " ".join(cmd_args),
                container_name,
            )
            raise ExecutionTimeoutError(
                f"Container execution timed out after {timeout} seconds"
            ) from exc
        finally:
            if needs_cleanup:
                # 1. Terminate/kill Docker CLI subprocess if still active
                if proc is not None and proc.returncode is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    # Reap process
                    try:
                        reap_task = asyncio.create_task(proc.wait())
                        try:
                            await asyncio.shield(reap_task)
                        except asyncio.CancelledError:
                            await reap_task
                    except Exception:
                        pass

                # 2. Forcefully remove container using a shielded task so cancellation
                # of the parent coroutine does not abort the container cleanup!
                cleanup_task = asyncio.create_task(self._cleanup_container(container_name))
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    await cleanup_task
                    raise
                except Exception as cleanup_err:
                    logger.warning(
                        "Error during shielded container cleanup for %s: %s",
                        container_name,
                        cleanup_err,
                    )


class LocalSubprocessBackend(ExecutionBackend):
    """
    Local host subprocess execution backend.
    Requires explicit authorization (allow_host_execution=True).
    Never used as a silent fallback when Docker is unavailable.
    """

    def __init__(self, allow_host_execution: bool = False) -> None:
        self.allow_host_execution = allow_host_execution

    async def run(
        self,
        cmd_args: list[str],
        cwd: Path,
        timeout: float,
        env: dict[str, str],
    ) -> ExecutionResult:
        if not self.allow_host_execution:
            raise HostExecutionNotAllowedError(
                "Local host execution is disabled by security policy. "
                "To permit host execution in development/testing, explicitly configure "
                "EXECUTION_BACKEND=local_subprocess and ALLOW_HOST_EXECUTION=true. "
                "Host execution will never be silently used."
            )

        logger.info("Executing local subprocess in %s: %s", cwd, " ".join(cmd_args))
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
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
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
            logger.warning("Command timed out after %.1fs: %s", timeout, " ".join(cmd_args))
            try:
                proc.kill()
            except Exception:
                pass
            raise ExecutionTimeoutError(f"Execution timed out after {timeout} seconds") from exc


class SandboxRunner:
    """
    Executes permitted test and analysis commands inside an isolated job workspace.
    Delegates to DockerExecutionBackend (default) or LocalSubprocessBackend (if authorized).
    """

    def __init__(
        self,
        default_timeout: float = DEFAULT_EXECUTION_TIMEOUT,
        backend: ExecutionBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.default_timeout = default_timeout
        self.settings = settings
        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._create_backend()

    def _create_backend(self) -> ExecutionBackend:
        from app.core.config import get_settings

        cfg = self.settings or get_settings()
        backend_type = (getattr(cfg, "execution_backend", "docker") or "docker").lower()

        if backend_type == "docker":
            return DockerExecutionBackend(
                image=getattr(cfg, "docker_image", "python:3.11-slim"),
                network_disabled=getattr(cfg, "docker_network_disabled", True),
                memory_limit=getattr(cfg, "docker_memory_limit", "512m"),
                cpu_limit=getattr(cfg, "docker_cpu_limit", 1.0),
                pids_limit=getattr(cfg, "docker_pids_limit", 100),
                read_only_rootfs=getattr(cfg, "docker_read_only_rootfs", True),
                user=getattr(cfg, "docker_user", "1000:1000"),
            )
        elif backend_type == "local_subprocess":
            return LocalSubprocessBackend(
                allow_host_execution=getattr(cfg, "allow_host_execution", False),
            )
        else:
            raise SandboxExecutionError(
                f"Unknown execution backend: '{backend_type}'. "
                "Supported backends: 'docker', 'local_subprocess'."
            )

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

        return await self.backend.run(
            cmd_args=cmd_args,
            cwd=cwd,
            timeout=exec_timeout,
            env=env,
        )

    async def run_pytest(
        self,
        cwd: Path,
        test_target: str = "tests",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Convenience method to execute pytest."""
        if isinstance(self.backend, DockerExecutionBackend):
            cmd = ["python", "-m", "pytest", test_target, "-v"]
        else:
            cmd = [sys.executable, "-m", "pytest", test_target, "-v"]
        return await self.run(cmd, cwd=cwd, timeout=timeout)
