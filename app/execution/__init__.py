"""
app.execution package
---------------------
Sandboxed execution environment: subprocess runner, Docker container isolation,
command validation, failure analysis, and automated repair loop.
"""

from app.execution.repair import RepairEngine, RepairLoopResult
from app.execution.runner import (
    ALLOWED_COMMAND_BINARIES,
    DEFAULT_EXECUTION_TIMEOUT,
    CommandNotAllowedError,
    DockerExecutionBackend,
    DockerUnavailableError,
    ExecutionBackend,
    ExecutionResult,
    ExecutionTimeoutError,
    HostExecutionNotAllowedError,
    InvalidDockerImageError,
    LocalSubprocessBackend,
    SandboxExecutionError,
    SandboxRunner,
    validate_docker_image,
)

__all__ = [
    "ALLOWED_COMMAND_BINARIES",
    "DEFAULT_EXECUTION_TIMEOUT",
    "CommandNotAllowedError",
    "DockerExecutionBackend",
    "DockerUnavailableError",
    "ExecutionBackend",
    "ExecutionResult",
    "ExecutionTimeoutError",
    "HostExecutionNotAllowedError",
    "InvalidDockerImageError",
    "LocalSubprocessBackend",
    "RepairEngine",
    "RepairLoopResult",
    "SandboxExecutionError",
    "SandboxRunner",
    "validate_docker_image",
]
