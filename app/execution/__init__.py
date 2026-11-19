"""
app.execution package
---------------------
Sandboxed execution environment: subprocess runner, command validation,
test execution, failure analysis, and automated repair loop.
"""

from app.execution.repair import RepairEngine, RepairLoopResult
from app.execution.runner import (
    CommandNotAllowedError,
    ExecutionResult,
    ExecutionTimeoutError,
    SandboxExecutionError,
    SandboxRunner,
)

__all__ = [
    "CommandNotAllowedError",
    "ExecutionResult",
    "ExecutionTimeoutError",
    "RepairEngine",
    "RepairLoopResult",
    "SandboxExecutionError",
    "SandboxRunner",
]
