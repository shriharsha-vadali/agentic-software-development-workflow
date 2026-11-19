"""
app/execution/repair.py
-----------------------
Automated test analysis and controlled repair loop.
Coordinates running tests, capturing failures, querying the code generator
for targeted fixes, and re-testing up to a bounded retry limit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Artifact, Job
from app.execution.runner import ExecutionResult, SandboxRunner

if TYPE_CHECKING:
    from app.intelligence.coder import CodeGenerator
    from app.intelligence.planner import ImplementationPlan

logger = logging.getLogger(__name__)

DEFAULT_MAX_REPAIR_ATTEMPTS: int = 3


class RepairLoopResult:
    """Outcome of the automated test-repair cycle."""

    def __init__(
        self,
        success: bool,
        attempts: int,
        latest_execution: ExecutionResult,
        error_message: str | None = None,
    ) -> None:
        self.success = success
        self.attempts = attempts
        self.latest_execution = latest_execution
        self.error_message = error_message


class RepairEngine:
    """Manages the execute-evaluate-repair cycle."""

    def __init__(
        self,
        runner: SandboxRunner | None = None,
        coder: CodeGenerator | None = None,
        max_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    ) -> None:
        self.runner = runner or SandboxRunner()
        self.coder = coder
        self.max_attempts = max_attempts

    def extract_failure_summary(self, exec_result: ExecutionResult) -> str:
        """Extract relevant failure traces and messages from test output."""
        output = f"STDOUT:\n{exec_result.stdout}\nSTDERR:\n{exec_result.stderr}"
        # Keep the most relevant trailing 3000 chars to avoid exceeding prompt sizes
        if len(output) > 3000:
            return output[-3000:]
        return output

    async def run_tests_and_repair_loop(
        self,
        db: AsyncSession,
        job: Job,
        workspace_path: Path,
        plan: ImplementationPlan,
        codebase_context: str,
        test_target: str = "tests",
    ) -> RepairLoopResult:
        """
        Execute automated test suite with controlled repair attempts upon failure.

        Parameters
        ----------
        db : AsyncSession
            Active database session for storing test output artifacts.
        job : Job
            The running Job record.
        workspace_path : Path
            Path to the job's isolated workspace.
        plan : ImplementationPlan
            Approved plan guiding the changes.
        codebase_context : str
            Retrieved context from repository.
        test_target : str
            Pytest path or directory within workspace.
        """
        attempt = 0
        latest_exec: ExecutionResult | None = None

        while attempt < self.max_attempts:
            attempt += 1
            logger.info(
                "Executing tests for Job %d (Attempt %d/%d)", job.id, attempt, self.max_attempts
            )

            # 1. Run tests
            latest_exec = await self.runner.run_pytest(workspace_path, test_target=test_target)

            # 2. Persist test_output artifact
            test_content = (
                f"### Test Execution (Attempt {attempt})\n"
                f"- **Exit Code**: {latest_exec.exit_code}\n"
                f"- **Duration**: {latest_exec.duration_seconds}s\n\n"
                f"#### Output\n```\n{latest_exec.stdout}\n{latest_exec.stderr}\n```"
            )
            artifact = Artifact(
                job_id=job.id,
                artifact_type="test_output",
                content=test_content,
            )
            db.add(artifact)
            await db.commit()

            # 3. Check for success
            if latest_exec.success:
                logger.info("Tests PASSED for Job %d on attempt %d", job.id, attempt)
                return RepairLoopResult(
                    success=True,
                    attempts=attempt,
                    latest_execution=latest_exec,
                )

            logger.warning(
                "Tests FAILED for Job %d on attempt %d (Exit code: %d)",
                job.id,
                attempt,
                latest_exec.exit_code,
            )

            # 4. If attempts remain and coder is configured, attempt repair
            if attempt < self.max_attempts and self.coder is not None:
                failure_summary = self.extract_failure_summary(latest_exec)
                logger.info("Requesting AI repair generation for Job %d", job.id)
                try:
                    repair_result = await self.coder.generate_code(
                        plan=plan,
                        codebase_context=codebase_context,
                        repair_context=failure_summary,
                    )
                    self.coder.validate_and_apply_changes(job.id, repair_result.changes)
                except Exception as err:
                    logger.error("Failed during repair generation/application: %s", err)
                    return RepairLoopResult(
                        success=False,
                        attempts=attempt,
                        latest_execution=latest_exec,
                        error_message=f"Repair failed: {err}",
                    )

        assert latest_exec is not None
        return RepairLoopResult(
            success=False,
            attempts=attempt,
            latest_execution=latest_exec,
            error_message=f"Tests failed after {attempt} attempts",
        )
