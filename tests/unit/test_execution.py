"""
tests/unit/test_execution.py
----------------------------
Unit tests for the sandboxed runner and repair engine (app/execution/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, Job, JobStatus
from app.execution.repair import RepairEngine
from app.execution.runner import (
    CommandNotAllowedError,
    ExecutionTimeoutError,
    SandboxRunner,
)
from app.intelligence.planner import ImplementationPlan

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_session() -> AsyncSession:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class TestSandboxRunner:
    """Verify command whitelisting, environment sanitization, and execution."""

    def test_disallowed_command_blocked(self, tmp_path: Path) -> None:
        runner = SandboxRunner()
        with pytest.raises(CommandNotAllowedError, match="not permitted"):
            runner.validate_command(["curl", "https://malicious.com"])

        with pytest.raises(CommandNotAllowedError, match="not permitted"):
            runner.validate_command(["bash", "-c", "echo pwned"])

    def test_environment_sanitization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "super_secret_pat")
        monkeypatch.setenv("LLM_API_KEY", "super_secret_llm")
        monkeypatch.setenv("NORMAL_VAR", "visible")

        runner = SandboxRunner()
        clean_env = runner.sanitize_environment()

        assert "GITHUB_TOKEN" not in clean_env
        assert "LLM_API_KEY" not in clean_env
        assert clean_env.get("NORMAL_VAR") == "visible"

    async def test_run_python_command(self, tmp_path: Path) -> None:
        runner = SandboxRunner()
        res = await runner.run([sys.executable, "-c", "print('sandbox_ok')"], cwd=tmp_path)

        assert res.success is True
        assert res.exit_code == 0
        assert "sandbox_ok" in res.stdout

    async def test_execution_timeout(self, tmp_path: Path) -> None:
        runner = SandboxRunner()
        with pytest.raises(ExecutionTimeoutError):
            await runner.run(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=tmp_path,
                timeout=0.3,
            )


class TestRepairEngine:
    """Verify test-repair-loop behavior and artifact persistence."""

    async def test_repair_loop_success_first_try(
        self, test_session: AsyncSession, tmp_path: Path
    ) -> None:
        # Create passing test
        ws = tmp_path / "job_1"
        ws.mkdir()
        (ws / "test_ok.py").write_text("def test_true(): assert True\n", encoding="utf-8")

        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            status=JobStatus.RUNNING_TESTS,
        )
        test_session.add(job)
        await test_session.commit()

        plan = ImplementationPlan(
            summary="Test plan",
            problem_statement="Test",
            technical_approach="Test",
            test_plan="Test",
        )
        engine = RepairEngine()
        result = await engine.run_tests_and_repair_loop(
            test_session, job, ws, plan, "context", test_target="test_ok.py"
        )

        assert result.success is True
        assert result.attempts == 1

    async def test_repair_loop_exhausts_retries(
        self, test_session: AsyncSession, tmp_path: Path
    ) -> None:
        # Create failing test
        ws = tmp_path / "job_2"
        ws.mkdir()
        (ws / "test_fail.py").write_text("def test_f(): assert False\n", encoding="utf-8")

        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/2",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=2,
            status=JobStatus.RUNNING_TESTS,
        )
        test_session.add(job)
        await test_session.commit()

        plan = ImplementationPlan(
            summary="Test plan",
            problem_statement="Test",
            technical_approach="Test",
            test_plan="Test",
        )
        # Without coder, will run max_attempts and stop
        engine = RepairEngine(max_attempts=2)
        result = await engine.run_tests_and_repair_loop(
            test_session, job, ws, plan, "context", test_target="test_fail.py"
        )

        assert result.success is False
        assert result.attempts == 2
