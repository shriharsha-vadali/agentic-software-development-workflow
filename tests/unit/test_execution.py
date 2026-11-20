"""
tests/unit/test_execution.py
----------------------------
Unit tests for sandboxed execution backends (Docker container isolation & local subprocess)
and the automated repair engine (app/execution/).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, Job, JobStatus
from app.execution.repair import RepairEngine
from app.execution.runner import (
    CommandNotAllowedError,
    DockerExecutionBackend,
    DockerUnavailableError,
    ExecutionTimeoutError,
    HostExecutionNotAllowedError,
    InvalidDockerImageError,
    LocalSubprocessBackend,
    SandboxRunner,
    validate_docker_image,
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
    """Verify command whitelisting and environment sanitization."""

    def test_disallowed_command_blocked(self, tmp_path: Path) -> None:
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        with pytest.raises(CommandNotAllowedError, match="not permitted"):
            runner.validate_command(["curl", "https://malicious.com"])

        with pytest.raises(CommandNotAllowedError, match="not permitted"):
            runner.validate_command(["bash", "-c", "echo pwned"])

    def test_empty_command_blocked(self) -> None:
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        with pytest.raises(CommandNotAllowedError, match="Empty command"):
            runner.validate_command([])

    def test_environment_sanitization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "super_secret_pat")
        monkeypatch.setenv("LLM_API_KEY", "super_secret_llm")
        monkeypatch.setenv("NORMAL_VAR", "visible")

        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        clean_env = runner.sanitize_environment()

        assert "GITHUB_TOKEN" not in clean_env
        assert "LLM_API_KEY" not in clean_env
        assert clean_env.get("NORMAL_VAR") == "visible"
        assert clean_env.get("PYTHONUNBUFFERED") == "1"


class TestLocalSubprocessBackend:
    """Verify local host execution gates and safety controls."""

    async def test_local_host_execution_blocked_by_default(self, tmp_path: Path) -> None:
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=False))
        with pytest.raises(
            HostExecutionNotAllowedError, match="Host execution will never be silently used"
        ):
            await runner.run([sys.executable, "-c", "print('should_fail')"], cwd=tmp_path)

    async def test_run_python_command_permitted_explicitly(self, tmp_path: Path) -> None:
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        res = await runner.run([sys.executable, "-c", "print('sandbox_ok')"], cwd=tmp_path)

        assert res.success is True
        assert res.exit_code == 0
        assert "sandbox_ok" in res.stdout

    async def test_execution_timeout(self, tmp_path: Path) -> None:
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        with pytest.raises(ExecutionTimeoutError, match="timed out"):
            await runner.run(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=tmp_path,
                timeout=0.3,
            )


class TestDockerExecutionBackend:
    """Verify Docker container isolation, image validation, and security flags."""

    def test_image_validation_valid_images(self) -> None:
        validate_docker_image("python:3.11-slim")
        validate_docker_image("ghcr.io/my-org/custom-test-runner:v1.0.0")
        validate_docker_image("ubuntu:22.04")
        validate_docker_image("debian_slim")

    def test_image_validation_invalid_and_injection_images(self) -> None:
        invalid_images = [
            "python; rm -rf /",
            "python:3.11-slim && cat /etc/passwd",
            "image name with spaces",
            "image$foo",
            "--privileged",
            "-v /:/host:rw",
            "",
            "a" * 260,
        ]
        for img in invalid_images:
            with pytest.raises(InvalidDockerImageError, match="Invalid or unsafe Docker image"):
                validate_docker_image(img)

    def test_docker_command_construction_security_flags(self, tmp_path: Path) -> None:
        backend = DockerExecutionBackend(
            image="python:3.11-slim",
            network_disabled=True,
            memory_limit="512m",
            cpu_limit=1.5,
            pids_limit=150,
            read_only_rootfs=True,
            user="1001:1001",
        )
        cmd = backend.build_command(
            ["python", "-m", "pytest", "tests"], tmp_path, "test-container-123"
        )

        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert "--name" in cmd
        assert "test-container-123" in cmd

        # Network isolation
        assert "--network" in cmd
        assert cmd[cmd.index("--network") + 1] == "none"

        # Non-root user
        assert "--user" in cmd
        assert cmd[cmd.index("--user") + 1] == "1001:1001"

        # Resource limits
        assert "--memory" in cmd
        assert cmd[cmd.index("--memory") + 1] == "512m"
        assert "--cpus" in cmd
        assert cmd[cmd.index("--cpus") + 1] == "1.5"
        assert "--pids-limit" in cmd
        assert cmd[cmd.index("--pids-limit") + 1] == "150"

        # Capabilities & security options
        assert "--cap-drop" in cmd
        assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
        assert "--security-opt=no-new-privileges" in cmd

        # Filesystem restrictions & isolated mount
        assert "--read-only" in cmd
        assert "--tmpfs" in cmd
        assert "-v" in cmd
        workspace_mount = cmd[cmd.index("-v") + 1]
        assert workspace_mount.endswith(":/workspace:rw")
        assert "-w" in cmd
        assert cmd[cmd.index("-w") + 1] == "/workspace"

        # Safe environment variables
        assert "-e" in cmd
        assert "PYTHONUNBUFFERED=1" in cmd

        # Image and command
        assert "python:3.11-slim" in cmd
        assert cmd[-4:] == ["python", "-m", "pytest", "tests"]

    def test_docker_network_enabled_flag(self, tmp_path: Path) -> None:
        backend = DockerExecutionBackend(
            image="python:3.11-slim",
            network_disabled=False,
        )
        cmd = backend.build_command(["python", "--version"], tmp_path, "c-net")
        assert "--network" not in cmd

    async def test_docker_unavailable_raises_explicit_error_without_silent_fallback(
        self, tmp_path: Path
    ) -> None:
        backend = DockerExecutionBackend()
        # Mock is_docker_available to return False
        with patch.object(backend, "is_docker_available", return_value=False):
            with pytest.raises(
                DockerUnavailableError,
                match="Docker execution backend is active, but Docker is unavailable",
            ):
                await backend.run(["python", "--version"], tmp_path, timeout=5.0, env={})

    async def test_docker_execution_success_mocked(self, tmp_path: Path) -> None:
        backend = DockerExecutionBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"All tests passed!\n", b"")
        mock_proc.returncode = 0

        with (
            patch.object(backend, "is_docker_available", return_value=True),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            res = await backend.run(["pytest", "tests"], tmp_path, timeout=10.0, env={})

            assert res.success is True
            assert res.exit_code == 0
            assert "All tests passed!" in res.stdout

    async def test_docker_execution_timeout_cleans_up_container(self, tmp_path: Path) -> None:
        backend = DockerExecutionBackend()

        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = TimeoutError()

        mock_cleanup_proc = AsyncMock()
        mock_cleanup_proc.communicate.return_value = (b"", b"")
        mock_cleanup_proc.returncode = 0

        with (
            patch.object(backend, "is_docker_available", return_value=True),
            patch(
                "asyncio.create_subprocess_exec", side_effect=[mock_proc, mock_cleanup_proc]
            ) as mock_exec,
        ):
            with pytest.raises(ExecutionTimeoutError, match="timed out"):
                await backend.run(["pytest"], tmp_path, timeout=0.1, env={})

            # Verify that docker rm -f was called to clean up the container
            cleanup_call = mock_exec.call_args_list[1]
            assert cleanup_call[0][1] == "rm"
            assert cleanup_call[0][2] == "-f"

    def test_docker_workspace_mount_path_normalization_windows_and_linux(self) -> None:
        backend = DockerExecutionBackend()

        # Windows path with drive letter, backslashes, and spaces
        win_path = Path("C:/Users/developer/My Workspace/job_42")
        cmd_win = backend.build_command(["python", "--version"], win_path, "win-c")
        v_idx_win = cmd_win.index("-v")
        mount_win = cmd_win[v_idx_win + 1]
        assert "\\\\" not in mount_win
        assert "\\ " not in mount_win
        assert mount_win.endswith(":/workspace:rw")
        assert "My Workspace" in mount_win

        # Linux absolute path
        linux_path = Path("/home/runner/workspaces/job_99")
        cmd_linux = backend.build_command(["python", "--version"], linux_path, "linux-c")
        v_idx_linux = cmd_linux.index("-v")
        mount_linux = cmd_linux[v_idx_linux + 1]
        assert mount_linux.endswith(":/workspace:rw")
        assert "job_99" in mount_linux

    async def test_docker_execution_cancellation_cleans_up_and_propagates(
        self, tmp_path: Path
    ) -> None:
        backend = DockerExecutionBackend()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate.side_effect = asyncio.CancelledError()
        mock_proc.wait.return_value = 0

        mock_cleanup_proc = AsyncMock()
        mock_cleanup_proc.communicate.return_value = (b"", b"")
        mock_cleanup_proc.returncode = 0

        with (
            patch.object(backend, "is_docker_available", return_value=True),
            patch(
                "asyncio.create_subprocess_exec", side_effect=[mock_proc, mock_cleanup_proc]
            ) as mock_exec,
        ):
            with pytest.raises(asyncio.CancelledError):
                await backend.run(["pytest"], tmp_path, timeout=5.0, env={})

            # Verify that proc.kill() was called and docker rm -f was executed
            mock_proc.kill.assert_called_once()
            assert mock_exec.call_count == 2
            cleanup_call = mock_exec.call_args_list[1]
            assert cleanup_call[0][1] == "rm"
            assert cleanup_call[0][2] == "-f"

    async def test_docker_command_unexpected_failure_cleans_up(self, tmp_path: Path) -> None:
        backend = DockerExecutionBackend()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate.side_effect = RuntimeError("Subprocess communication failed")
        mock_proc.wait.return_value = 0

        mock_cleanup_proc = AsyncMock()
        mock_cleanup_proc.communicate.return_value = (b"", b"")
        mock_cleanup_proc.returncode = 0

        with (
            patch.object(backend, "is_docker_available", return_value=True),
            patch(
                "asyncio.create_subprocess_exec", side_effect=[mock_proc, mock_cleanup_proc]
            ) as mock_exec,
        ):
            with pytest.raises(RuntimeError, match="Subprocess communication failed"):
                await backend.run(["pytest"], tmp_path, timeout=5.0, env={})

            # Verify cleanup was triggered
            assert mock_exec.call_count == 2
            cleanup_call = mock_exec.call_args_list[1]
            assert cleanup_call[0][1] == "rm"
            assert cleanup_call[0][2] == "-f"

    async def test_docker_cleanup_failure_does_not_mask_original_error(
        self, tmp_path: Path
    ) -> None:
        backend = DockerExecutionBackend()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate.side_effect = TimeoutError()
        mock_proc.wait.return_value = 0

        # Simulate cleanup command failure
        mock_cleanup_proc = AsyncMock()
        mock_cleanup_proc.communicate.side_effect = OSError("Docker daemon socket closed")

        with (
            patch.object(backend, "is_docker_available", return_value=True),
            patch("asyncio.create_subprocess_exec", side_effect=[mock_proc, mock_cleanup_proc]),
        ):
            # Must raise ExecutionTimeoutError, not the cleanup OSError!
            with pytest.raises(ExecutionTimeoutError, match="timed out"):
                await backend.run(["pytest"], tmp_path, timeout=0.1, env={})

    async def test_docker_cleanup_timeout_kills_subprocess_and_preserves_error(
        self, tmp_path: Path
    ) -> None:
        backend = DockerExecutionBackend()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate.side_effect = TimeoutError()
        mock_proc.wait.return_value = 0

        # Simulate cleanup command timing out
        mock_cleanup_proc = AsyncMock()
        mock_cleanup_proc.kill = MagicMock()
        mock_cleanup_proc.communicate.side_effect = TimeoutError()
        mock_cleanup_proc.wait.return_value = 0

        with (
            patch.object(backend, "is_docker_available", return_value=True),
            patch("asyncio.create_subprocess_exec", side_effect=[mock_proc, mock_cleanup_proc]),
        ):
            with pytest.raises(ExecutionTimeoutError, match="timed out"):
                await backend.run(["pytest"], tmp_path, timeout=0.1, env={})

            mock_cleanup_proc.kill.assert_called_once()

    async def test_cleanup_container_direct_timeout(self) -> None:
        backend = DockerExecutionBackend()
        mock_cleanup_proc = AsyncMock()
        mock_cleanup_proc.kill = MagicMock()
        mock_cleanup_proc.communicate.side_effect = TimeoutError()
        mock_cleanup_proc.wait.return_value = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_cleanup_proc):
            await backend._cleanup_container("test-container", timeout=0.01)
            mock_cleanup_proc.kill.assert_called_once()


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
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        engine = RepairEngine(runner=runner)
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
        runner = SandboxRunner(backend=LocalSubprocessBackend(allow_host_execution=True))
        engine = RepairEngine(runner=runner, max_attempts=2)
        result = await engine.run_tests_and_repair_loop(
            test_session, job, ws, plan, "context", test_target="test_fail.py"
        )

        assert result.success is False
        assert result.attempts == 2
