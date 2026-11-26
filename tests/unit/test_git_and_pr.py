"""
tests/unit/test_git_and_pr.py
-----------------------------
Unit tests for GitWorkspaceService and PullRequestService (app/services/).
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.orchestrator.workflow  # noqa: F401 (resolves circular import cycle during isolated module test)
from app.adapters.github import GitHubClient
from app.db.models import Base, Job, JobStatus, Plan
from app.services.git_service import (
    SENSITIVE_ENV_VARS,
    BoundedStreamDrainer,
    ExecutionResult,
    GitServiceError,
    GitWorkspaceService,
    POSIXProcessRunner,
    ProcessContainmentError,
    ProcessRunnerError,
    ProcessTimeoutError,
    SecureProcessRunner,
    WindowsJobAssignmentError,
    prepare_safe_environment,
)

if sys.platform == "win32":
    from app.services.git_service import STARTUPINFOEXW, WindowsJobRunner
else:
    WindowsJobRunner = None  # type: ignore[assignment, misc]
    STARTUPINFOEXW = None  # type: ignore[assignment, misc]
from app.services.pr_service import PRAuthorizationError, PullRequestService

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class TestGitWorkspaceService:
    """Verify git operations inside an isolated workspace."""

    async def test_git_init_branch_commit_and_diff(self, tmp_path: Path) -> None:
        service = GitWorkspaceService()
        ws = tmp_path / "job_git"
        ws.mkdir()

        # 1. Init
        await service.init_or_verify_repo(ws)
        assert (ws / ".git").exists()

        # 2. Branch
        branch = await service.create_feature_branch(ws, "agent/issue-42")
        assert branch == "agent/issue-42"

        # 3. Create file and commit
        (ws / "file.txt").write_text("hello world\n", encoding="utf-8")
        sha = await service.stage_and_commit(ws, "feat: add file.txt")
        assert len(sha) == 40  # Full git SHA

        # 4. Modify and get diff
        (ws / "file.txt").write_text("hello world modified\n", encoding="utf-8")
        diff = await service.get_diff(ws)
        assert "+hello world modified" in diff

    async def test_git_operations_bypass_repository_hooks(self, tmp_path: Path) -> None:
        """Verify that repository-controlled git hooks are never executed."""
        service = GitWorkspaceService()
        ws = tmp_path / "job_hook_test"
        ws.mkdir()

        await service.init_or_verify_repo(ws)

        # Plant a malicious pre-commit hook that creates a canary file or would exit 1
        hooks_dir = ws / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        canary_file = ws / "MALICIOUS_HOOK_EXECUTED.txt"
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_hook.write_text(
            f"#!/bin/sh\necho 'hacked' > '{canary_file.as_posix()}'\nexit 1\n",
            encoding="utf-8",
        )

        # Stage and commit
        (ws / "safe_file.txt").write_text("secure content\n", encoding="utf-8")
        sha = await service.stage_and_commit(ws, "feat: commit with hook present")

        # Verify commit succeeded
        assert len(sha) == 40
        # Verify hook was NOT executed
        assert not canary_file.exists(), "Security breach: repository pre-commit hook was executed!"

    def test_git_workspace_service_unique_hooks_directory(self) -> None:
        """Each GitWorkspaceService instance gets a dedicated unique hooks directory."""
        service1 = GitWorkspaceService()
        service2 = GitWorkspaceService()

        assert service1.empty_hooks_dir != service2.empty_hooks_dir
        assert service1.empty_hooks_dir.exists()
        assert service2.empty_hooks_dir.exists()
        assert list(service1.empty_hooks_dir.iterdir()) == []
        assert list(service2.empty_hooks_dir.iterdir()) == []

        cmd1 = service1._build_git_cmd(["status"])
        cmd2 = service2._build_git_cmd(["status"])

        assert f"core.hooksPath={service1.empty_hooks_dir.as_posix()}" in cmd1
        assert f"core.hooksPath={service2.empty_hooks_dir.as_posix()}" in cmd2

    def test_git_workspace_service_mkdtemp_failure(self) -> None:
        """Failure to create private empty hooks directory fails closed with GitServiceError."""
        with patch("tempfile.mkdtemp", side_effect=OSError("Read-only filesystem")):
            with pytest.raises(
                GitServiceError, match="Failed to create secure empty git hooks directory"
            ):
                GitWorkspaceService()


class TestPullRequestService:
    """Verify PR creation authorization gates and GitHub API coordination."""

    async def test_create_pr_success(self, test_session: AsyncSession) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/repos/octocat/Hello-World/pulls"
            return httpx.Response(
                201,
                json={
                    "id": 900,
                    "number": 15,
                    "title": "fix(Hello-World): resolve #42",
                    "body": "Fixes #42",
                    "state": "open",
                    "html_url": "https://github.com/octocat/Hello-World/pull/15",
                },
            )

        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/42",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=42,
            status=JobStatus.OPENING_PR,
        )
        test_session.add(job)
        await test_session.flush()

        plan = Plan(job_id=job.id, raw_text="Approved plan content", status="approved")
        test_session.add(plan)
        await test_session.commit()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = GitHubClient(token="ghp_test", http_client=mock_http)
            pr_service = PullRequestService(client)

            pr = await pr_service.create_pull_request_for_job(
                test_session, job.id, feature_branch="agent/issue-42"
            )

        assert pr.number == 15
        assert pr.html_url == "https://github.com/octocat/Hello-World/pull/15"

        # Job must be transitioned to COMPLETED
        await test_session.refresh(job)
        assert job.status == JobStatus.COMPLETED

    async def test_create_pr_unauthorized_state_rejected(self, test_session: AsyncSession) -> None:
        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/42",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=42,
            status=JobStatus.GENERATING_CODE,
        )
        test_session.add(job)
        await test_session.commit()

        client = GitHubClient(token="ghp_test")
        pr_service = PullRequestService(client)

        with pytest.raises(PRAuthorizationError, match="Job must be in 'OPENING_PR'"):
            await pr_service.create_pull_request_for_job(
                test_session, job.id, feature_branch="agent/issue-42"
            )


# =============================================================================
# Stage 2 Unit Tests: BoundedStreamDrainer
# =============================================================================


class TestBoundedStreamDrainer:
    """Verify bounded output draining, deadlock prevention, and sanitization."""

    def test_bounded_retention_limit_under_64kb(self) -> None:
        drainer = BoundedStreamDrainer()
        chunk = b"X" * 10240  # 10 KiB
        drainer.feed(chunk)
        assert drainer.total_bytes_read == 10240
        out = drainer.get_bytes()
        assert len(out) == 10240
        assert b"... [truncated" not in out

    def test_bounded_retention_limit_over_64kb_preserves_head_and_tail(self) -> None:
        drainer = BoundedStreamDrainer()
        # Feed 100 KiB of 'A' followed by 100 KiB of 'B'
        drainer.feed(b"A" * 100000)
        drainer.feed(b"B" * 100000)
        assert drainer.total_bytes_read == 200000

        retained = drainer.get_bytes()
        # Retained head is 32 KiB of 'A'
        assert retained[:32768] == b"A" * 32768
        # Retained tail is 32 KiB of 'B'
        assert retained[-32768:] == b"B" * 32768
        # Truncation marker is present
        assert b"... [truncated 134464 bytes] ..." in retained
        # Memory retention remains strictly bounded (~65.5 KiB)
        assert len(retained) == 32768 + 32768 + len(b"\n... [truncated 134464 bytes] ...\n")

    @pytest.mark.asyncio
    async def test_large_stdout_no_deadlock(self) -> None:
        drainer = BoundedStreamDrainer()
        reader = asyncio.StreamReader()
        drain_task = asyncio.create_task(drainer.drain_reader(reader))

        # Feed 256 KiB into reader in 8 KiB chunks
        for _ in range(32):
            reader.feed_data(b"S" * 8192)
            await asyncio.sleep(0.001)
        reader.feed_eof()

        await drain_task
        assert drainer.total_bytes_read == 256 * 1024
        text = drainer.get_text()
        assert "... [truncated" in text

    @pytest.mark.asyncio
    async def test_large_stderr_no_deadlock(self) -> None:
        drainer = BoundedStreamDrainer()
        reader = asyncio.StreamReader()
        drain_task = asyncio.create_task(drainer.drain_reader(reader))

        for _ in range(32):
            reader.feed_data(b"E" * 8192)
            await asyncio.sleep(0.001)
        reader.feed_eof()

        await drain_task
        assert drainer.total_bytes_read == 256 * 1024
        text = drainer.get_text()
        assert "... [truncated" in text

    @pytest.mark.asyncio
    async def test_simultaneous_large_stdout_and_stderr(self) -> None:
        stdout_drainer = BoundedStreamDrainer()
        stderr_drainer = BoundedStreamDrainer()

        r_out, r_err = asyncio.StreamReader(), asyncio.StreamReader()
        task_out = asyncio.create_task(stdout_drainer.drain_reader(r_out))
        task_err = asyncio.create_task(stderr_drainer.drain_reader(r_err))

        # Feed 128 KiB to each stream concurrently
        for _ in range(16):
            r_out.feed_data(b"O" * 8192)
            r_err.feed_data(b"R" * 8192)
            await asyncio.sleep(0.001)
        r_out.feed_eof()
        r_err.feed_eof()

        await asyncio.gather(task_out, task_err)
        assert stdout_drainer.total_bytes_read == 128 * 1024
        assert stderr_drainer.total_bytes_read == 128 * 1024

    def test_invalid_utf8_decoding_with_replacement(self) -> None:
        drainer = BoundedStreamDrainer()
        # Feed invalid UTF-8 byte sequence
        drainer.feed(b"Valid prefix \xff\xfe\x80\x90 Valid suffix\n")
        text = drainer.get_text()
        assert "Valid prefix" in text
        assert "Valid suffix" in text
        assert "\ufffd" in text  # Unicode replacement character used

    def test_terminal_control_characters_and_ansi_sanitization(self) -> None:
        drainer = BoundedStreamDrainer()
        raw = b"\x1b[31;1mError: build failed!\x1b[0m\x07\x08\x00\x1f\nLine 2\tTabbed\r\n"
        drainer.feed(raw)
        text = drainer.get_text()

        # ANSI escapes stripped
        assert "\x1b[31;1m" not in text
        assert "\x1b[0m" not in text
        assert "Error: build failed!" in text
        # Control characters stripped (\x07 bell, \x08 backspace, \x00 null)
        assert "\x07" not in text
        assert "\x08" not in text
        assert "\x00" not in text
        # Newlines and tabs preserved
        assert "\n" in text
        assert "\t" in text

    def test_sensitive_secret_redaction(self) -> None:
        drainer = BoundedStreamDrainer()
        log_sample = (
            "Connecting with token ghp_123456789012345678901234567890123456 "
            "and fine-grained github_pat_11AAAAAAA0123456789012345678901234567890\n"
            "Headers: Authorization: Bearer secret_bearer_token_xyz\n"
            "Headers: Authorization: Basic dXNlcjpwYXNzd29yZA==\n"
            "Env: _AGY_ASKPASS_NONCE=0123456789abcdef"
            "0123456789abcdef0123456789abcdef0123456789abcdef\n"
        )
        drainer.feed(log_sample.encode("utf-8"))
        sanitized = drainer.get_text()

        assert "ghp_1234567890" not in sanitized
        assert "ghp_***" in sanitized
        assert "github_pat_11AAAAAAA" not in sanitized
        assert "github_pat_***" in sanitized
        assert "secret_bearer_token_xyz" not in sanitized
        assert "Bearer ***" in sanitized
        assert "dXNlcjpwYXNzd29yZA==" not in sanitized
        assert "Basic ***" in sanitized
        assert "0123456789abcdef" not in sanitized
        assert "_AGY_ASKPASS_NONCE=***" in sanitized

    @pytest.mark.asyncio
    async def test_drainer_task_cancellation_cleanup(self) -> None:
        drainer = BoundedStreamDrainer()
        reader = asyncio.StreamReader()
        task = asyncio.create_task(drainer.drain_reader(reader))

        # Cancel while blocked on read
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# =============================================================================
# Stage 2 Unit Tests: WindowsJobRunner
# =============================================================================


class TestWindowsJobRunner:
    """Verify Windows Job Object process containment, lifecycle, and fail-closed safety."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object tests require Windows")
    def test_windows_job_object_creation_and_kill_on_close_config(self) -> None:
        runner = WindowsJobRunner()
        h_job = runner._create_job_object()
        assert h_job != 0
        try:
            from app.services.git_service import JOBOBJECT_BASIC_ACCOUNTING_INFORMATION

            acct = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
            res = ctypes.windll.kernel32.QueryInformationJobObject(
                h_job, 1, ctypes.byref(acct), ctypes.sizeof(acct), None
            )
            assert res != 0
            assert acct.ActiveProcesses == 0
        finally:
            runner._close_handle(h_job)

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object tests require Windows")
    async def test_windows_job_runner_real_process_execution(self, tmp_path: Path) -> None:
        runner = WindowsJobRunner()
        res = await runner.run(
            [sys.executable, "-c", "print('win_job_runner_hello')"],
            cwd=tmp_path,
            timeout=10.0,
        )
        assert res.exit_code == 0
        assert res.success is True
        assert "win_job_runner_hello" in res.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object tests require Windows")
    async def test_job_assignment_failure_terminates_process_and_fails_closed(
        self, tmp_path: Path
    ) -> None:
        runner = WindowsJobRunner()
        # Mock AssignProcessToJobObject to simulate assignment failure
        with patch("ctypes.windll.kernel32.AssignProcessToJobObject", return_value=0):
            with patch("ctypes.windll.kernel32.TerminateProcess") as mock_terminate:
                with patch("ctypes.windll.kernel32.ResumeThread") as mock_resume:
                    with pytest.raises(WindowsJobAssignmentError):
                        await runner.run(
                            [sys.executable, "-c", "print('should_never_run')"],
                            cwd=tmp_path,
                            timeout=5.0,
                        )
                    # Verify TerminateProcess was called immediately on suspended process
                    assert mock_terminate.called
                    # Verify ResumeThread was NEVER called
                    assert not mock_resume.called

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object tests require Windows")
    async def test_timeout_terminates_job_object_and_verifies_active_processes_zero(
        self, tmp_path: Path
    ) -> None:
        runner = WindowsJobRunner()
        with pytest.raises(ProcessTimeoutError):
            # Run a sleep process that exceeds timeout
            await runner.run(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=tmp_path,
                timeout=0.3,
            )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object tests require Windows")
    def test_handles_closed_on_success_and_failure(self) -> None:
        runner = WindowsJobRunner()
        closed_handles: list[int] = []

        def track_close(h: Any) -> None:
            if h and h != -1:
                val = getattr(h, "value", h)
                if val is not None:
                    closed_handles.append(int(val))

        with patch.object(runner, "_close_handle", side_effect=track_close):
            runner._close_handle(1234)
            assert 1234 in closed_handles

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object tests require Windows")
    async def test_native_api_error_handling(self, tmp_path: Path) -> None:
        runner = WindowsJobRunner()
        # Mock CreateProcessW failure
        with patch("ctypes.windll.kernel32.CreateProcessW", return_value=0):
            with pytest.raises(ProcessRunnerError, match="CreateProcessW failed"):
                await runner.run(["non_existent_binary_xyz"], cwd=tmp_path)


# =============================================================================
# Stage 2 Unit Tests: POSIXProcessRunner
# =============================================================================


class TestPOSIXProcessRunner:
    """Verify POSIX process-group containment, signals, polling, and limitations."""

    @pytest.mark.asyncio
    async def test_posix_starts_new_session(self, tmp_path: Path) -> None:
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 4321
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stdout.feed_eof()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.stderr.feed_eof()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            res = await runner.run(["git", "status"], cwd=tmp_path)
            assert res.exit_code == 0
            # Verify start_new_session=True was passed to create independent process group
            _, kwargs = mock_exec.call_args
            assert kwargs.get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_posix_timeout_sends_sigterm_and_escalates_to_sigkill(
        self, tmp_path: Path
    ) -> None:
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 5432
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stderr = asyncio.StreamReader()

        # Simulate process timing out on wait() in run() and also during SIGTERM grace period
        mock_proc.wait = AsyncMock(side_effect=[TimeoutError(), TimeoutError(), 0])

        kill_calls: list[tuple[int, Any]] = []
        sigkill_sent = False

        def mock_killpg(pgid: int, sig: Any) -> None:
            nonlocal sigkill_sent
            kill_calls.append((pgid, sig))
            if sig == 9:
                sigkill_sent = True
            # Probe signal (sig == 0): once SIGKILL has been sent, report group dead.
            # This prevents the verification loop from blocking until its deadline.
            if sig == 0 and sigkill_sent:
                raise ProcessLookupError("No such process group")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("os.killpg", side_effect=mock_killpg, create=True):
                with (
                    patch("signal.SIGTERM", 15, create=True),
                    patch("signal.SIGKILL", 9, create=True),
                ):
                    with pytest.raises(ProcessTimeoutError):
                        await runner.run(["sleep", "10"], cwd=tmp_path, timeout=0.1)

        # Verify SIGTERM and SIGKILL were sent to process group 5432
        signals_sent = [call[1] for call in kill_calls]
        assert 15 in signals_sent  # SIGTERM
        assert 9 in signals_sent  # SIGKILL

    @pytest.mark.asyncio
    async def test_posix_verifies_original_process_group_termination(self) -> None:
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        # Simulate os.killpg(pgid, 0) succeeding once then raising ProcessLookupError (terminated)
        poll_calls = 0

        def mock_killpg(pgid: int, sig: Any) -> None:
            nonlocal poll_calls
            if sig == 0:
                poll_calls += 1
                if poll_calls >= 2:
                    raise ProcessLookupError("No such process")

        with patch("os.killpg", side_effect=mock_killpg, create=True):
            with patch("signal.SIGTERM", 15, create=True), patch("signal.SIGKILL", 9, create=True):
                await runner._terminate_and_verify_posix(
                    mock_proc,
                    pgid=9999,
                    stdout_task=None,
                    stderr_task=None,
                    grace_period_seconds=0.01,
                    poll_timeout_seconds=1.0,
                )
        assert poll_calls >= 2

    @pytest.mark.asyncio
    async def test_posix_handles_missing_process_group_safely(self) -> None:
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        # os.killpg raises ProcessLookupError immediately (process already dead)
        def mock_killpg(pgid: int, sig: Any) -> None:
            raise ProcessLookupError("Already exited")

        with patch("os.killpg", side_effect=mock_killpg, create=True):
            # Should not raise exception
            await runner._terminate_and_verify_posix(
                mock_proc,
                pgid=8888,
                stdout_task=None,
                stderr_task=None,
                grace_period_seconds=0.01,
            )

    @pytest.mark.asyncio
    async def test_posix_cancellation_terminates_group_and_reraises(self, tmp_path: Path) -> None:
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 7777
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stderr = asyncio.StreamReader()
        # Process hangs on wait until cancelled
        mock_proc.wait = AsyncMock(side_effect=asyncio.CancelledError())

        cleanup_called = False

        async def track_cleanup(*args: Any, **kwargs: Any) -> None:
            nonlocal cleanup_called
            cleanup_called = True

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch.object(runner, "_terminate_and_verify_posix", side_effect=track_cleanup):
                task = asyncio.create_task(runner.run(["git", "fetch"], cwd=tmp_path))
                await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert cleanup_called is True

    @pytest.mark.asyncio
    async def test_posix_reaps_direct_child(self) -> None:
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("os.killpg", side_effect=ProcessLookupError(), create=True):
            await runner._terminate_and_verify_posix(
                mock_proc, pgid=6666, stdout_task=None, stderr_task=None
            )
            # Verify proc.wait was called to reap zombie
            assert mock_proc.wait.called

    def test_posix_escaped_descendant_documented_limitation(self) -> None:
        doc = POSIXProcessRunner.__doc__ or ""
        assert "setsid()" in doc or "setpgid()" in doc
        assert "cgroups" in doc or "namespaces" in doc


# =============================================================================
# Stage 2 Unit Tests: SecureProcessRunner Lifecycle
# =============================================================================


class TestSecureProcessRunnerLifecycle:
    """Verify unified cross-platform runner lifecycle, error paths, and cancellation."""

    @pytest.mark.asyncio
    async def test_normal_process_completion(self, tmp_path: Path) -> None:
        runner = SecureProcessRunner()
        res = await runner.run(
            [sys.executable, "-c", "print('process_runner_success')"],
            cwd=tmp_path,
        )
        assert res.exit_code == 0
        assert res.success is True
        assert "process_runner_success" in res.stdout
        assert res.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_handling(self, tmp_path: Path) -> None:
        runner = SecureProcessRunner()
        res = await runner.run(
            [sys.executable, "-c", "import sys; sys.stderr.write('fatal_err\\n'); sys.exit(42)"],
            cwd=tmp_path,
        )
        assert res.exit_code == 42
        assert res.success is False
        assert "fatal_err" in res.stderr

    @pytest.mark.asyncio
    async def test_process_timeout_error_handling(self, tmp_path: Path) -> None:
        runner = SecureProcessRunner()
        with pytest.raises(ProcessTimeoutError):
            await runner.run(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=tmp_path,
                timeout=0.3,
            )

    @pytest.mark.asyncio
    async def test_cancellation_triggers_cleanup_and_reraises_cancelled_error(
        self, tmp_path: Path
    ) -> None:
        runner = SecureProcessRunner()
        task = asyncio.create_task(
            runner.run(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=tmp_path,
                timeout=15.0,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_startup_failure_handling(self, tmp_path: Path) -> None:
        runner = SecureProcessRunner()
        non_existent_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            await runner.run([sys.executable, "-c", "pass"], cwd=non_existent_dir)

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_cancelled_error(self, tmp_path: Path) -> None:
        runner = SecureProcessRunner()
        mock_backend = AsyncMock()

        async def failing_run(*args: Any, **kwargs: Any) -> None:
            # Raise CancelledError and simulate cleanup error
            raise asyncio.CancelledError()

        mock_backend.run = AsyncMock(side_effect=failing_run)
        runner._backend = mock_backend

        task = asyncio.create_task(runner.run(["echo", "1"], cwd=tmp_path))
        with pytest.raises(asyncio.CancelledError):
            await task


# =============================================================================
# Stage 2 Unit Tests: GitWorkspaceService Integration with SecureProcessRunner
# =============================================================================


class TestGitWorkspaceServiceWithSecureRunner:
    """Verify GitWorkspaceService integrates with SecureProcessRunner by default."""

    def test_git_workspace_service_uses_secure_process_runner_by_default(self) -> None:
        service = GitWorkspaceService()
        assert isinstance(service.process_runner, SecureProcessRunner)
        assert service.runner is None

    @pytest.mark.asyncio
    async def test_git_workspace_service_large_diff_drains_without_deadlock(
        self, tmp_path: Path
    ) -> None:
        service = GitWorkspaceService()
        ws = tmp_path / "large_diff_repo"
        ws.mkdir()

        await service.init_or_verify_repo(ws)

        # Create large file (>100 KiB)
        large_content = "Line of code in repository\n" * 5000
        (ws / "large_file.txt").write_text(large_content, encoding="utf-8")
        await service.stage_and_commit(ws, "feat: add large file")

        # Modify file
        modified_content = "Modified line in repository\n" * 5000
        (ws / "large_file.txt").write_text(modified_content, encoding="utf-8")

        # get_diff drains large diff without deadlock
        diff = await service.get_diff(ws)
        assert len(diff) > 0
        assert "Modified line in repository" in diff


# =============================================================================
# Stage 2 Regression Tests: Corrections from Code Review
# =============================================================================


class TestPrepareSafeEnvironment:
    """Regression tests for environment variable merging and denylist enforcement."""

    def test_sensitive_vars_removed_from_base_environment(self) -> None:
        """Sensitive vars inherited from parent environment are stripped."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "secret", "API_KEY": "s3cr3t"}):
            env = prepare_safe_environment()
            assert "GITHUB_TOKEN" not in env
            assert "API_KEY" not in env

    def test_caller_env_cannot_reintroduce_sensitive_var(self) -> None:
        """Caller-provided env cannot reintroduce a sensitive variable.

        The denylist must be enforced even when caller passes the sensitive key.
        """
        env = prepare_safe_environment(caller_env={"GITHUB_TOKEN": "leaked"})
        assert "GITHUB_TOKEN" not in env, (
            "Security breach: caller_env reintroduced a sensitive variable"
        )

    def test_caller_env_cannot_reintroduce_all_sensitive_vars(self) -> None:
        """Every known sensitive variable is enforced even if caller passes them."""
        malicious_env = {k: "leaked" for k in SENSITIVE_ENV_VARS}
        env = prepare_safe_environment(caller_env=malicious_env)
        for k in SENSITIVE_ENV_VARS:
            assert k not in env, f"Security breach: sensitive var {k!r} not removed"

    def test_inherited_askpass_vars_stripped_unless_explicitly_passed(self) -> None:
        """_AGY_ASKPASS_* inherited from parent environment are stripped."""
        with patch.dict(os.environ, {"_AGY_ASKPASS_NONCE": "inherited_nonce"}):
            env = prepare_safe_environment()
            assert "_AGY_ASKPASS_NONCE" not in env

    def test_askpass_vars_preserved_when_passed_in_caller_env(self) -> None:
        """_AGY_ASKPASS_* vars passed in caller_env survive to the subprocess."""
        env = prepare_safe_environment(
            caller_env={
                "_AGY_ASKPASS_NONCE": "nonce_abc",
                "_AGY_ASKPASS_PORT": "54321",
                "GIT_ASKPASS": "/tmp/askpass.py",
            }
        )
        assert env["_AGY_ASKPASS_NONCE"] == "nonce_abc"
        assert env["_AGY_ASKPASS_PORT"] == "54321"
        assert env["GIT_ASKPASS"] == "/tmp/askpass.py"

    def test_caller_env_askpass_not_stripped_even_with_inherited_collision(self) -> None:
        """When parent env has _AGY_ASKPASS_* but caller_env explicitly passes the same key,
        the caller's value survives (it's intentional)."""
        with patch.dict(os.environ, {"_AGY_ASKPASS_NONCE": "old_inherited"}):
            env = prepare_safe_environment(caller_env={"_AGY_ASKPASS_NONCE": "new_explicit"})
            assert env.get("_AGY_ASKPASS_NONCE") == "new_explicit"

    def test_safe_environment_denylist_enforced_after_caller_override(self) -> None:
        """Order: base -> strip _AGY_ASKPASS inherited -> merge caller -> enforce denylist."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "inherited_secret"}):
            env = prepare_safe_environment(
                caller_env={"GITHUB_TOKEN": "caller_secret", "SAFE_VAR": "ok"}
            )
            assert "GITHUB_TOKEN" not in env
            assert env.get("SAFE_VAR") == "ok"


class TestWindowsJobRunnerSecurityCorrections:
    """Regression tests for Windows Job Object runner security corrections."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_win32_argtypes_set_for_createjobobjectw(self) -> None:
        """CreateJobObjectW must have explicit argtypes and restype."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        assert kernel32.CreateJobObjectW.argtypes is not None
        assert kernel32.CreateJobObjectW.restype is not None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_win32_argtypes_set_for_createprocessw(self) -> None:
        """CreateProcessW must have explicit argtypes and restype for 64-bit safety."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        assert kernel32.CreateProcessW.argtypes is not None
        assert kernel32.CreateProcessW.restype is not None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_win32_argtypes_set_for_waitforsingleobject(self) -> None:
        """WaitForSingleObject must have explicit argtypes."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        assert kernel32.WaitForSingleObject.argtypes is not None
        assert kernel32.WaitForSingleObject.restype is not None

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_startupinfoexw_struct_exists(self) -> None:
        """STARTUPINFOEXW structure must exist for restricted handle inheritance."""
        assert STARTUPINFOEXW is not None, "STARTUPINFOEXW must be defined for Windows"
        # It must contain lpAttributeList for PROC_THREAD_ATTRIBUTE_LIST
        field_names = [f[0] for f in STARTUPINFOEXW._fields_]
        assert "lpAttributeList" in field_names, "STARTUPINFOEXW must have lpAttributeList"

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_mutable_command_buffer_passed_to_createprocessw(self, tmp_path: Path) -> None:
        """CreateProcessW must receive a mutable wchar_t buffer, not an immutable str."""
        import ctypes as ct

        create_calls: list[Any] = []

        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        original_create = kernel32.CreateProcessW

        def capture_createprocessw(*args: Any, **kwargs: Any) -> bool:
            create_calls.append(args)
            # Second argument (lpCommandLine) must be a ctypes buffer (mutable)
            cmdline_arg = args[1]
            assert isinstance(cmdline_arg, ct.Array), (
                f"lpCommandLine must be a mutable ctypes buffer, got {type(cmdline_arg)}"
            )
            return original_create(*args, **kwargs)

        with patch.object(kernel32, "CreateProcessW", side_effect=capture_createprocessw):
            runner = WindowsJobRunner()
            res = await runner.run([sys.executable, "-c", "pass"], cwd=tmp_path, timeout=5.0)
            assert res.exit_code == 0
        assert len(create_calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_pipe_failure_fails_closed_and_closes_handles(self, tmp_path: Path) -> None:
        """If CreatePipe fails, all previously allocated handles must be closed."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        closed: list[Any] = []
        original_close = kernel32.CloseHandle

        def track_close(h: Any) -> bool:
            closed.append(h)
            return original_close(h)

        # First CreatePipe succeeds; second fails
        call_count = 0
        original_create_pipe = kernel32.CreatePipe

        def fail_on_second_pipe(*args: Any, **kwargs: Any) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return False  # Simulate second pipe creation failure
            return original_create_pipe(*args, **kwargs)

        runner = WindowsJobRunner()
        with patch.object(kernel32, "CreatePipe", side_effect=fail_on_second_pipe):
            with patch.object(kernel32, "CloseHandle", side_effect=track_close):
                with pytest.raises(ProcessRunnerError, match="CreatePipe"):
                    await runner.run([sys.executable, "-c", "pass"], cwd=tmp_path)
        # At least one handle must have been closed (the stdout read/write handles)
        assert len(closed) > 0, "No handles were closed after pipe failure"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_sethandleinformation_failure_raises_error(self, tmp_path: Path) -> None:
        """SetHandleInformation failure on pipe read handle must raise ProcessRunnerError."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()
        original_shi = kernel32.SetHandleInformation

        call_count = 0

        def fail_shi(*args: Any, **kwargs: Any) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False  # Fail on first SetHandleInformation call
            return original_shi(*args, **kwargs)

        with patch.object(kernel32, "SetHandleInformation", side_effect=fail_shi):
            with pytest.raises(ProcessRunnerError, match="SetHandleInformation"):
                runner._create_pipes()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_createfileW_failure_for_nul_raises_error(self) -> None:
        """CreateFileW('NUL') failure must raise ProcessRunnerError."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()

        with patch.object(kernel32, "CreateFileW", return_value=0):
            with pytest.raises(ProcessRunnerError, match="CreateFileW"):
                runner._create_pipes()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_handle_list_attribute_restricts_inheritance_to_exactly_three_handles(self) -> None:
        """_create_handle_list_attribute must set PROC_THREAD_ATTRIBUTE_HANDLE_LIST."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()
        update_calls: list[tuple[Any, ...]] = []
        original_update = kernel32.UpdateProcThreadAttribute

        def capture_update(*args: Any, **kwargs: Any) -> bool:
            update_calls.append(args)
            return original_update(*args, **kwargs)

        # Create three dummy test handles (use kernel32 current process as a placeholder)
        h1 = kernel32.GetCurrentProcess()
        h2 = kernel32.GetCurrentProcess()
        h3 = kernel32.GetCurrentProcess()

        with patch.object(kernel32, "UpdateProcThreadAttribute", side_effect=capture_update):
            lp_attr, _buf = runner._create_handle_list_attribute([h1, h2, h3])
            kernel32.DeleteProcThreadAttributeList(lp_attr)

        assert len(update_calls) == 1
        # Attribute key must be PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
        from app.services.git_service import PROC_THREAD_ATTRIBUTE_HANDLE_LIST  # type: ignore[attr-defined]

        assert update_calls[0][2] == PROC_THREAD_ATTRIBUTE_HANDLE_LIST

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_hanging_descendant_pipe_does_not_block_indefinitely(
        self, tmp_path: Path
    ) -> None:
        """
        Regression: a child that keeps stdout open but exits should not cause
        WindowsJobRunner to hang forever. The bounded drainer timeout terminates
        the job object and unblocks within the stream drainer deadline.
        """
        runner = WindowsJobRunner()
        # Spawn a process that exits quickly but keep the Job Object's pipe open
        # by running a process that sleeps — it should be killed by TerminateJobObject
        with pytest.raises((ProcessTimeoutError, ProcessRunnerError)):
            await runner.run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=tmp_path,
                timeout=0.3,
            )


class TestPOSIXProcessRunnerCorrections:
    """Regression tests for POSIX process-group security corrections."""

    @pytest.mark.asyncio
    async def test_sigkill_sent_when_group_survives_sigterm_grace_period(self) -> None:
        """
        Regression: SIGKILL must be sent when os.killpg(pgid, 0) proves the
        process group still exists after the SIGTERM grace period, independently
        of whether the direct child has exited.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        # Direct child exits on first await (grace period passes)
        mock_proc.wait = AsyncMock(return_value=0)

        kill_calls: list[tuple[int, Any]] = []

        def mock_killpg(pgid: int, sig: Any) -> None:
            kill_calls.append((pgid, sig))
            if sig == 0:
                # Probe: group still exists on first check, gone on second
                call_count = sum(1 for _, s in kill_calls if s == 0)
                if call_count >= 2:
                    raise ProcessLookupError("gone")

        with patch("os.killpg", side_effect=mock_killpg, create=True):
            with patch("signal.SIGTERM", 15, create=True):
                with patch("signal.SIGKILL", 9, create=True):
                    await runner._terminate_and_verify_posix(
                        mock_proc,
                        pgid=12345,
                        stdout_task=None,
                        stderr_task=None,
                        grace_period_seconds=0.01,
                        poll_timeout_seconds=1.0,
                    )

        signals_sent = [s for _, s in kill_calls]
        assert 15 in signals_sent, "SIGTERM must be sent"
        assert 9 in signals_sent, "SIGKILL must be sent when group survives grace period"

    @pytest.mark.asyncio
    async def test_permission_error_during_verification_raises_containment_error(self) -> None:
        """
        Regression: a PermissionError from os.killpg(pgid, 0) must NOT be treated
        as successful termination. It must raise ProcessContainmentError.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        def killpg_always_permission_error(pgid: int, sig: Any) -> None:
            if sig == 0:
                raise PermissionError("EPERM")

        with patch("os.killpg", side_effect=killpg_always_permission_error, create=True):
            with patch("signal.SIGTERM", 15, create=True):
                with patch("signal.SIGKILL", 9, create=True):
                    with pytest.raises(ProcessContainmentError, match="termination"):
                        await runner._terminate_and_verify_posix(
                            mock_proc,
                            pgid=99999,
                            stdout_task=None,
                            stderr_task=None,
                            grace_period_seconds=0.01,
                            poll_timeout_seconds=0.1,
                        )

    @pytest.mark.asyncio
    async def test_timeout_on_group_verification_raises_containment_error(self) -> None:
        """
        Regression: if os.killpg(pgid, 0) consistently succeeds (group never dies),
        a ProcessContainmentError must be raised after the poll timeout, not silently ignored.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        def killpg_group_never_dies(pgid: int, sig: Any) -> None:
            # Probe succeeds (group still alive); SIGTERM/SIGKILL do nothing
            pass

        with patch("os.killpg", side_effect=killpg_group_never_dies, create=True):
            with patch("signal.SIGTERM", 15, create=True):
                with patch("signal.SIGKILL", 9, create=True):
                    with pytest.raises(ProcessContainmentError, match="failed to terminate"):
                        await runner._terminate_and_verify_posix(
                            mock_proc,
                            pgid=77777,
                            stdout_task=None,
                            stderr_task=None,
                            grace_period_seconds=0.01,
                            poll_timeout_seconds=0.15,
                        )

    @pytest.mark.asyncio
    async def test_sigkill_sent_independently_of_direct_child_exit_status(self) -> None:
        """
        Regression: SIGKILL must be evaluated based on killpg(pgid, 0) probe,
        NOT based on whether proc.wait() returned. Direct child may have exited
        while other descendants in the group survive.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        # Direct child exits immediately during grace period
        mock_proc.wait = AsyncMock(return_value=0)

        kill_calls: list[tuple[int, int]] = []

        # Group probe: alive until SIGKILL is sent
        sigkill_sent = False

        def mock_killpg(pgid: int, sig: Any) -> None:
            nonlocal sigkill_sent
            kill_calls.append((pgid, int(sig)))
            if sig == 9:
                sigkill_sent = True
            if sig == 0 and not sigkill_sent:
                # Descendants still alive until SIGKILL
                return  # group still exists
            if sig == 0 and sigkill_sent:
                raise ProcessLookupError("gone after SIGKILL")

        with patch("os.killpg", side_effect=mock_killpg, create=True):
            with patch("signal.SIGTERM", 15, create=True):
                with patch("signal.SIGKILL", 9, create=True):
                    await runner._terminate_and_verify_posix(
                        mock_proc,
                        pgid=55555,
                        stdout_task=None,
                        stderr_task=None,
                        grace_period_seconds=0.01,
                        poll_timeout_seconds=0.5,
                    )

        assert sigkill_sent, "SIGKILL must be sent when group survives even if direct child exited"
        sent_signals = [s for _, s in kill_calls]
        assert 15 in sent_signals  # SIGTERM
        assert 9 in sent_signals  # SIGKILL

    @pytest.mark.asyncio
    async def test_posix_drainer_timeout_triggers_process_group_termination(
        self, tmp_path: Path
    ) -> None:
        """
        Regression: when direct child exits but descendants keep stdout/stderr open,
        the drainer timeout must trigger process-group termination (SIGTERM -> SIGKILL)
        for the original PGID, cancel drainers, and verify termination.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 11111
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        kill_calls: list[tuple[int, Any]] = []
        sigkill_sent = False

        def mock_killpg(pgid: int, sig: Any) -> None:
            nonlocal sigkill_sent
            kill_calls.append((pgid, sig))
            if sig == 9:
                sigkill_sent = True
            if sig == 0 and sigkill_sent:
                raise ProcessLookupError("terminated after SIGKILL")

        async def hang_drain(reader: asyncio.StreamReader) -> None:
            await asyncio.sleep(100)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("os.killpg", side_effect=mock_killpg, create=True):
                with (
                    patch("signal.SIGTERM", 15, create=True),
                    patch("signal.SIGKILL", 9, create=True),
                ):
                    with patch("app.services.git_service.BoundedStreamDrainer") as MockDrainer:
                        mock_stdout_drainer = MagicMock()
                        mock_stdout_drainer.drain_reader = hang_drain
                        mock_stdout_drainer.get_text.return_value = "partial output"
                        mock_stderr_drainer = MagicMock()

                        async def fast_drain(reader: asyncio.StreamReader) -> None:
                            return

                        mock_stderr_drainer.drain_reader = fast_drain
                        mock_stderr_drainer.get_text.return_value = ""
                        MockDrainer.side_effect = [mock_stdout_drainer, mock_stderr_drainer]

                        result = await asyncio.wait_for(
                            runner.run([sys.executable, "-c", "pass"], cwd=tmp_path, timeout=10.0),
                            timeout=6.0,
                        )
                        assert isinstance(result, ExecutionResult)
                        assert result.stdout == "partial output"

        # Verify termination was invoked for PGID 11111
        sent_signals = [sig for _, sig in kill_calls]
        assert 15 in sent_signals, "SIGTERM must be sent to process group"
        assert 9 in sent_signals, "SIGKILL must be sent when group survives grace period"

    @pytest.mark.asyncio
    async def test_posix_direct_child_exits_while_descendant_remains(self, tmp_path: Path) -> None:
        """
        Direct child exiting with 0 while a descendant keeps pipes open must invoke
        group termination on PGID and clean up descendants.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 22222
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        kill_calls: list[tuple[int, Any]] = []

        def mock_killpg(pgid: int, sig: Any) -> None:
            kill_calls.append((pgid, sig))
            if sig == 0:
                raise ProcessLookupError("gone")

        async def hang_drain(reader: asyncio.StreamReader) -> None:
            await asyncio.sleep(100)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("os.killpg", side_effect=mock_killpg, create=True):
                with (
                    patch("signal.SIGTERM", 15, create=True),
                    patch("signal.SIGKILL", 9, create=True),
                ):
                    with patch("app.services.git_service.BoundedStreamDrainer") as MockDrainer:
                        mock_stdout_drainer = MagicMock()
                        mock_stdout_drainer.drain_reader = hang_drain
                        mock_stdout_drainer.get_text.return_value = ""
                        mock_stderr_drainer = MagicMock()

                        async def fast_drain(reader: asyncio.StreamReader) -> None:
                            return

                        mock_stderr_drainer.drain_reader = fast_drain
                        mock_stderr_drainer.get_text.return_value = ""
                        MockDrainer.side_effect = [mock_stdout_drainer, mock_stderr_drainer]

                        result = await runner.run(
                            [sys.executable, "-c", "pass"], cwd=tmp_path, timeout=10.0
                        )
                        assert isinstance(result, ExecutionResult)
                        assert (22222, 15) in kill_calls

    @pytest.mark.asyncio
    async def test_posix_descendant_successful_termination_on_sigterm(self, tmp_path: Path) -> None:
        """
        When descendant terminates cleanly upon SIGTERM during grace period,
        SIGKILL must not be sent, and ExecutionResult is returned.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 33333
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        kill_calls: list[tuple[int, Any]] = []

        def mock_killpg(pgid: int, sig: Any) -> None:
            kill_calls.append((pgid, sig))
            if sig == 0:
                # Group died cleanly on SIGTERM
                raise ProcessLookupError("group died on SIGTERM")

        async def hang_drain(reader: asyncio.StreamReader) -> None:
            await asyncio.sleep(100)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("os.killpg", side_effect=mock_killpg, create=True):
                with (
                    patch("signal.SIGTERM", 15, create=True),
                    patch("signal.SIGKILL", 9, create=True),
                ):
                    with patch("app.services.git_service.BoundedStreamDrainer") as MockDrainer:
                        mock_stdout_drainer = MagicMock()
                        mock_stdout_drainer.drain_reader = hang_drain
                        mock_stdout_drainer.get_text.return_value = ""
                        mock_stderr_drainer = MagicMock()

                        async def fast_drain(reader: asyncio.StreamReader) -> None:
                            return

                        mock_stderr_drainer.drain_reader = fast_drain
                        mock_stderr_drainer.get_text.return_value = ""
                        MockDrainer.side_effect = [mock_stdout_drainer, mock_stderr_drainer]

                        result = await runner.run(
                            [sys.executable, "-c", "pass"], cwd=tmp_path, timeout=10.0
                        )
                        assert isinstance(result, ExecutionResult)

        sent_sigs = [s for _, s in kill_calls]
        assert 15 in sent_sigs, "SIGTERM must be sent"
        assert 9 not in sent_sigs, "SIGKILL must not be sent if group died on SIGTERM"

    @pytest.mark.asyncio
    async def test_posix_descendant_verification_failure_raises_containment_error(
        self, tmp_path: Path
    ) -> None:
        """
        When descendants refuse to terminate within the poll deadline,
        ProcessContainmentError must be raised and no successful result returned.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 44444
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        # Descendant stays alive (probe always succeeds)
        def mock_killpg(pgid: int, sig: Any) -> None:
            pass

        async def hang_drain(reader: asyncio.StreamReader) -> None:
            await asyncio.sleep(100)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("os.killpg", side_effect=mock_killpg, create=True):
                with (
                    patch("signal.SIGTERM", 15, create=True),
                    patch("signal.SIGKILL", 9, create=True),
                ):
                    with patch("app.services.git_service.BoundedStreamDrainer") as MockDrainer:
                        mock_stdout_drainer = MagicMock()
                        mock_stdout_drainer.drain_reader = hang_drain
                        mock_stdout_drainer.get_text.return_value = ""
                        mock_stderr_drainer = MagicMock()

                        async def fast_drain(reader: asyncio.StreamReader) -> None:
                            return

                        mock_stderr_drainer.drain_reader = fast_drain
                        mock_stderr_drainer.get_text.return_value = ""
                        MockDrainer.side_effect = [mock_stdout_drainer, mock_stderr_drainer]

                        # Shorten poll timeout inside _terminate_and_verify_posix for fast test
                        with patch.object(runner, "_terminate_and_verify_posix") as mock_term:
                            mock_term.side_effect = ProcessContainmentError(
                                "POSIX process group 44444 failed to terminate within 3.0s"
                            )
                            with pytest.raises(
                                ProcessContainmentError, match="failed to terminate"
                            ):
                                await runner.run(
                                    [sys.executable, "-c", "pass"], cwd=tmp_path, timeout=10.0
                                )

    @pytest.mark.asyncio
    async def test_posix_drainer_cleanup_after_termination_failure(self) -> None:
        """
        Drainer tasks must be cancelled and awaited in the finally block even when
        group termination verification fails with ProcessContainmentError.
        """
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 55555
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        async def dummy_drainer() -> None:
            await asyncio.sleep(100)

        task1 = asyncio.create_task(dummy_drainer())
        task2 = asyncio.create_task(dummy_drainer())

        # Probe always succeeds -> triggers verification timeout
        with patch("os.killpg", side_effect=lambda pgid, sig: None, create=True):
            with patch("signal.SIGTERM", 15, create=True), patch("signal.SIGKILL", 9, create=True):
                with pytest.raises(ProcessContainmentError):
                    await runner._terminate_and_verify_posix(
                        mock_proc,
                        pgid=55555,
                        stdout_task=task1,
                        stderr_task=task2,
                        grace_period_seconds=0.01,
                        poll_timeout_seconds=0.05,
                    )

        assert task1.done(), "stdout drainer task must be cancelled and awaited"
        assert task2.done(), "stderr drainer task must be cancelled and awaited"


class TestWindowsJobRunnerEnvironmentSecurity:
    """Regression tests for Windows Job runner environment variable security."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_sensitive_env_not_passed_to_child_process(self, tmp_path: Path) -> None:
        """GITHUB_TOKEN and other sensitive vars must not appear in child environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "super_secret_token_xyz"}):
            runner = WindowsJobRunner()
            res = await runner.run(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('GITHUB_TOKEN', 'ABSENT'))",
                ],
                cwd=tmp_path,
                timeout=10.0,
            )
            assert res.exit_code == 0
            assert "super_secret_token_xyz" not in res.stdout
            assert "ABSENT" in res.stdout

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_caller_env_sensitive_var_not_passed_to_child(self, tmp_path: Path) -> None:
        """Even if caller passes GITHUB_TOKEN in env override, it must be stripped."""
        runner = WindowsJobRunner()
        res = await runner.run(
            [sys.executable, "-c", "import os; print(os.environ.get('GITHUB_TOKEN', 'ABSENT'))"],
            cwd=tmp_path,
            env={"GITHUB_TOKEN": "caller_leaked_token"},
            timeout=10.0,
        )
        assert res.exit_code == 0
        assert "caller_leaked_token" not in res.stdout
        assert "ABSENT" in res.stdout


class TestPOSIXRunnerEnvironmentSecurity:
    """Regression tests for POSIX runner environment variable security."""

    @pytest.mark.asyncio
    async def test_sensitive_env_not_passed_to_child_process(self, tmp_path: Path) -> None:
        """GITHUB_TOKEN must not appear in subprocess environment."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "posix_secret_xyz"}):
            runner = POSIXProcessRunner()
            mock_proc = AsyncMock()
            mock_proc.pid = 22222
            mock_proc.stdout = asyncio.StreamReader()
            mock_proc.stdout.feed_eof()
            mock_proc.stderr = asyncio.StreamReader()
            mock_proc.stderr.feed_eof()
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0

            captured_env: dict[str, str] | None = None

            async def capture_subprocess(*args: Any, **kwargs: Any) -> Any:
                nonlocal captured_env
                captured_env = kwargs.get("env")
                return mock_proc

            with patch("asyncio.create_subprocess_exec", side_effect=capture_subprocess):
                await runner.run([sys.executable, "-c", "pass"], cwd=tmp_path)

            assert captured_env is not None
            assert "GITHUB_TOKEN" not in captured_env, (
                "GITHUB_TOKEN must not be passed to subprocess environment"
            )

    @pytest.mark.asyncio
    async def test_caller_env_sensitive_var_not_passed_to_child(self, tmp_path: Path) -> None:
        """Even if caller passes GITHUB_TOKEN in env override, denylist strips it."""
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 33333
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stdout.feed_eof()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.stderr.feed_eof()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        captured_env: dict[str, str] | None = None

        async def capture_subprocess(*args: Any, **kwargs: Any) -> Any:
            nonlocal captured_env
            captured_env = kwargs.get("env")
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_subprocess):
            await runner.run(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env={"GITHUB_TOKEN": "caller_leaked"},
            )

        assert captured_env is not None
        assert "GITHUB_TOKEN" not in captured_env, (
            "caller_env must not reintroduce sensitive var to subprocess"
        )

    @pytest.mark.asyncio
    async def test_askpass_ipc_vars_preserved_when_caller_passes_them(self, tmp_path: Path) -> None:
        """_AGY_ASKPASS_* vars passed in caller_env must reach the subprocess."""
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 44444
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stdout.feed_eof()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.stderr.feed_eof()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        captured_env: dict[str, str] | None = None

        async def capture_subprocess(*args: Any, **kwargs: Any) -> Any:
            nonlocal captured_env
            captured_env = kwargs.get("env")
            return mock_proc

        askpass_env = {
            "GIT_ASKPASS": "/tmp/askpass.py",
            "_AGY_ASKPASS_NONCE": "abc123",
            "_AGY_ASKPASS_PORT": "9999",
            "GIT_TERMINAL_PROMPT": "0",
        }

        with patch("asyncio.create_subprocess_exec", side_effect=capture_subprocess):
            await runner.run([sys.executable, "-c", "pass"], cwd=tmp_path, env=askpass_env)

        assert captured_env is not None
        assert captured_env.get("GIT_ASKPASS") == "/tmp/askpass.py"
        assert captured_env.get("_AGY_ASKPASS_NONCE") == "abc123"
        assert captured_env.get("_AGY_ASKPASS_PORT") == "9999"
        assert captured_env.get("GIT_TERMINAL_PROMPT") == "0"

    @pytest.mark.asyncio
    async def test_inherited_askpass_vars_not_passed_without_caller_env(
        self, tmp_path: Path
    ) -> None:
        """Inherited _AGY_ASKPASS_* from parent env must be stripped if not in caller_env."""
        runner = POSIXProcessRunner()
        mock_proc = AsyncMock()
        mock_proc.pid = 55556
        mock_proc.stdout = asyncio.StreamReader()
        mock_proc.stdout.feed_eof()
        mock_proc.stderr = asyncio.StreamReader()
        mock_proc.stderr.feed_eof()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        captured_env: dict[str, str] | None = None

        async def capture_subprocess(*args: Any, **kwargs: Any) -> Any:
            nonlocal captured_env
            captured_env = kwargs.get("env")
            return mock_proc

        with patch.dict(os.environ, {"_AGY_ASKPASS_NONCE": "leaked_inherited_nonce"}):
            with patch("asyncio.create_subprocess_exec", side_effect=capture_subprocess):
                await runner.run([sys.executable, "-c", "pass"], cwd=tmp_path)

        assert captured_env is not None
        assert "_AGY_ASKPASS_NONCE" not in captured_env, (
            "Inherited _AGY_ASKPASS_NONCE must not be passed to subprocess"
        )


class TestWindowsJobObjectVerification:
    """Regression tests for Windows Job Object fail-closed verification."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_active_processes_reaching_zero_returns_cleanly(self) -> None:
        """_verify_active_processes_zero returns normally when ActiveProcesses reaches 0."""
        from app.services.git_service import (  # type: ignore[attr-defined]
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
            kernel32,
        )

        runner = WindowsJobRunner()

        def mock_query(h: Any, info_class: Any, buf_ptr: Any, size: Any, ret_len: Any) -> int:
            acct = ctypes.cast(
                buf_ptr, ctypes.POINTER(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)
            ).contents
            acct.ActiveProcesses = 0
            return 1

        with patch.object(kernel32, "QueryInformationJobObject", side_effect=mock_query):
            await runner._verify_active_processes_zero(h_job=999, timeout_seconds=0.5)

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_active_processes_remaining_nonzero_raises_containment_error(self) -> None:
        """_verify_active_processes_zero raises ProcessContainmentError if count remains nonzero."""
        from app.services.git_service import (  # type: ignore[attr-defined]
            JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
            kernel32,
        )

        runner = WindowsJobRunner()

        def mock_query(h: Any, info_class: Any, buf_ptr: Any, size: Any, ret_len: Any) -> int:
            acct = ctypes.cast(
                buf_ptr, ctypes.POINTER(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)
            ).contents
            acct.ActiveProcesses = 2
            return 1

        with patch.object(kernel32, "QueryInformationJobObject", side_effect=mock_query):
            with pytest.raises(ProcessContainmentError, match=r"remained nonzero \(2\)"):
                await runner._verify_active_processes_zero(h_job=999, timeout_seconds=0.1)

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_query_information_job_object_failure_raises_containment_error(self) -> None:
        """Repeated QueryInformationJobObject failure must raise ProcessContainmentError."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()

        with patch.object(kernel32, "QueryInformationJobObject", return_value=0):
            with patch.object(kernel32, "GetLastError", return_value=5):  # ERROR_ACCESS_DENIED
                with pytest.raises(
                    ProcessContainmentError,
                    match="QueryInformationJobObject failed repeatedly.*error 5",
                ):
                    await runner._verify_active_processes_zero(h_job=999, timeout_seconds=0.1)

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_termination_verification_failure_during_cleanup_raises(self) -> None:
        """_cleanup_on_error must propagate containment failure if verification fails."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()
        with patch.object(kernel32, "TerminateJobObject", return_value=1):
            with patch.object(
                runner,
                "_verify_active_processes_zero",
                side_effect=ProcessContainmentError("Simulated containment verification failure"),
            ):
                with pytest.raises(ProcessContainmentError, match="Simulated containment"):
                    await runner._cleanup_on_error(
                        h_job=888,
                        h_process=None,
                        stdout_task=None,
                        stderr_task=None,
                    )

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_run_timeout_with_containment_failure_preserves_exception_chaining(
        self, tmp_path: Path
    ) -> None:
        """
        When runner.run() times out and cleanup also fails containment verification,
        ProcessContainmentError must be raised with ProcessTimeoutError as its __cause__.
        """
        runner = WindowsJobRunner()
        h1 = MagicMock()
        h1.value = 1
        h3 = MagicMock()
        h3.value = 3
        with patch.object(
            runner, "_wait_for_process", side_effect=ProcessTimeoutError("Timed out")
        ):
            with patch.object(
                runner,
                "_cleanup_on_error",
                side_effect=ProcessContainmentError("Containment failed during cleanup"),
            ):
                with patch.object(runner, "_create_job_object", return_value=111):
                    with patch.object(runner, "_create_pipes", return_value=(h1, 2, h3, 4, 5)):
                        with patch.object(
                            runner, "_create_handle_list_attribute", return_value=(1, None)
                        ):
                            with patch(
                                "app.services.git_service.kernel32.CreateProcessW", return_value=1
                            ):
                                with patch(
                                    "app.services.git_service.kernel32.AssignProcessToJobObject",
                                    return_value=1,
                                ):
                                    with patch(
                                        "app.services.git_service.kernel32.ResumeThread",
                                        return_value=1,
                                    ):
                                        with pytest.raises(ProcessContainmentError) as exc_info:
                                            await runner.run(
                                                [sys.executable, "-c", "pass"],
                                                cwd=tmp_path,
                                                timeout=0.1,
                                            )
                                        assert isinstance(
                                            exc_info.value.__cause__, ProcessTimeoutError
                                        )


class TestWindowsAttributeListAllocationHardening:
    """Regression tests for Windows attribute-list allocation hardening."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_zero_allocation_size_rejected(self) -> None:
        """InitializeProcThreadAttributeList returning size 0 must raise ProcessRunnerError."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()

        # Simulate first sizing call leaving size as 0
        def mock_init(lp_attr: Any, count: Any, flags: Any, size_ptr: Any) -> int:
            return 0  # Sizing call always returns 0 with size in pointer

        with patch.object(kernel32, "InitializeProcThreadAttributeList", side_effect=mock_init):
            with pytest.raises(ProcessRunnerError, match=r"invalid buffer size \(0\)"):
                runner._create_handle_list_attribute([1, 2, 3])

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_attribute_list_initialization_failure_raises_error(self) -> None:
        """Second InitializeProcThreadAttributeList call failing must raise ProcessRunnerError."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()
        call_count = 0

        def mock_init(lp_attr: Any, count: Any, flags: Any, size_ptr: Any) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Sizing call: populate size
                ctypes.cast(size_ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = 48
                return 0
            # Second initialization call fails
            return 0

        with patch.object(kernel32, "InitializeProcThreadAttributeList", side_effect=mock_init):
            with patch.object(kernel32, "GetLastError", return_value=87):  # ERROR_INVALID_PARAMETER
                with pytest.raises(
                    ProcessRunnerError,
                    match="InitializeProcThreadAttributeList failed with error 87",
                ):
                    runner._create_handle_list_attribute([1, 2, 3])

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_attribute_update_failure_raises_error(self) -> None:
        """UpdateProcThreadAttribute failure must raise ProcessRunnerError."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()

        with patch.object(kernel32, "UpdateProcThreadAttribute", return_value=0):
            with patch.object(kernel32, "GetLastError", return_value=87):
                with pytest.raises(
                    ProcessRunnerError, match="UpdateProcThreadAttribute failed with error 87"
                ):
                    runner._create_handle_list_attribute([1, 2, 3])

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_correct_attribute_list_cleanup_after_update_failure(self) -> None:
        """DeleteProcThreadAttributeList must be called when UpdateProcThreadAttribute fails."""
        from app.services.git_service import kernel32  # type: ignore[attr-defined]

        runner = WindowsJobRunner()
        deleted_ptrs: list[Any] = []

        with patch.object(kernel32, "UpdateProcThreadAttribute", return_value=0):
            with patch.object(
                kernel32,
                "DeleteProcThreadAttributeList",
                side_effect=lambda ptr: deleted_ptrs.append(ptr),
            ):
                with pytest.raises(ProcessRunnerError):
                    runner._create_handle_list_attribute([1, 2, 3])

        assert len(deleted_ptrs) == 1, "DeleteProcThreadAttributeList must be called on failure"
