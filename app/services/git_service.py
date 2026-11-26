"""
app/services/git_service.py
---------------------------
Secure cross-platform process runner and isolated workspace Git operations.

Responsibilities:
- WindowsJobRunner: Windows Job Object process containment using native Win32 APIs
  via ctypes with CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread,
  fail-closed assignment, TerminateJobObject, and ActiveProcesses == 0 verification.
- POSIXProcessRunner: POSIX process-group management using start_new_session=True,
  SIGTERM -> grace period -> SIGKILL escalation, original process group verification
  via os.killpg(pgid, 0), and child reaping.
- BoundedStreamDrainer: Asynchronous, bounded stdout/stderr draining (64 KiB total retention
  with 32 KiB head and 32 KiB tail), continuous draining to prevent pipe deadlock, safe
  UTF-8 decoding, ANSI/control character sanitization, and secret redaction.
- SecureProcessRunner: Cross-platform runner coordinating containment, bounded draining,
  timeouts, and cancellation resilience with shielded cleanup.
- GitWorkspaceService: Manages Git version control actions inside an isolated job workspace
  with repository hook bypass and secure execution boundaries.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from app.execution.runner import ExecutionResult as ExecutionResult
from app.execution.runner import SandboxRunner

logger = logging.getLogger(__name__)

# Sensitive environment variables stripped from process execution environment
SENSITIVE_ENV_VARS: frozenset[str] = frozenset(
    {
        "GITHUB_TOKEN",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "SECRET_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "API_KEY",
    }
)

# Explicitly allowed Askpass IPC variables
ASKPASS_IPC_ENV_VARS: frozenset[str] = frozenset(
    {
        "GIT_ASKPASS",
        "GIT_TERMINAL_PROMPT",
        "_AGY_ASKPASS_NONCE",
        "_AGY_ASKPASS_SOCK",
        "_AGY_ASKPASS_PORT",
    }
)


def prepare_safe_environment(caller_env: dict[str, str] | None = None) -> dict[str, str]:
    """
    Construct a secure environment dictionary for subprocess execution:
    1. Start with copy of parent environment.
    2. Strip inherited _AGY_ASKPASS_* unless explicitly passed in caller_env.
    3. Merge caller-provided environment overrides.
    4. Strictly enforce the sensitive-variable denylist (preventing re-introduction).
    """
    exec_env = os.environ.copy()

    # Strip inherited _AGY_ASKPASS_* unless explicitly passed in caller_env
    for k in list(exec_env.keys()):
        if k.startswith("_AGY_ASKPASS_") and (not caller_env or k not in caller_env):
            exec_env.pop(k, None)

    # Merge caller-provided overrides
    if caller_env:
        exec_env.update(caller_env)

    # Strictly enforce denylist after caller-provided overrides
    for k in SENSITIVE_ENV_VARS:
        exec_env.pop(k, None)

    return exec_env


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class GitServiceError(Exception):
    """Base exception for workspace git operations."""


class ProcessRunnerError(GitServiceError):
    """Base exception for process runner failures."""


class ProcessTimeoutError(ProcessRunnerError):
    """Raised when a process exceeds the configured timeout."""


class WindowsJobAssignmentError(ProcessRunnerError):
    """Raised when assigning a process to a Windows Job Object fails."""


class ProcessContainmentError(ProcessRunnerError):
    """Raised when process containment fails or cannot be verified."""


# =============================================================================
# Bounded Stream Drainer
# =============================================================================


class BoundedStreamDrainer:
    """
    Accumulates and drains an output stream with bounded memory retention.

    Guarantees:
    - Retains at most 64 KiB total output per stream (32 KiB head + 32 KiB tail).
    - Preserves beginning and ending output with a truncation notice if truncated.
    - Continuously reads until EOF regardless of output size to prevent pipe deadlock.
    - Decodes text safely with replacement characters for invalid UTF-8.
    - Sanitizes ANSI escape sequences and non-printable control characters.
    - Redacts credentials, PATs, bearer tokens, and Askpass nonces.
    """

    MAX_RETENTION_BYTES: int = 64 * 1024  # 64 KiB
    HEAD_RETENTION_BYTES: int = 32 * 1024  # 32 KiB
    TAIL_RETENTION_BYTES: int = 32 * 1024  # 32 KiB

    _ANSI_ESCAPE_RE: re.Pattern[str] = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    _CONTROL_CHAR_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"ghp_[A-Za-z0-9_]{36,}"), "ghp_***"),
        (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), "github_pat_***"),
        (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/-]+=*", re.IGNORECASE), r"\1***"),
        (re.compile(r"(Basic\s+)[A-Za-z0-9+/=]{10,}", re.IGNORECASE), r"\1***"),
        (re.compile(r"(_AGY_ASKPASS_NONCE=)[a-f0-9]{32,}", re.IGNORECASE), r"\1***"),
    ]

    def __init__(
        self,
        max_retention_bytes: int = MAX_RETENTION_BYTES,
        head_bytes: int = HEAD_RETENTION_BYTES,
        tail_bytes: int = TAIL_RETENTION_BYTES,
    ) -> None:
        self.max_retention_bytes = max_retention_bytes
        self.head_limit = head_bytes
        self.tail_limit = tail_bytes
        self._head = bytearray()
        self._tail = bytearray()
        self.total_bytes_read = 0

    def feed(self, chunk: bytes) -> None:
        """Feed a raw byte chunk from the stream into the bounded accumulator."""
        if not chunk:
            return
        self.total_bytes_read += len(chunk)

        # Fill head buffer up to limit
        head_needed = self.head_limit - len(self._head)
        if head_needed > 0:
            to_head = chunk[:head_needed]
            self._head.extend(to_head)
            remaining = chunk[head_needed:]
        else:
            remaining = chunk

        # Remainder goes to sliding tail buffer
        if remaining:
            self._tail.extend(remaining)
            overflow = len(self._tail) - self.tail_limit
            if overflow > 0:
                del self._tail[:overflow]

    def get_bytes(self) -> bytes:
        """Return the retained bytes, inserting a truncation notice if truncated."""
        if self.total_bytes_read <= (self.head_limit + self.tail_limit):
            return bytes(self._head + self._tail)
        truncated = self.total_bytes_read - len(self._head) - len(self._tail)
        marker = f"\n... [truncated {truncated} bytes] ...\n".encode("ascii")
        return bytes(self._head) + marker + bytes(self._tail)

    def sanitize(self, text: str) -> str:
        """Sanitize decoded text: strip ANSI escapes, control chars, and secrets."""
        # 1. Strip ANSI escape sequences
        text = self._ANSI_ESCAPE_RE.sub("", text)
        # 2. Strip non-printable control characters (preserving \t, \n, \r)
        text = self._CONTROL_CHAR_RE.sub("", text)
        # 3. Scrub secrets
        for pattern, replacement in self._SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def get_text(self) -> str:
        """Return safely decoded and sanitized stream text."""
        raw_bytes = self.get_bytes()
        decoded = raw_bytes.decode("utf-8", errors="replace")
        return self.sanitize(decoded)

    async def drain_reader(self, reader: asyncio.StreamReader) -> None:
        """Asynchronously read from an asyncio.StreamReader to EOF."""
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                self.feed(chunk)
        except asyncio.CancelledError:
            raise

    def drain_handle_sync(self, handle_val: int) -> None:
        """Synchronously read from a Win32 pipe handle until EOF in a background thread."""
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        h = wintypes.HANDLE(handle_val)
        buffer = (ctypes.c_char * 4096)()
        bytes_read = wintypes.DWORD()
        try:
            while True:
                ok = kernel32.ReadFile(h, buffer, 4096, ctypes.byref(bytes_read), None)
                if not ok or bytes_read.value == 0:
                    break
                self.feed(buffer.raw[: bytes_read.value])
        finally:
            try:
                kernel32.CloseHandle(h)
            except Exception:
                pass

    def drain_fd_sync(self, fd: int) -> None:
        """Synchronously read from a file descriptor until EOF in a background thread."""
        try:
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                self.feed(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


# =============================================================================
# Windows Win32 API Definitions & Job Object Runner
# =============================================================================

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class STARTUPINFOEXW(ctypes.Structure):
        _fields_ = [
            ("StartupInfo", STARTUPINFOW),
            ("lpAttributeList", wintypes.LPVOID),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryLimit", ctypes.c_size_t),
            ("PeakJobMemoryLimit", ctypes.c_size_t),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", wintypes.LARGE_INTEGER),
            ("TotalKernelTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
            ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    JobObjectBasicAccountingInformation = 1
    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    CREATE_UNICODE_ENVIRONMENT = 0x00000400
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    STARTF_USESTDHANDLES = 0x00000100
    HANDLE_FLAG_INHERIT = 0x00000001
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    OPEN_EXISTING = 3
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258
    INVALID_HANDLE_VALUE = -1

    # Explicit, 64-bit safe argtypes and restype signatures for all Win32 APIs
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL

    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.TerminateJobObject.restype = wintypes.BOOL

    kernel32.CreatePipe.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.CreatePipe.restype = wintypes.BOOL

    kernel32.SetHandleInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.SetHandleInformation.restype = wintypes.BOOL

    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE

    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL

    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD

    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.TerminateProcess.restype = wintypes.BOOL

    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD

    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL

    kernel32.GetLastError.argtypes = []
    kernel32.GetLastError.restype = wintypes.DWORD

    kernel32.InitializeProcThreadAttributeList.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL

    kernel32.UpdateProcThreadAttribute.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_size_t,
        wintypes.LPVOID,
        ctypes.c_size_t,
        wintypes.LPVOID,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL

    kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
    kernel32.DeleteProcThreadAttributeList.restype = None


def _is_valid_handle(h: Any) -> bool:
    """Return True if h is a valid non-null, non-invalid Win32 handle."""
    if h is None:
        return False
    val = getattr(h, "value", h)
    if val is None or val == 0 or val == -1 or val == 0xFFFFFFFFFFFFFFFF:
        return False
    return True


class WindowsJobRunner:
    """
    Windows process runner providing complete Job Object containment.

    Guarantees:
    - Creates the process suspended via CREATE_SUSPENDED.
    - Restricts handle inheritance strictly using STARTUPINFOEXW and
      PROC_THREAD_ATTRIBUTE_HANDLE_LIST (does not leak parent handles).
    - Passes a mutable Unicode command-line buffer to CreateProcessW.
    - Creates and configures a Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    - Assigns the suspended process to the Job Object before resuming execution.
    - If Job Object assignment fails, terminates the suspended process and fails closed.
    - Resumes the primary thread only after successful assignment.
    - Bounded stream draining ensures descendants holding pipes open do not block execution.
    - On timeout/cancellation: invokes TerminateJobObject and verifies ActiveProcesses == 0.
    - Concurrently drains stdout and stderr with BoundedStreamDrainer.
    - Closes all native handles reliably on every exit path.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise ProcessRunnerError("WindowsJobRunner is only supported on Windows")

    def _close_handle(self, handle: Any) -> None:
        """Safely close a native Win32 handle."""
        if _is_valid_handle(handle):
            try:
                kernel32.CloseHandle(handle)
            except Exception:
                pass

    def _create_job_object(self) -> Any:
        """Create a Windows Job Object configured with kill-on-close."""
        h_job = kernel32.CreateJobObjectW(None, None)
        if not _is_valid_handle(h_job):
            err = kernel32.GetLastError()
            raise ProcessRunnerError(f"CreateJobObjectW failed with error code {err}")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        res = kernel32.SetInformationJobObject(
            h_job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not res:
            err = kernel32.GetLastError()
            self._close_handle(h_job)
            raise ProcessRunnerError(f"SetInformationJobObject failed with error code {err}")

        return h_job

    def _create_pipes(self) -> tuple[Any, Any, Any, Any, Any]:
        """
        Create inheritable anonymous pipes for stdout and stderr and a NUL handle for stdin.
        Checks every Win32 API result and fails closed, cleaning up any previously
        allocated handles if any step fails.
        """
        sa = SECURITY_ATTRIBUTES()
        sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
        sa.bInheritHandle = True
        sa.lpSecurityDescriptor = None

        h_read_stdout = wintypes.HANDLE()
        h_write_stdout = wintypes.HANDLE()
        h_read_stderr = wintypes.HANDLE()
        h_write_stderr = wintypes.HANDLE()
        h_nul: Any = None

        try:
            # 1. Stdout pipe
            ok = kernel32.CreatePipe(
                ctypes.byref(h_read_stdout), ctypes.byref(h_write_stdout), ctypes.byref(sa), 0
            )
            if (
                not ok
                or not _is_valid_handle(h_read_stdout)
                or not _is_valid_handle(h_write_stdout)
            ):
                err = kernel32.GetLastError()
                raise ProcessRunnerError(f"CreatePipe (stdout) failed with error code {err}")

            ok = kernel32.SetHandleInformation(h_read_stdout, HANDLE_FLAG_INHERIT, 0)
            if not ok:
                err = kernel32.GetLastError()
                raise ProcessRunnerError(
                    f"SetHandleInformation (stdout read) failed with error {err}"
                )

            # 2. Stderr pipe
            ok = kernel32.CreatePipe(
                ctypes.byref(h_read_stderr), ctypes.byref(h_write_stderr), ctypes.byref(sa), 0
            )
            if (
                not ok
                or not _is_valid_handle(h_read_stderr)
                or not _is_valid_handle(h_write_stderr)
            ):
                err = kernel32.GetLastError()
                raise ProcessRunnerError(f"CreatePipe (stderr) failed with error code {err}")

            ok = kernel32.SetHandleInformation(h_read_stderr, HANDLE_FLAG_INHERIT, 0)
            if not ok:
                err = kernel32.GetLastError()
                raise ProcessRunnerError(
                    f"SetHandleInformation (stderr read) failed with error {err}"
                )

            # 3. NUL device for stdin
            h_nul = kernel32.CreateFileW(
                "NUL",
                GENERIC_READ,
                FILE_SHARE_READ,
                ctypes.byref(sa),
                OPEN_EXISTING,
                0,
                None,
            )
            if not _is_valid_handle(h_nul):
                err = kernel32.GetLastError()
                raise ProcessRunnerError(f"CreateFileW ('NUL') failed with error code {err}")

            return h_read_stdout, h_write_stdout, h_read_stderr, h_write_stderr, h_nul

        except Exception:
            self._close_handle(h_read_stdout)
            self._close_handle(h_write_stdout)
            self._close_handle(h_read_stderr)
            self._close_handle(h_write_stderr)
            self._close_handle(h_nul)
            raise

    def _create_handle_list_attribute(self, handles: list[Any]) -> tuple[wintypes.LPVOID, Any]:
        """
        Allocate and configure PROC_THREAD_ATTRIBUTE_HANDLE_LIST to restrict
        handle inheritance strictly to the specified handles.
        Returns (lp_attr_list, attr_buf). Caller must ensure
        DeleteProcThreadAttributeList is called on success.
        """
        size = ctypes.c_size_t(0)
        # Sizing call: lpAttributeList=None returns FALSE, size populated
        kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        if size.value <= 0:
            err = kernel32.GetLastError()
            raise ProcessRunnerError(
                f"InitializeProcThreadAttributeList returned invalid buffer size ({size.value}) "
                f"with error {err}"
            )

        attr_buf = (ctypes.c_char * size.value)()
        lp_attr_list = ctypes.cast(attr_buf, wintypes.LPVOID)

        initialized = False
        try:
            ok = kernel32.InitializeProcThreadAttributeList(lp_attr_list, 1, 0, ctypes.byref(size))
            if not ok:
                err = kernel32.GetLastError()
                raise ProcessRunnerError(
                    f"InitializeProcThreadAttributeList failed with error {err}"
                )
            initialized = True

            handle_array_type = wintypes.HANDLE * len(handles)
            handle_array = handle_array_type(*handles)

            ok = kernel32.UpdateProcThreadAttribute(
                lp_attr_list,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.cast(handle_array, wintypes.LPVOID),
                ctypes.sizeof(handle_array),
                None,
                None,
            )
            if not ok:
                err = kernel32.GetLastError()
                raise ProcessRunnerError(f"UpdateProcThreadAttribute failed with error {err}")

            return lp_attr_list, attr_buf
        except Exception:
            if initialized and lp_attr_list:
                try:
                    kernel32.DeleteProcThreadAttributeList(lp_attr_list)
                except Exception:
                    pass
            raise

    async def _verify_active_processes_zero(self, h_job: Any, timeout_seconds: float = 5.0) -> None:
        """
        Poll QueryInformationJobObject until ActiveProcesses == 0.
        Raises ProcessContainmentError if the count remains nonzero after deadline
        or if QueryInformationJobObject fails repeatedly and zero cannot be established.
        """
        deadline = time.monotonic() + timeout_seconds
        acct = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        last_error_code: int | None = None
        consecutive_query_failures = 0
        total_queries = 0

        while time.monotonic() < deadline:
            total_queries += 1
            res = kernel32.QueryInformationJobObject(
                h_job,
                JobObjectBasicAccountingInformation,
                ctypes.byref(acct),
                ctypes.sizeof(acct),
                None,
            )
            if res != 0:
                consecutive_query_failures = 0
                if acct.ActiveProcesses == 0:
                    return
            else:
                consecutive_query_failures += 1
                last_error_code = kernel32.GetLastError()

            await asyncio.sleep(0.05)

        # Reached deadline without establishing ActiveProcesses == 0
        if consecutive_query_failures > 0 and consecutive_query_failures == total_queries:
            raise ProcessContainmentError(
                f"QueryInformationJobObject failed repeatedly (error {last_error_code}); "
                f"cannot establish zero active processes in Job Object"
            )

        raise ProcessContainmentError(
            f"Job Object ActiveProcesses count remained nonzero ({acct.ActiveProcesses}) "
            f"after {timeout_seconds:.1f}s deadline"
        )

    async def _wait_for_process(self, h_process: Any, timeout_seconds: float) -> int:
        """Wait for process completion in bounded polling intervals to support cancellation."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessTimeoutError(
                    f"Process execution timed out after {timeout_seconds} seconds"
                )
            wait_ms = min(int(remaining * 1000), 200)
            res = await asyncio.to_thread(kernel32.WaitForSingleObject, h_process, wait_ms)
            if res == WAIT_OBJECT_0:
                exit_code = wintypes.DWORD()
                kernel32.GetExitCodeProcess(h_process, ctypes.byref(exit_code))
                return int(exit_code.value)
            elif res == WAIT_TIMEOUT:
                continue
            else:
                err = kernel32.GetLastError()
                raise ProcessRunnerError(
                    f"WaitForSingleObject failed with code {res} (error {err})"
                )

    async def _cleanup_on_error(
        self,
        h_job: Any,
        h_process: Any,
        stdout_task: asyncio.Task[None] | None,
        stderr_task: asyncio.Task[None] | None,
    ) -> None:
        """Terminate Job Object, verify active processes reaches 0, and cancel drainers."""
        containment_exc: Exception | None = None
        if h_job:
            try:
                kernel32.TerminateJobObject(h_job, 1)
            except Exception as exc:
                logger.warning("TerminateJobObject failed: %s", exc)

            try:
                await self._verify_active_processes_zero(h_job, timeout_seconds=5.0)
            except Exception as exc:
                logger.error("Job Object termination verification failed during cleanup: %s", exc)
                containment_exc = exc

        for task in (stdout_task, stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if h_process:
            try:
                kernel32.WaitForSingleObject(h_process, 1000)
            except Exception:
                pass

        if containment_exc is not None:
            raise containment_exc

    async def run(
        self,
        cmd: list[str] | str,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> ExecutionResult:
        """
        Execute command on Windows inside a secure Job Object with suspended-startup containment.
        """
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        cmd_str = subprocess.list2cmdline(cmd) if isinstance(cmd, list) else cmd
        # Pass a mutable Unicode command-line buffer to CreateProcessW
        cmd_buf = ctypes.create_unicode_buffer(cmd_str)
        start_time = time.monotonic()

        stdout_drainer = BoundedStreamDrainer()
        stderr_drainer = BoundedStreamDrainer()

        # Build safe environment block (denylist enforced after caller overrides)
        exec_env = prepare_safe_environment(env)
        env_str = "\0".join(f"{k}={v}" for k, v in sorted(exec_env.items())) + "\0\0"
        env_buf = ctypes.create_unicode_buffer(env_str)

        h_job: Any = None
        h_process: Any = None
        h_thread: Any = None
        h_read_stdout: Any = None
        h_write_stdout: Any = None
        h_read_stderr: Any = None
        h_write_stderr: Any = None
        h_nul: Any = None
        lp_attr_list: wintypes.LPVOID | None = None
        _attr_buf: Any = None

        stdout_task: asyncio.Task[None] | None = None
        stderr_task: asyncio.Task[None] | None = None

        try:
            # 1. Create Job Object
            h_job = self._create_job_object()

            # 2. Create Pipes (all results verified; fails closed)
            (
                h_read_stdout,
                h_write_stdout,
                h_read_stderr,
                h_write_stderr,
                h_nul,
            ) = self._create_pipes()

            # 3. Setup attribute list for strict handle inheritance restriction
            lp_attr_list, _attr_buf = self._create_handle_list_attribute(
                [h_write_stdout, h_write_stderr, h_nul]
            )

            # 4. Setup STARTUPINFOEXW and PROCESS_INFORMATION
            si_ex = STARTUPINFOEXW()
            si_ex.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            si_ex.StartupInfo.dwFlags = STARTF_USESTDHANDLES
            si_ex.StartupInfo.hStdOutput = h_write_stdout
            si_ex.StartupInfo.hStdError = h_write_stderr
            si_ex.StartupInfo.hStdInput = h_nul
            si_ex.lpAttributeList = lp_attr_list

            pi = PROCESS_INFORMATION()

            # 5. Create suspended process with restricted handle list
            creation_flags = (
                CREATE_SUSPENDED
                | CREATE_NO_WINDOW
                | CREATE_UNICODE_ENVIRONMENT
                | EXTENDED_STARTUPINFO_PRESENT
            )
            ok = kernel32.CreateProcessW(
                None,
                cmd_buf,
                None,
                None,
                True,
                creation_flags,
                env_buf,
                str(cwd.resolve()),
                ctypes.byref(si_ex),
                ctypes.byref(pi),
            )
            if not ok:
                err = kernel32.GetLastError()
                raise ProcessRunnerError(f"CreateProcessW failed with error code {err}")

            h_process = pi.hProcess
            h_thread = pi.hThread

            # Close write pipe ends and NUL in parent immediately
            self._close_handle(h_write_stdout)
            self._close_handle(h_write_stderr)
            self._close_handle(h_nul)
            h_write_stdout = None
            h_write_stderr = None
            h_nul = None

            # 6. Assign suspended process to Job Object (FAIL CLOSED if assignment fails)
            assign_ok = kernel32.AssignProcessToJobObject(h_job, h_process)
            if not assign_ok:
                err = kernel32.GetLastError()
                # Terminate suspended process immediately
                kernel32.TerminateProcess(h_process, 1)
                self._close_handle(h_thread)
                h_thread = None
                raise WindowsJobAssignmentError(
                    f"Failed to assign suspended process to Windows Job Object: error {err}"
                )

            # 7. Resume primary thread
            resume_res = kernel32.ResumeThread(h_thread)
            if resume_res == 0xFFFFFFFF or resume_res == -1:
                err = kernel32.GetLastError()
                kernel32.TerminateProcess(h_process, 1)
                self._close_handle(h_thread)
                h_thread = None
                raise ProcessRunnerError(f"ResumeThread failed with error {err}")

            # Close thread handle now that process is executing
            self._close_handle(h_thread)
            h_thread = None

            # 8. Start concurrent stream draining
            stdout_task = asyncio.create_task(
                asyncio.to_thread(stdout_drainer.drain_handle_sync, h_read_stdout.value)
            )
            stderr_task = asyncio.create_task(
                asyncio.to_thread(stderr_drainer.drain_handle_sync, h_read_stderr.value)
            )
            h_read_stdout = None
            h_read_stderr = None

            # 9. Wait for process exit or timeout
            exit_code = await self._wait_for_process(h_process, timeout)

            # 10. Bounded drainer completion (prevent hanging on descendants holding pipes open)
            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=3.0,
                )
            except TimeoutError:
                logger.warning(
                    "Stream drainers timed out after process exit; terminating Job Object"
                )
                if h_job:
                    try:
                        kernel32.TerminateJobObject(h_job, 1)
                    except Exception:
                        pass
                for t in (stdout_task, stderr_task):
                    if t is not None and not t.done():
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

            await self._verify_active_processes_zero(h_job, timeout_seconds=3.0)

            duration = time.monotonic() - start_time
            return ExecutionResult(
                command=cmd_str,
                exit_code=exit_code,
                stdout=stdout_drainer.get_text(),
                stderr=stderr_drainer.get_text(),
                duration_seconds=round(duration, 3),
            )

        except (ProcessTimeoutError, asyncio.CancelledError, Exception) as exc:
            cleanup_err: Exception | None = None
            try:
                await asyncio.shield(
                    self._cleanup_on_error(h_job, h_process, stdout_task, stderr_task)
                )
            except Exception as c_err:
                cleanup_err = c_err
                logger.error(
                    "Job Object containment failure during cleanup of %s: %s",
                    type(exc).__name__,
                    c_err,
                )

            if cleanup_err is not None:
                # If timeout or cancellation, chain containment failure so original failure
                # is preserved on __cause__ while containment failure is surfaced
                if isinstance(exc, (ProcessTimeoutError, asyncio.CancelledError)):
                    exc_name = type(exc).__name__
                    raise ProcessContainmentError(
                        f"Job Object containment verification failed during "
                        f"{exc_name} cleanup: {cleanup_err}"
                    ) from exc
                raise cleanup_err from exc
            raise

        finally:
            if lp_attr_list:
                try:
                    kernel32.DeleteProcThreadAttributeList(lp_attr_list)
                except Exception:
                    pass
            # Reliably close all native handles
            self._close_handle(h_thread)
            self._close_handle(h_process)
            self._close_handle(h_job)
            self._close_handle(h_write_stdout)
            self._close_handle(h_write_stderr)
            self._close_handle(h_nul)
            self._close_handle(h_read_stdout)
            self._close_handle(h_read_stderr)


# =============================================================================
# POSIX Process Runner & Process Group Containment
# =============================================================================


class POSIXProcessRunner:
    """
    POSIX process runner providing process-group management and containment.

    Guarantees:
    - Creates a new session (setsid) so Git and descendants belong to a distinct process group.
    - Tracks the original process group ID (pgid = proc.pid).
    - Concurrently drains stdout and stderr with BoundedStreamDrainer.
    - On timeout/cancellation: sends SIGTERM to process group, allows a bounded grace period,
      checks if the process group still exists (independently of direct child exit), escalates
      to SIGKILL, and verifies that the process group has exited via os.killpg(pgid, 0).
    - If process-group termination cannot be verified due to permission errors or timeouts,
      raises ProcessContainmentError (does not treat errors as successful termination).
    - Reaps the direct child process.

    Documented Limitation:
    - Verifies termination of the original process group only.
    - Descendant processes that deliberately escape by calling setsid() or setpgid()
      are outside this guarantee. Universal descendant containment on Linux requires
      cgroups v2 or PID namespaces.
    """

    async def _terminate_and_verify_posix(
        self,
        proc: asyncio.subprocess.Process,
        pgid: int,
        stdout_task: asyncio.Task[None] | None,
        stderr_task: asyncio.Task[None] | None,
        grace_period_seconds: float = 1.0,
        poll_timeout_seconds: float = 3.0,
    ) -> None:
        """
        Send SIGTERM, allow grace period, check if original process group still exists
        (independently of direct child exit), escalate to SIGKILL, verify termination
        with fail-closed error handling, and reap child.
        """
        import signal

        killpg = getattr(os, "killpg", None)
        sigterm = getattr(signal, "SIGTERM", None)
        sigkill = getattr(signal, "SIGKILL", None)

        # 1. Send SIGTERM to process group
        if killpg is not None and sigterm is not None:
            try:
                killpg(pgid, sigterm)
            except (ProcessLookupError, PermissionError):
                pass

        # 2. Bounded grace period for process group / direct child
        if grace_period_seconds > 0:
            try:
                if proc.returncode is None:
                    await asyncio.wait_for(proc.wait(), timeout=grace_period_seconds)
                else:
                    await asyncio.sleep(grace_period_seconds)
            except TimeoutError:
                pass
            except Exception:
                pass

        # 3. Check whether the original process group still exists,
        # independently of whether the direct child has exited
        group_survives = False
        if killpg is not None:
            try:
                killpg(pgid, 0)
                # 0 signal succeeded -> at least one process in process group is STILL ALIVE
                group_survives = True
            except ProcessLookupError:
                # ESRCH -> process group is completely dead
                group_survives = False
            except Exception:
                # Unexpected error or PermissionError -> cannot confirm dead, treat as surviving
                group_survives = True

        # 4. If any process in the group remains, escalate to SIGKILL
        if group_survives and killpg is not None and sigkill is not None:
            try:
                killpg(pgid, sigkill)
            except (ProcessLookupError, PermissionError):
                pass

        try:
            # 5. Verify original process group termination (fail-closed)
            if killpg is not None:
                deadline = time.monotonic() + poll_timeout_seconds
                group_terminated = False
                last_error: Exception | None = None
                while time.monotonic() < deadline:
                    try:
                        killpg(pgid, 0)
                        await asyncio.sleep(0.05)
                    except ProcessLookupError:
                        group_terminated = True
                        break
                    except (PermissionError, OSError) as exc:
                        last_error = exc
                        await asyncio.sleep(0.05)

                if not group_terminated:
                    if last_error is not None:
                        raise ProcessContainmentError(
                            f"Failed to verify POSIX process group {pgid} termination: {last_error}"
                        ) from last_error
                    raise ProcessContainmentError(
                        f"POSIX process group {pgid} failed to terminate "
                        f"within {poll_timeout_seconds}s"
                    )
        finally:
            # 6. Reap direct child
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                pass

            # 7. Cancel and await stream drainers
            for task in (stdout_task, stderr_task):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

    async def run(
        self,
        cmd: list[str] | str,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> ExecutionResult:
        """Execute command in a new POSIX session with process group tracking."""
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        cmd_args = shlex.split(cmd, posix=True) if isinstance(cmd, str) else list(cmd)
        start_time = time.monotonic()

        stdout_drainer = BoundedStreamDrainer()
        stderr_drainer = BoundedStreamDrainer()

        # Build safe environment block (denylist enforced after caller overrides)
        exec_env = prepare_safe_environment(env)

        proc: asyncio.subprocess.Process | None = None
        stdout_task: asyncio.Task[None] | None = None
        stderr_task: asyncio.Task[None] | None = None
        cleaned_up = False

        try:
            proc = await asyncio.create_subprocess_exec(
                cmd_args[0],
                *cmd_args[1:],
                cwd=str(cwd.resolve()),
                env=exec_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            pgid = proc.pid

            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout_task = asyncio.create_task(stdout_drainer.drain_reader(proc.stdout))
            stderr_task = asyncio.create_task(stderr_drainer.drain_reader(proc.stderr))

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except TimeoutError as exc:
                raise ProcessTimeoutError(
                    f"Process execution timed out after {timeout} seconds"
                ) from exc

            # Bounded drainer completion (prevent hanging on descendants holding pipes open)
            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=3.0,
                )
            except TimeoutError:
                logger.warning(
                    "POSIX stream drainers timed out after direct child exit (PGID: %d); "
                    "terminating process group to contain orphaned descendants",
                    pgid,
                )
                cleaned_up = True
                await self._terminate_and_verify_posix(proc, pgid, stdout_task, stderr_task)

            duration = time.monotonic() - start_time
            exit_code = proc.returncode if proc.returncode is not None else -1

            return ExecutionResult(
                command=" ".join(cmd_args),
                exit_code=exit_code,
                stdout=stdout_drainer.get_text(),
                stderr=stderr_drainer.get_text(),
                duration_seconds=round(duration, 3),
            )

        except (ProcessTimeoutError, asyncio.CancelledError, Exception):
            if proc is not None and not cleaned_up:
                await asyncio.shield(
                    self._terminate_and_verify_posix(proc, pgid, stdout_task, stderr_task)
                )
            raise


# =============================================================================
# Unified Secure Process Runner
# =============================================================================


class _ProcessRunnerBackend(Protocol):
    """Protocol for platform-specific process containment runners."""

    async def run(
        self,
        cmd: list[str] | str,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> ExecutionResult: ...


class SecureProcessRunner:
    """
    Unified cross-platform secure process runner for Git subprocess operations.
    Delegates to WindowsJobRunner on Windows and POSIXProcessRunner on POSIX systems.
    """

    def __init__(self, default_timeout: float = 60.0) -> None:
        self.default_timeout = default_timeout
        self._backend: _ProcessRunnerBackend
        if sys.platform == "win32":
            self._backend = WindowsJobRunner()
        else:
            self._backend = POSIXProcessRunner()

    async def run(
        self,
        cmd: list[str] | str,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Execute command with platform-appropriate containment, bounded stream draining,
        and timeout / cancellation protection.
        """
        exec_timeout = timeout or self.default_timeout
        res: ExecutionResult = await self._backend.run(
            cmd=cmd,
            cwd=cwd,
            env=env,
            timeout=exec_timeout,
        )
        return res


# Alias for callers expecting ProcessResult
ProcessResult = ExecutionResult


# =============================================================================
# Git Workspace Service
# =============================================================================


class GitWorkspaceService:
    """Manages Git version control actions inside an isolated job workspace."""

    def __init__(
        self,
        runner: SandboxRunner | None = None,
        empty_hooks_dir: Path | None = None,
        process_runner: SecureProcessRunner | None = None,
    ) -> None:
        self.runner = runner
        self.process_runner = process_runner or SecureProcessRunner()

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

    async def _execute_git(
        self,
        git_args: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute Git command via SandboxRunner (if provided) or SecureProcessRunner."""
        cmd = self._build_git_cmd(git_args)
        if self.runner is not None:
            return await self.runner.run(cmd, cwd=cwd, timeout=timeout)
        return await self.process_runner.run(cmd, cwd=cwd, env=env, timeout=timeout)

    async def init_or_verify_repo(self, workspace_path: Path) -> None:
        """Initialize a local git repository in workspace if one does not exist."""
        git_dir = workspace_path / ".git"
        if not git_dir.exists():
            res = await self._execute_git(["init"], cwd=workspace_path)
            if not res.success:
                raise GitServiceError(f"Failed to git init workspace: {res.stderr}")
            # Set local user config so commits succeed in automated environments
            await self._execute_git(
                ["config", "user.name", "Agentic Developer Bot"],
                cwd=workspace_path,
            )
            await self._execute_git(
                ["config", "user.email", "agent@developer.local"],
                cwd=workspace_path,
            )

    async def create_feature_branch(
        self,
        workspace_path: Path,
        branch_name: str,
    ) -> str:
        """Create and checkout a dedicated feature branch for the job."""
        await self.init_or_verify_repo(workspace_path)

        res = await self._execute_git(
            ["checkout", "-b", branch_name],
            cwd=workspace_path,
        )
        if not res.success:
            # If branch already exists, try checking it out
            res_checkout = await self._execute_git(
                ["checkout", branch_name],
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
        add_res = await self._execute_git(
            ["add", "-A"],
            cwd=workspace_path,
        )
        if not add_res.success:
            raise GitServiceError(f"git add failed: {add_res.stderr}")

        # git commit -m
        commit_res = await self._execute_git(
            ["commit", "-m", commit_message],
            cwd=workspace_path,
        )
        if not commit_res.success and "nothing to commit" not in commit_res.stdout:
            raise GitServiceError(f"git commit failed: {commit_res.stderr}")

        # Get commit SHA
        sha_res = await self._execute_git(
            ["rev-parse", "HEAD"],
            cwd=workspace_path,
        )
        sha = sha_res.stdout.strip()
        logger.info("Committed changes in %s (SHA: %s)", workspace_path, sha)
        return sha

    async def get_diff(self, workspace_path: Path, base_ref: str = "HEAD~1") -> str:
        """Return the unified diff of the latest commit."""
        res = await self._execute_git(
            ["diff", base_ref],
            cwd=workspace_path,
        )
        if res.success:
            return res.stdout
        # Fallback to diff of working directory against HEAD
        fallback_res = await self._execute_git(
            ["diff", "HEAD"],
            cwd=workspace_path,
        )
        return fallback_res.stdout if fallback_res.success else ""
