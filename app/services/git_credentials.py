"""
app/services/git_credentials.py
--------------------------------
Credential provider and ephemeral Askpass authentication session for Git operations.

Responsibilities:
- Defines the GitCredential, AskpassContext, and GitCredentialProvider abstractions.
- Implements SettingsPATCredentialProvider for Personal Access Tokens.
- Implements GitAskpassSession providing bounded, ephemeral, host-bound authentication
  for Git HTTPS requests via GIT_ASKPASS.
- Strictly validates prompts against canonical expected hosts and HTTPS schemes.
- Supports both username-then-password and password-only authentication flows.
- Enforces strict request quotas (max 1 username, max 1 password) and timeouts.
- Never persists credentials to disk, CLI arguments, or Git configuration.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import SecretStr

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Strict regex matching Git HTTPS credential prompts:
# Matches both:
#   "Username for 'https://github.com': "
#   "Password for 'https://x-access-token@github.com': "
#   "Password for 'https://github.com': "
#   "Password for 'https://github.com/owner/repo.git': "
_GIT_PROMPT_PATTERN = re.compile(
    r"^(?P<kind>Username|Password)\s+for\s+'(?P<url>[^']+)':\s*$",
    re.IGNORECASE,
)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class GitCredentialError(Exception):
    """Base exception for Git credential and Askpass session errors."""


class GitAskpassAuthError(GitCredentialError):
    """Raised on authentication handshake or nonce validation failure."""


class GitUnexpectedHostError(GitCredentialError):
    """Raised when Git prompts for credentials for an unexpected or unauthorized host."""


class GitAskpassQuotaExceededError(GitCredentialError):
    """Raised when Git makes more credential requests than permitted by the quota."""


# =============================================================================
# Data Models
# =============================================================================


@dataclass(frozen=True)
class GitCredential:
    """Encapsulated Git credential pairing."""

    username: str
    password: SecretStr

    def __repr__(self) -> str:
        return f"<GitCredential username={self.username!r} password=***>"


@dataclass(frozen=True)
class AskpassContext:
    """Paths and environment variables required for Git subprocess execution."""

    helper_path: Path
    env_overrides: dict[str, str]


# =============================================================================
# Prompt Validation & Host Binding
# =============================================================================


def parse_and_validate_prompt(prompt: str, expected_host: str) -> tuple[str, str]:
    """
    Parse a Git authentication prompt and verify it targets the expected host and HTTPS.

    Parameters
    ----------
    prompt:
        The raw prompt string passed by Git to GIT_ASKPASS.
    expected_host:
        The expected canonical hostname (e.g. 'github.com' or '127.0.0.1:8443').

    Returns
    -------
    tuple[str, str]
        (prompt_type, validated_host) where prompt_type is 'username' or 'password'.

    Raises
    ------
    GitAskpassAuthError:
        If the prompt does not match the expected Git HTTPS format or uses an insecure scheme.
    GitUnexpectedHostError:
        If the target host does not match expected_host.
    """
    match = _GIT_PROMPT_PATTERN.match(prompt.strip())
    if not match:
        raise GitAskpassAuthError(f"Malformed or unsupported Git prompt: {prompt!r}")

    kind = match.group("kind").lower()
    raw_url = match.group("url")

    parsed = urlsplit(raw_url)
    if parsed.scheme.lower() != "https":
        raise GitAskpassAuthError(
            f"Insecure scheme in Git prompt: {parsed.scheme!r}. Only HTTPS is permitted."
        )

    # Extract hostname and optional port
    prompt_host = (parsed.hostname or "").lower()
    if parsed.port is not None:
        prompt_host_with_port = f"{prompt_host}:{parsed.port}"
    else:
        prompt_host_with_port = prompt_host

    normalized_expected = expected_host.strip().lower()

    # Compare host (supporting both host and host:port expected values)
    target_matches = hmac.compare_digest(prompt_host, normalized_expected) or hmac.compare_digest(
        prompt_host_with_port, normalized_expected
    )

    if not target_matches:
        raise GitUnexpectedHostError(
            f"Host mismatch in Git prompt: expected {expected_host!r}, got {prompt_host!r}"
        )

    return kind, prompt_host_with_port


# =============================================================================
# Ephemeral Askpass Session
# =============================================================================


class GitAskpassSession(AbstractAsyncContextManager[AskpassContext]):
    """
    Ephemeral, bounded authentication session for Git subprocess invocations.

    Guarantees:
    - Binds credential release strictly to the canonical expected host and HTTPS.
    - Supports both two-prompt (username then password) and password-only flows.
    - Limits requests to strictly 1 username and 1 password prompt (max 2 requests total).
    - Session closes automatically upon serving password or exceeding quota/timeout.
    - Zero token strings exist in helper scripts, process environments, or CLI args.
    """

    def __init__(
        self,
        credential: GitCredential,
        expected_host: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.credential = credential
        self.expected_host = expected_host
        self.timeout_seconds = timeout_seconds

        # Cryptographic single-use nonce
        self._nonce = secrets.token_hex(32)

        # Quotas
        self._username_quota = 1
        self._password_quota = 1
        self._max_total_requests = 2
        self._total_requests = 0

        # Lifecycle
        self._server: asyncio.Server | None = None
        self._temp_dir: Path | None = None
        self._helper_path: Path | None = None
        self._is_closed = False
        self._use_unix_socket = hasattr(socket, "AF_UNIX")
        self._socket_path: Path | None = None
        self._tcp_port: int | None = None

    async def __aenter__(self) -> AskpassContext:
        raw_dir = tempfile.mkdtemp(prefix="agy_askpass_")
        self._temp_dir = Path(raw_dir).resolve()

        # Set restrictive permissions on POSIX
        if self._use_unix_socket and hasattr(os, "chmod"):
            try:
                os.chmod(self._temp_dir, 0o700)
            except OSError:
                pass

        # Start IPC Server
        if self._use_unix_socket and hasattr(asyncio, "start_unix_server"):
            start_unix_server = getattr(asyncio, "start_unix_server")
            self._socket_path = self._temp_dir / f"askpass_{secrets.token_hex(8)}.sock"
            self._server = await start_unix_server(
                self._handle_client,
                path=str(self._socket_path),
            )
            if hasattr(os, "chmod"):
                try:
                    os.chmod(self._socket_path, 0o600)
                except OSError:
                    pass
        else:
            # Fallback to loopback TCP (e.g. Windows)
            self._server = await asyncio.start_server(
                self._handle_client,
                host="127.0.0.1",
                port=0,
            )
            self._tcp_port = self._server.sockets[0].getsockname()[1]

        # Generate helper script
        self._helper_path = self._generate_helper_script()

        # Construct environment overrides
        env: dict[str, str] = {
            "GIT_ASKPASS": str(self._helper_path),
            "GIT_TERMINAL_PROMPT": "0",
            "_AGY_ASKPASS_NONCE": self._nonce,
        }
        if self._use_unix_socket and self._socket_path:
            env["_AGY_ASKPASS_SOCK"] = str(self._socket_path)
        elif self._tcp_port:
            env["_AGY_ASKPASS_PORT"] = str(self._tcp_port)

        return AskpassContext(helper_path=self._helper_path, env_overrides=env)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close IPC server and securely clean up temporary files."""
        if self._is_closed:
            return
        self._is_closed = True

        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None

        if self._temp_dir is not None and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception as exc:
                logger.debug("Failed cleaning askpass temp dir: %s", exc)
            self._temp_dir = None

    def _generate_helper_script(self) -> Path:
        """
        Generate the ephemeral Python Askpass helper script.
        Contains ZERO credentials. Connects to the IPC endpoint, passes prompt,
        writes response to stdout, and exits.
        """
        assert self._temp_dir is not None
        helper_file = self._temp_dir / f"askpass_{secrets.token_hex(8)}.py"

        # Ephemeral Python helper script
        script_content = (
            "import os, sys, socket\n"
            "try:\n"
            "    prompt = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''\n"
            "    nonce = os.environ.get('_AGY_ASKPASS_NONCE', '')\n"
            "    sock_path = os.environ.get('_AGY_ASKPASS_SOCK')\n"
            "    tcp_port = os.environ.get('_AGY_ASKPASS_PORT')\n"
            "    if sock_path and hasattr(socket, 'AF_UNIX'):\n"
            "        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "        s.connect(sock_path)\n"
            "    elif tcp_port:\n"
            "        s = socket.create_connection(('127.0.0.1', int(tcp_port)), timeout=10.0)\n"
            "    else:\n"
            "        sys.exit(1)\n"
            "    with s:\n"
            "        payload = f'{nonce}\\t{prompt}\\n'.encode('utf-8')\n"
            "        s.sendall(payload)\n"
            "        response = b''\n"
            "        while not response.endswith(b'\\n'):\n"
            "            chunk = s.recv(4096)\n"
            "            if not chunk:\n"
            "                break\n"
            "            response += chunk\n"
            "        sys.stdout.write(response.decode('utf-8', errors='replace'))\n"
            "        sys.stdout.flush()\n"
            "except Exception:\n"
            "    sys.stdout.write('\\n')\n"
            "    sys.stdout.flush()\n"
        )

        helper_file.write_text(script_content, encoding="utf-8")
        if sys.platform == "win32":
            helper_cmd = self._temp_dir / f"askpass_{secrets.token_hex(8)}.cmd"
            py_exe = sys.executable
            helper_cmd.write_text(f'@echo off\n"{py_exe}" "{helper_file}" %*\n', encoding="utf-8")
            return helper_cmd

        if hasattr(os, "chmod"):
            try:
                os.chmod(helper_file, 0o700)
            except OSError:
                pass

        return helper_file

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming connection from the ephemeral Askpass helper."""
        try:
            line_bytes = await asyncio.wait_for(reader.readline(), timeout=self.timeout_seconds)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")

            if "\t" not in line:
                # Malformed request format
                writer.write(b"\n")
                await writer.drain()
                return

            incoming_nonce, prompt = line.split("\t", 1)

            # Constant-time comparison for nonce validation
            if not hmac.compare_digest(incoming_nonce, self._nonce):
                logger.warning("Askpass rejected client: invalid nonce.")
                writer.write(b"\n")
                await writer.drain()
                return

            # Check total request quota
            if self._total_requests >= self._max_total_requests:
                logger.warning("Askpass rejected client: total quota exceeded.")
                writer.write(b"\n")
                await writer.drain()
                if self._server is not None:
                    self._server.close()
                return

            # Parse and validate prompt and expected host
            try:
                kind, _ = parse_and_validate_prompt(prompt, self.expected_host)
            except GitUnexpectedHostError as host_err:
                logger.warning("Askpass rejected unexpected host: %s", host_err)
                writer.write(b"\n")
                await writer.drain()
                # Unexpected host is a security violation: close server immediately
                if self._server is not None:
                    self._server.close()
                return
            except GitAskpassAuthError as auth_err:
                logger.warning("Askpass rejected invalid prompt: %s", auth_err)
                writer.write(b"\n")
                await writer.drain()
                if self._server is not None:
                    self._server.close()
                return

            response_text = ""

            # Classify prompt
            if kind == "username":
                if self._username_quota > 0:
                    self._username_quota -= 1
                    self._total_requests += 1
                    response_text = self.credential.username
                else:
                    logger.warning("Askpass rejected: username quota exceeded.")
            elif kind == "password":
                if self._password_quota > 0:
                    self._password_quota -= 1
                    self._total_requests += 1
                    response_text = self.credential.password.get_secret_value()
                else:
                    logger.warning("Askpass rejected: password quota exceeded.")

            # Send response followed by newline
            writer.write(response_text.encode("utf-8") + b"\n")
            await writer.drain()

            # If password quota is exhausted, close listener
            if self._password_quota == 0:
                if self._server is not None:
                    self._server.close()

        except Exception as exc:
            logger.debug("Error handling askpass client: %s", exc)
            try:
                writer.write(b"\n")
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# =============================================================================
# Credential Provider Protocol & PAT Implementation
# =============================================================================


@runtime_checkable
class GitCredentialProvider(Protocol):
    """Abstract protocol for Git credential provision and Askpass session acquisition."""

    async def get_credentials(self, host: str, owner: str, repo: str) -> GitCredential:
        """Retrieve credentials for the specified repository."""
        ...

    def acquire_askpass_context(
        self, host: str, owner: str, repo: str
    ) -> AbstractAsyncContextManager[AskpassContext]:
        """Produce an asynchronous context manager providing an active Askpass session."""
        ...


class SettingsPATCredentialProvider:
    """
    Personal Access Token credential provider using application Settings.
    Binds credentials strictly to the canonical GitHub API/Git host.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _resolve_canonical_host(self) -> str:
        """Resolve canonical Git hostname from GitHub API base URL."""
        parsed = urlsplit(self._settings.github_api_base_url)
        host = (parsed.hostname or "").lower()
        if host in ("api.github.com", "github.com", ""):
            return "github.com"
        # Enterprise host
        return host

    async def get_credentials(self, host: str, owner: str, repo: str) -> GitCredential:
        """
        Retrieve Git credentials for the specified host, owner, and repository.
        """
        canonical_host = self._resolve_canonical_host()
        normalized_host = host.strip().lower()

        # Verify host
        if normalized_host not in (canonical_host, "github.com"):
            raise GitUnexpectedHostError(
                f"Unauthorized Git host: {host!r}. Expected canonical host {canonical_host!r}"
            )

        return GitCredential(
            username="x-access-token",
            password=self._settings.github_token,
        )

    def acquire_askpass_context(
        self, host: str, owner: str, repo: str
    ) -> AbstractAsyncContextManager[AskpassContext]:
        """
        Acquire an ephemeral Askpass session bound strictly to the canonical host.
        """
        canonical_host = self._resolve_canonical_host()
        normalized_host = host.strip().lower()

        if normalized_host not in (canonical_host, "github.com"):
            raise GitUnexpectedHostError(
                f"Cannot acquire askpass context for unauthorized host: {host!r}"
            )

        credential = GitCredential(
            username="x-access-token",
            password=self._settings.github_token,
        )
        return GitAskpassSession(
            credential=credential,
            expected_host=normalized_host,
            timeout_seconds=self._settings.github_timeout_seconds,
        )
