"""
tests/unit/test_git_credentials.py
-----------------------------------
Unit and integration tests for Git credentials, ephemeral Askpass sessions,
and host-bound HTTPS authentication (app/services/git_credentials.py).

Test categories:
- TestPromptParsingAndHostBinding: Strict prompt format, scheme, and canonical host validation.
- TestSettingsPATCredentialProvider: Protocol conformance, canonical host resolution, and context.
- TestGitAskpassSessionUnit: Multi-flow auth (two-prompt & password-only), quotas, and IPC.
- TestHttpsAskpassIntegration: Real Git subprocess auth against local HTTPS with trusted test CA.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import os
import shutil
import socket
import ssl
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.services.git_credentials import (
    GitAskpassAuthError,
    GitAskpassSession,
    GitCredential,
    GitCredentialProvider,
    GitUnexpectedHostError,
    SettingsPATCredentialProvider,
    parse_and_validate_prompt,
)

# =============================================================================
# Ephemeral Test CA Certificate & Key (Valid for 127.0.0.1 and localhost)
# =============================================================================

TEST_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDJTCCAg2gAwIBAgIUdh9tDkBNRKrB/oOFwv30cpHe6VYwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJMTI3LjAuMC4xMB4XDTI2MTEyNDE2NDI0MFoXDTI3MTEy
NDE2NDI0MFowFDESMBAGA1UEAwwJMTI3LjAuMC4xMIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAuCKAZeeHAenYNFeQTtHse4D/7/S85XQ9FczF0y/uAJKK
2m84Q/5oH7mj/UQyRlvHy+lgiMErIxPLUrZk7k1To/1dhMUuSpo21MTU0XfUvgnt
d6s2hv/GJgHZZa/Z+o8h/dP7mnLRrsDujfoSDeXjg82hSoxlA+NXKq3/jOxKa+wM
x9aiJ3dHy8fiPPKkrEMxUF6I4jryiBON53ZbmFVHgJ9bYbb4oMWHr156Y6bLEZ/B
0sh1xxzX7qzub2/x/oplqdhlnYJHmWeAsfRq5RN/Rde6A2PXzgoD3s+g0y8cVoq1
Jq9E1uUQVd9kpzufdgXuSoOj1AF1qUo2goq7hc3fhQIDAQABo28wbTAdBgNVHQ4E
FgQUtub/KoA32b1JNs1nt0qIysT5OPcwHwYDVR0jBBgwFoAUtub/KoA32b1JNs1n
t0qIysT5OPcwDwYDVR0TAQH/BAUwAwEB/zAaBgNVHREEEzARhwR/AAABgglsb2Nh
bGhvc3QwDQYJKoZIhvcNAQELBQADggEBAG9QxPwM7W63FeaqZRAAOAEjz8OIDoaD
KY+Gzj+K4NQZLJ7W9n5iIgALgvGU152hNRvgX+/ZMpJoF/xxE1zvo1UL/LkQT3M1
F+lPC7SM0DJSIMNLO4EMdw+qSXk5ksOzB4pxzOFDSY4HY/9QuKm8at6/lyyLbmpT
hhp/PWHPz9tP1ICPRp9gWWEknQArD6gyCoZUeCN3bO1wbilts5YhDqWsoW3IP1C7
fWoLeBIHhyYH7XfY2AfRUCc9bNsqO/iEhXYgx9j1fly5egfVK6SrFsAMOvntBwiN
HW1fN595WZg0/UwcvanYXQeZ0+Wv0nuHEA6W4FwhrAFen/rmVk2yXvE=
-----END CERTIFICATE-----"""

TEST_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC4IoBl54cB6dg0
V5BO0ex7gP/v9LzldD0VzMXTL+4AkorabzhD/mgfuaP9RDJGW8fL6WCIwSsjE8tS
tmTuTVOj/V2ExS5KmjbUxNTRd9S+Ce13qzaG/8YmAdllr9n6jyH90/uactGuwO6N
+hIN5eODzaFKjGUD41cqrf+M7Epr7AzH1qInd0fLx+I88qSsQzFQXojiOvKIE43n
dluYVUeAn1thtvigxYevXnpjpssRn8HSyHXHHNfurO5vb/H+imWp2GWdgkeZZ4Cx
9GrlE39F17oDY9fOCgPez6DTLxxWirUmr0TW5RBV32SnO592Be5Kg6PUAXWpSjaC
iruFzd+FAgMBAAECggEAIEuY1oa1pK3FiGSlQFXMUtsMyKLVsK7i+dxZBBHffgU1
ylsAdNh9nbbbHtQjtzhowgh0MHgk3zmdCyxHZo50cYpCKUPGOzpyDH5bT0PjsdMe
7cK/RQ0+TTM81Htz0WDiVtkjw3KI+UwIayllmj9sWJUryV33Z6erhaZP8IEPCn9m
lr03lX0X42hxDpCNs3XwcPOrPBSScE1MR+RsvEaWaCnbLL0I+Lp412Fw+UDCXT+D
vLzijgPQNC9Bqv6pjnGEMiac5Cv4ioHuBxqcf9UVv6mv6JQ+NgFTKFFS2QGKf91s
XaETz0Gylijzs/H8A6VBNJrDV0hf1vHAtNaDesH6QQKBgQD+/18bgwhxZKaOyj5B
uDyFd1s4RRpWLuYKBvZBANOl7MU4qvVUese13JMLqPy+ONoHucWmcLQxOZ2YdZ53
Zrp6GWe9khjAfIpX7izZM50mux29DBgk4CxlAg9Wq+S44MPcYPWrxZ2vPNDhpXnt
ZlYTxp9rNffbpvuBWOO5yBl2ZQKBgQC429BkrHU4ljojIT58f5OHmm1yF+z8JBae
QNqDJBlrLHvuQE/RYxsVNY0qI8IlNi6MVdIVnhe2LMUsykts6nuCDJX11SlvzluB
OfuJk5VKU8AtlCxEchz6uqo8t2ajyx94nSifcNtdq9YXA7k2BGOnpnJK21ATvPDU
cOe7Xj0ioQKBgQDSYXLgSe1PUJEyQZ23rv8PxneWJz0hWbQlFbVZ0ArOXFgi9rlN
p7bZTlBrUFKlxlXsBdMSDO49pqFlRbUmIbU8Q3WZq82MEbQ586FA2hqP99UDQdUx
p1o5Q8fP5CXiel5dlglaXMJJ780OPe4odUQTMM4aL/58VHowL1xCzNplGQKBgQCo
cTt5kJ+beV/uRMuQXBtNE80oil6Z/jRR7Vxu5MT4oGco2Jcif4DdGZfQzxaIG8/c
dhLhOdX3XOIqBwclbqAuB1PZ1mtUgp9kqDXozX7aZGgX4A1aLPT1DoQyoRc4psy9
dc7fQdHRj9Ob0vvAGc3IpOU+Cor0AVUc7uhpFdz1IQKBgBdmTw2XSjiudDGnJ29p
bxOQ2OEEqzZK9ojSLSROhC4szYP+q3qEj1oIPPNiWqMyUK19kfEHarJCpUrQlzCG
T2QOqvwFH6c0o5KidKEfnDz2YJcuADy2MUdRq9SHNOGUH2ZVqLbaj2qqWZ3nT6A3
pnNWjQ3B6ZtrA1VRM8JUc0PM
-----END PRIVATE KEY-----"""


# =============================================================================
# Test HTTPS Git Authentication Server
# =============================================================================


class GitHttpAuthHandler(http.server.BaseHTTPRequestHandler):
    """
    Mock HTTPS handler simulating Git smart-HTTP authentication.
    Responds with 401 Unauthorized (Basic realm="Git") when unauthenticated,
    and 200 OK with minimal smart-http response when valid Basic Auth header arrives.
    """

    protocol_version = "HTTP/1.1"
    auth_headers: list[str] = []
    expected_basic_auth: str = ""

    def do_GET(self) -> None:
        auth_header = self.headers.get("Authorization", "")
        if auth_header:
            GitHttpAuthHandler.auth_headers.append(auth_header)

        if auth_header != self.expected_basic_auth:
            body = b"Authentication required\n"
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Git"')
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return

        # Authenticated successfully: serve minimal valid smart-http info/refs
        body = b"001e# service=git-upload-pack\n00000000"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-git-upload-pack-advertisement")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard logging during tests
        pass


@pytest.fixture
def test_tls_ca(tmp_path: Path) -> tuple[Path, Path]:
    """Write test certificate and private key to disk and return paths."""
    cert_path = tmp_path / "test_ca.pem"
    key_path = tmp_path / "test_key.pem"
    cert_path.write_text(TEST_CERT_PEM, encoding="utf-8")
    key_path.write_text(TEST_KEY_PEM, encoding="utf-8")
    return cert_path, key_path


@pytest.fixture
def https_git_server(
    test_tls_ca: tuple[Path, Path],
) -> Generator[tuple[str, str, list[str], Path], None, None]:
    """
    Start an ephemeral local HTTPS server requiring HTTP Basic Authentication.
    Returns (server_url, server_host, auth_headers_list, ca_cert_path).
    """
    cert_path, key_path = test_tls_ca
    GitHttpAuthHandler.auth_headers = []

    # Expected basic auth header: base64("x-access-token:ghp_mock_test_token_12345")
    expected_user_pass = "x-access-token:ghp_mock_test_token_12345"
    encoded = base64.b64encode(expected_user_pass.encode()).decode()
    GitHttpAuthHandler.expected_basic_auth = f"Basic {encoded}"

    # Setup SSL Context
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain(str(cert_path), str(key_path))

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), GitHttpAuthHandler)
    httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)

    port = httpd.socket.getsockname()[1]
    server_host = f"127.0.0.1:{port}"
    server_url = f"https://{server_host}/test-repo.git"

    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    yield server_url, server_host, GitHttpAuthHandler.auth_headers, cert_path

    httpd.shutdown()
    httpd.server_close()


# =============================================================================
# Unit Tests: Prompt Parsing & Host Validation
# =============================================================================


class TestPromptParsingAndHostBinding:
    """Verify parsing and canonical host validation for Git prompts."""

    def test_valid_username_prompt(self) -> None:
        prompt = "Username for 'https://github.com': "
        kind, host = parse_and_validate_prompt(prompt, expected_host="github.com")
        assert kind == "username"
        assert host == "github.com"

    def test_valid_password_prompt_bare_host(self) -> None:
        prompt = "Password for 'https://github.com': "
        kind, host = parse_and_validate_prompt(prompt, expected_host="github.com")
        assert kind == "password"
        assert host == "github.com"

    def test_valid_password_prompt_with_user(self) -> None:
        prompt = "Password for 'https://x-access-token@github.com': "
        kind, host = parse_and_validate_prompt(prompt, expected_host="github.com")
        assert kind == "password"
        assert host == "github.com"

    def test_valid_password_prompt_with_repo_path(self) -> None:
        prompt = "Password for 'https://github.com/octocat/Hello-World.git': "
        kind, host = parse_and_validate_prompt(prompt, expected_host="github.com")
        assert kind == "password"
        assert host == "github.com"

    def test_valid_prompt_with_port(self) -> None:
        prompt = "Password for 'https://127.0.0.1:8443/repo.git': "
        kind, host = parse_and_validate_prompt(prompt, expected_host="127.0.0.1:8443")
        assert kind == "password"
        assert host == "127.0.0.1:8443"

    def test_case_insensitive_host_matching(self) -> None:
        prompt = "Password for 'https://GITHUB.COM': "
        kind, host = parse_and_validate_prompt(prompt, expected_host="github.com")
        assert kind == "password"
        assert host == "github.com"

    def test_unexpected_host_rejected(self) -> None:
        prompt = "Password for 'https://evil-attacker.com': "
        with pytest.raises(GitUnexpectedHostError, match="Host mismatch"):
            parse_and_validate_prompt(prompt, expected_host="github.com")

    def test_subdomain_spoofing_rejected(self) -> None:
        prompt = "Password for 'https://github.com.evil.com': "
        with pytest.raises(GitUnexpectedHostError):
            parse_and_validate_prompt(prompt, expected_host="github.com")

    def test_insecure_http_scheme_rejected(self) -> None:
        prompt = "Password for 'http://github.com': "
        with pytest.raises(GitAskpassAuthError, match="Insecure scheme"):
            parse_and_validate_prompt(prompt, expected_host="github.com")

    def test_ssh_scheme_rejected(self) -> None:
        prompt = "Password for 'ssh://git@github.com': "
        with pytest.raises(GitAskpassAuthError, match="Insecure scheme"):
            parse_and_validate_prompt(prompt, expected_host="github.com")

    def test_malformed_prompt_format_rejected(self) -> None:
        invalid_prompts = [
            "Enter password: ",
            "Passphrase for key '/home/user/.ssh/id_rsa': ",
            "Are you sure you want to continue connecting (yes/no)? ",
            "Password for '': ",
            "Random unformatted text",
        ]
        for p in invalid_prompts:
            with pytest.raises(GitAskpassAuthError, match="Malformed or unsupported"):
                parse_and_validate_prompt(p, expected_host="github.com")


# =============================================================================
# Unit Tests: SettingsPATCredentialProvider
# =============================================================================


class TestSettingsPATCredentialProvider:
    """Verify PAT credential provider and protocol compliance."""

    def test_implements_protocol(self) -> None:
        provider = SettingsPATCredentialProvider(
            settings=Settings(
                github_token=SecretStr("ghp_test_token_12345"),
                llm_api_key=SecretStr("mock_key"),
            )
        )
        assert isinstance(provider, GitCredentialProvider)

    async def test_get_credentials_canonical_host_success(self) -> None:
        provider = SettingsPATCredentialProvider(
            settings=Settings(
                github_token=SecretStr("ghp_valid_secret"),
                llm_api_key=SecretStr("mock_key"),
                github_api_base_url="https://api.github.com",
            )
        )
        cred = await provider.get_credentials("github.com", "octocat", "Hello-World")
        assert cred.username == "x-access-token"
        assert cred.password.get_secret_value() == "ghp_valid_secret"
        # SecretStr representation masked
        assert "***" in repr(cred)
        assert "ghp_valid_secret" not in repr(cred)

    async def test_get_credentials_unauthorized_host_raises(self) -> None:
        provider = SettingsPATCredentialProvider(
            settings=Settings(
                github_token=SecretStr("ghp_valid_secret"),
                llm_api_key=SecretStr("mock_key"),
            )
        )
        with pytest.raises(GitUnexpectedHostError, match="Unauthorized Git host"):
            await provider.get_credentials("evil.com", "octocat", "Hello-World")

    def test_acquire_askpass_context_unauthorized_host_raises(self) -> None:
        provider = SettingsPATCredentialProvider(
            settings=Settings(
                github_token=SecretStr("ghp_valid_secret"),
                llm_api_key=SecretStr("mock_key"),
            )
        )
        with pytest.raises(GitUnexpectedHostError):
            provider.acquire_askpass_context("unauthorized.com", "octocat", "Hello-World")


# =============================================================================
# Unit Tests: GitAskpassSession Multi-Flow, Quotas & Lifecycle
# =============================================================================


class TestGitAskpassSessionUnit:
    """Verify Askpass multi-flow handling, quotas, and cleanup."""

    @pytest.mark.asyncio
    async def test_two_prompt_authentication_flow(self) -> None:
        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("secret_pat_98765"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",
            timeout_seconds=5.0,
        )

        async with session as ctx:
            assert ctx.helper_path.exists()
            nonce = ctx.env_overrides["_AGY_ASKPASS_NONCE"]
            port = ctx.env_overrides.get("_AGY_ASKPASS_PORT")
            sock = ctx.env_overrides.get("_AGY_ASKPASS_SOCK")

            # 1. Connect and ask for Username
            resp_user = await self._send_request(
                nonce, "Username for 'https://github.com': ", port, sock
            )
            assert resp_user == "x-access-token"

            # 2. Connect and ask for Password
            resp_pass = await self._send_request(
                nonce, "Password for 'https://x-access-token@github.com': ", port, sock
            )
            assert resp_pass == "secret_pat_98765"

        # Verify cleanup
        assert not ctx.helper_path.exists()

    @pytest.mark.asyncio
    async def test_password_only_authentication_flow(self) -> None:
        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("secret_pat_password_only"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",
            timeout_seconds=5.0,
        )

        async with session as ctx:
            nonce = ctx.env_overrides["_AGY_ASKPASS_NONCE"]
            port = ctx.env_overrides.get("_AGY_ASKPASS_PORT")
            sock = ctx.env_overrides.get("_AGY_ASKPASS_SOCK")

            # Immediate password prompt without username
            resp_pass = await self._send_request(
                nonce, "Password for 'https://x-access-token@github.com': ", port, sock
            )
            assert resp_pass == "secret_pat_password_only"

    @pytest.mark.asyncio
    async def test_quota_exhaustion_rejects_subsequent_requests(self) -> None:
        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("secret_token"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",
            timeout_seconds=5.0,
        )

        async with session as ctx:
            nonce = ctx.env_overrides["_AGY_ASKPASS_NONCE"]
            port = ctx.env_overrides.get("_AGY_ASKPASS_PORT")
            sock = ctx.env_overrides.get("_AGY_ASKPASS_SOCK")

            # First password prompt succeeds
            resp1 = await self._send_request(
                nonce, "Password for 'https://github.com': ", port, sock
            )
            assert resp1 == "secret_token"

            # Second password prompt must be rejected (quota exhausted)
            try:
                resp2 = await self._send_request(
                    nonce, "Password for 'https://github.com': ", port, sock
                )
                assert resp2 == ""
            except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError):
                pass  # Listener closed upon password quota exhaustion

    @pytest.mark.asyncio
    async def test_invalid_nonce_rejected(self) -> None:
        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("secret_token"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",
            timeout_seconds=5.0,
        )

        async with session as ctx:
            port = ctx.env_overrides.get("_AGY_ASKPASS_PORT")
            sock = ctx.env_overrides.get("_AGY_ASKPASS_SOCK")

            # Send request with invalid nonce
            resp = await self._send_request(
                "invalid_bogus_nonce_string",
                "Password for 'https://github.com': ",
                port,
                sock,
            )
            assert resp == ""

    @pytest.mark.asyncio
    async def test_unexpected_host_rejected_by_session(self) -> None:
        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("secret_token"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",
            timeout_seconds=5.0,
        )

        async with session as ctx:
            nonce = ctx.env_overrides["_AGY_ASKPASS_NONCE"]
            port = ctx.env_overrides.get("_AGY_ASKPASS_PORT")
            sock = ctx.env_overrides.get("_AGY_ASKPASS_SOCK")

            # Send prompt for different host
            resp = await self._send_request(
                nonce,
                "Password for 'https://evil.com': ",
                port,
                sock,
            )
            assert resp == ""

    @pytest.mark.asyncio
    async def test_helper_script_contains_zero_tokens(self) -> None:
        credential = GitCredential(
            username="secret_user",
            password=SecretStr("super_sensitive_token_content_xyz"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",
        )

        async with session as ctx:
            for file_path in ctx.helper_path.parent.iterdir():
                if file_path.is_file():
                    content = file_path.read_text(encoding="utf-8")
                    assert "super_sensitive_token_content_xyz" not in content
                    assert "secret_user" not in content

    async def _send_request(
        self,
        nonce: str,
        prompt: str,
        port: str | None,
        sock_path: str | None,
    ) -> str:
        """Helper to communicate directly with Askpass IPC listener."""
        if sock_path and hasattr(socket, "AF_UNIX"):
            reader, writer = await asyncio.open_unix_connection(sock_path)
        elif port:
            reader, writer = await asyncio.open_connection("127.0.0.1", int(port))
        else:
            raise RuntimeError("No valid IPC endpoint provided")

        try:
            payload = f"{nonce}\t{prompt}\n".encode()
            writer.write(payload)
            await writer.drain()

            resp_line = await reader.readline()
            return resp_line.decode("utf-8").rstrip("\r\n")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


# =============================================================================
# Integration Tests: Real Git Subprocess with Trusted Ephemeral CA
# =============================================================================


class TestHttpsAskpassIntegration:
    """
    Test real Git subprocess interaction with GitAskpassSession over HTTPS.
    Enforces TLS certificate verification using our ephemeral test CA.
    """

    @pytest.fixture(autouse=True)
    def check_git_available(self) -> None:
        if shutil.which("git") is None:
            pytest.skip("git CLI binary not available in PATH")

    @pytest.mark.asyncio
    async def test_git_https_two_prompt_auth_success(
        self,
        https_git_server: tuple[str, str, list[str], Path],
    ) -> None:
        server_url, server_host, auth_headers, ca_cert_path = https_git_server

        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("ghp_mock_test_token_12345"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host=server_host,
            timeout_seconds=15.0,
        )

        async with session as askpass_ctx:
            # Construct Git command vector enforcing TLS verification via ca_cert_path
            cmd = [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "http.sslBackend=openssl",
                "-c",
                f"http.sslCAInfo={ca_cert_path.as_posix()}",
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.https.allow=always",
                "ls-remote",
                server_url,
            ]

            env = dict(os.environ)
            env.update(askpass_ctx.env_overrides)

            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

            assert proc.returncode == 0, f"git ls-remote failed: {stderr.decode()}"
            # Verify that Basic Auth header reached server
            assert len(auth_headers) >= 1
            expected_encoded = base64.b64encode(
                b"x-access-token:ghp_mock_test_token_12345"
            ).decode()
            assert f"Basic {expected_encoded}" in auth_headers

    @pytest.mark.asyncio
    async def test_git_https_password_only_auth_success(
        self,
        https_git_server: tuple[str, str, list[str], Path],
    ) -> None:
        server_url, server_host, auth_headers, ca_cert_path = https_git_server

        # Embed username in URL: https://x-access-token@127.0.0.1:port/test-repo.git
        parsed = urlsplit(server_url)
        url_with_user = f"https://x-access-token@{parsed.netloc}{parsed.path}"

        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("ghp_mock_test_token_12345"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host=server_host,
            timeout_seconds=15.0,
        )

        async with session as askpass_ctx:
            cmd = [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "http.sslBackend=openssl",
                "-c",
                f"http.sslCAInfo={ca_cert_path.as_posix()}",
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.https.allow=always",
                "ls-remote",
                url_with_user,
            ]

            env = dict(os.environ)
            env.update(askpass_ctx.env_overrides)

            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

            assert proc.returncode == 0, f"git ls-remote failed: {stderr.decode()}"
            assert len(auth_headers) >= 1
            expected_encoded = base64.b64encode(
                b"x-access-token:ghp_mock_test_token_12345"
            ).decode()
            assert f"Basic {expected_encoded}" in auth_headers

    @pytest.mark.asyncio
    async def test_git_insecure_transports_strictly_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Verify production transport policy (protocol.allow=never, protocol.https.allow=always)
        strictly rejects file://, http://, and ssh:// transports.
        """
        insecure_urls = [
            f"file://{tmp_path.as_posix()}/dummy.git",
            "http://127.0.0.1:8080/dummy.git",
            "ssh://git@github.com/dummy.git",
        ]

        for target_url in insecure_urls:
            cmd = [
                "git",
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.https.allow=always",
                "ls-remote",
                target_url,
            ]

            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            # Must fail with protocol error
            assert proc.returncode != 0
            err_msg = stderr.decode().lower()
            assert "protocol" in err_msg or "disabled" in err_msg or "not allowed" in err_msg

    @pytest.mark.asyncio
    async def test_git_askpass_rejects_unexpected_host(
        self,
        https_git_server: tuple[str, str, list[str], Path],
    ) -> None:
        server_url, server_host, auth_headers, ca_cert_path = https_git_server

        # Askpass configured strictly for github.com, but target is server_host (127.0.0.1:port)
        credential = GitCredential(
            username="x-access-token",
            password=SecretStr("ghp_mock_test_token_12345"),
        )
        session = GitAskpassSession(
            credential=credential,
            expected_host="github.com",  # Intentionally mismatched!
            timeout_seconds=15.0,
        )

        async with session as askpass_ctx:
            cmd = [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "http.sslBackend=openssl",
                "-c",
                f"http.sslCAInfo={ca_cert_path.as_posix()}",
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.https.allow=always",
                "ls-remote",
                server_url,
            ]

            env = dict(os.environ)
            env.update(askpass_ctx.env_overrides)

            proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

            # Must fail because Askpass refused to provide credentials for unexpected host
            assert proc.returncode != 0
            err_msg = stderr.decode()
            assert (
                "Authentication failed" in err_msg
                or "could not read" in err_msg
                or "terminal prompts disabled" in err_msg
            )
