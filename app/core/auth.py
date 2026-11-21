"""
app/core/auth.py
----------------
API authentication and constant-time key verification.
Provides FastAPI dependencies for securing endpoints with API keys.
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
HTTP_BEARER = HTTPBearer(auto_error=False)


def constant_time_compare(val_a: str, val_b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    """
    return secrets.compare_digest(val_a.encode("utf-8"), val_b.encode("utf-8"))


def verify_api_key(
    provided_key: str | None,
    expected_key: SecretStr | None,
) -> bool:
    """
    Validate an incoming API key against the configured expected secret in constant time.
    Fails closed if either key is missing or empty.
    """
    if not provided_key or not expected_key:
        return False
    secret_val = expected_key.get_secret_value().strip()
    if not secret_val:
        return False
    return constant_time_compare(provided_key.strip(), secret_val)


async def get_current_api_key(
    api_key_header: Annotated[str | None, Security(API_KEY_HEADER)] = None,
    bearer_creds: Annotated[HTTPAuthorizationCredentials | None, Security(HTTP_BEARER)] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> str | None:
    """
    FastAPI security dependency to validate the API key.

    Behaviour:
    - If settings.api_auth_enabled is False:
      Allows request without authentication (returns None).
    - If settings.api_auth_enabled is True:
      - Accepts key from either 'X-API-Key' header or 'Authorization: Bearer <key>'.
      - If header is missing: raises HTTP 401 Unauthorized.
      - If key does not match: raises HTTP 401 Unauthorized.
      - If settings.api_key is not configured: fails closed with HTTP 401 Unauthorized.
      - If key matches: returns the validated key string.
    """
    if settings is None:
        settings = get_settings()

    if not settings.api_auth_enabled:
        return None

    # Fail closed if auth is enabled but no server key configured
    if settings.api_key is None or not settings.api_key.get_secret_value().strip():
        logger.error("API authentication is enabled but no valid api_key is configured.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication unavailable: server API key is not configured.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Extract provided key from X-API-Key or Bearer token
    provided_key: str | None = None
    if api_key_header and api_key_header.strip():
        provided_key = api_key_header.strip()
    elif bearer_creds and bearer_creds.credentials.strip():
        provided_key = bearer_creds.credentials.strip()

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing API key. Provide via 'X-API-Key' or 'Authorization: Bearer <key>' header."
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not verify_api_key(provided_key, settings.api_key):
        logger.warning("Rejected API request: invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return provided_key
