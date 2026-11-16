"""
app/api/health.py
-----------------
Health check endpoint router for system monitoring and liveness probes.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Predictable response payload for the health check endpoint."""

    status: str = Field(
        default="healthy",
        description="Application operational status indicator.",
        examples=["healthy"],
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check probe",
    description="Returns simple operational status without exposing internal system details.",
)
async def health_check() -> HealthResponse:
    """Return application operational status."""
    return HealthResponse(status="healthy")
