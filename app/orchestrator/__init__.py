"""
app.orchestrator package
------------------------
Workflow orchestration engine: state machine, persistent pipeline execution,
approval gates, background tasks, and crash recovery.
"""

from app.orchestrator.state_machine import (
    VALID_JOB_TRANSITIONS,
    InvalidTransitionError,
    JobStateMachine,
)
from app.orchestrator.workflow import JobOrchestrator

__all__ = [
    "InvalidTransitionError",
    "JobOrchestrator",
    "JobStateMachine",
    "VALID_JOB_TRANSITIONS",
]
