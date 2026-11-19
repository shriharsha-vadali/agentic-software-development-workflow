"""
tests/unit/test_llm_and_planner.py
----------------------------------
Unit tests for LLM adapter (app/adapters/llm.py) and Planning Engine (app/intelligence/planner.py).
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.llm import (
    LLMAuthError,
    LLMTimeoutError,
    MockLLMClient,
    OpenAILLMClient,
)
from app.db.models import Base, Job, JobStatus
from app.intelligence.planner import PlanningEngine
from app.services.issue_ingestion import NormalizedIssueData

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


class DummySchema(BaseModel):
    name: str
    count: int


class TestLLMAdapters:
    """Verify OpenAILLMClient and MockLLMClient behaviors."""

    async def test_mock_llm_client_structured(self) -> None:
        client = MockLLMClient(predefined_responses={"test": {"name": "hello", "count": 5}})
        result = await client.generate_structured("Please test this", DummySchema)
        assert result.name == "hello"
        assert result.count == 5

    async def test_openai_client_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "Authorization" in request.headers
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"name": "production", "count": 42}'}}]},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = OpenAILLMClient(
                api_key="sk-test-key",
                http_client=mock_http,
            )
            result = await client.generate_structured("Generate config", DummySchema)
            assert result.name == "production"
            assert result.count == 42

    async def test_openai_client_auth_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid_api_key"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = OpenAILLMClient(api_key="bad-key", http_client=mock_http)
            with pytest.raises(LLMAuthError, match="authentication failed"):
                await client.generate_text("Hello")

    async def test_openai_client_timeout_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timeout")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            client = OpenAILLMClient(api_key="key", http_client=mock_http)
            with pytest.raises(LLMTimeoutError, match="timed out"):
                await client.generate_text("Hello")


class TestPlanningEngine:
    """Verify PlanningEngine prompt building and database plan persistence."""

    async def test_generate_and_persist_plan(self, test_session: AsyncSession) -> None:
        mock_plan_data = {
            "summary": "Fix calculation error in math module",
            "problem_statement": "Division by zero occurs when denominator is 0",
            "acceptance_criteria": ["Handle zero denominator by raising ZeroDivisionError"],
            "technical_approach": "Add input validation in math.py",
            "files_to_create": [],
            "files_to_modify": ["math.py"],
            "files_to_delete": [],
            "steps": [
                {
                    "step_number": 1,
                    "description": "Check if denom is 0",
                    "file_paths": ["math.py"],
                }
            ],
            "test_plan": "Add test_division_by_zero in test_math.py",
            "risks": "Low risk of breaking callers",
            "estimated_complexity": "low",
        }
        mock_client = MockLLMClient(predefined_responses={"GitHub Issue": mock_plan_data})
        engine = PlanningEngine(mock_client)

        job = Job(
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            repo_owner="octocat",
            repo_name="Hello-World",
            issue_number=1,
            status=JobStatus.INDEXING_CODEBASE,
        )
        test_session.add(job)
        await test_session.commit()
        await test_session.refresh(job)

        issue_data = NormalizedIssueData(
            owner="octocat",
            repo="Hello-World",
            issue_number=1,
            issue_url="https://github.com/octocat/Hello-World/issues/1",
            title="Zero division error",
            body="Division by zero raises unexpected crash",
            state="open",
            author="alice",
        )

        plan_row, plan_struct = await engine.create_and_persist_plan(
            test_session, job, issue_data, "### Code Context\nmath.py"
        )

        assert plan_row.job_id == job.id
        assert plan_row.status == "draft"
        assert len(plan_row.steps) == 1
        assert plan_struct.summary == "Fix calculation error in math module"
        assert job.status == JobStatus.AWAITING_APPROVAL
