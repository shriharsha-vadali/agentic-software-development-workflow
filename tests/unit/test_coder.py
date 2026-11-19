"""
tests/unit/test_coder.py
------------------------
Unit tests for AI code generator and patch application (app/intelligence/coder.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.llm import MockLLMClient
from app.intelligence.coder import CodeGenerator, FileChange
from app.intelligence.planner import ImplementationPlan, ImplementationStep
from app.intelligence.workspace import WorkspaceManager


class TestCodeGenerator:
    """Verify code generation, path validation, and diff generation."""

    def test_apply_create_modify_delete(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(base_workspaces_dir=tmp_path)
        ws = mgr.create_workspace(job_id=1)

        generator = CodeGenerator(llm_client=MockLLMClient(), workspace_manager=mgr)

        # 1. Create file
        changes = [
            FileChange(
                path="src/calc.py",
                action="create",
                content="def add(a, b):\n    return a + b\n",
            )
        ]
        modified, diff = generator.validate_and_apply_changes(1, changes)
        assert "src/calc.py" in modified
        assert (ws / "src" / "calc.py").exists()
        assert "+def add(a, b):" in diff

        # 2. Modify file
        changes2 = [
            FileChange(
                path="src/calc.py",
                action="modify",
                content="def add(a, b):\n    return a + b + 0\n",
            )
        ]
        modified2, diff2 = generator.validate_and_apply_changes(1, changes2)
        assert "src/calc.py" in modified2
        assert "-    return a + b" in diff2
        assert "+    return a + b + 0" in diff2

        # 3. Delete file
        changes3 = [FileChange(path="src/calc.py", action="delete")]
        modified3, diff3 = generator.validate_and_apply_changes(1, changes3)
        assert "src/calc.py" in modified3
        assert not (ws / "src" / "calc.py").exists()
        assert "--- a/src/calc.py" in diff3

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(base_workspaces_dir=tmp_path)
        mgr.create_workspace(job_id=2)
        generator = CodeGenerator(llm_client=MockLLMClient(), workspace_manager=mgr)

        with pytest.raises(ValueError, match="Prohibited path"):
            generator.validate_and_apply_changes(
                2, [FileChange(path="../../outside.py", action="create", content="bad")]
            )

        with pytest.raises(ValueError, match="Prohibited path"):
            generator.validate_and_apply_changes(
                2, [FileChange(path=".git/config", action="modify", content="bad")]
            )

    async def test_generate_code_mock(self) -> None:
        mock_data = {
            "summary": "Implemented addition function",
            "changes": [
                {
                    "path": "math.py",
                    "action": "create",
                    "content": "def add(x, y):\n    return x + y\n",
                }
            ],
        }
        client = MockLLMClient(predefined_responses={"Approved Implementation Plan": mock_data})
        generator = CodeGenerator(llm_client=client)

        plan = ImplementationPlan(
            summary="Add math functions",
            problem_statement="Need math helpers",
            technical_approach="Create math.py",
            steps=[
                ImplementationStep(
                    step_number=1, description="Create math.py", file_paths=["math.py"]
                )
            ],
            test_plan="Write unit tests",
        )
        res = await generator.generate_code(plan, "### Context\nNone")
        assert res.summary == "Implemented addition function"
        assert len(res.changes) == 1
        assert res.changes[0].path == "math.py"
