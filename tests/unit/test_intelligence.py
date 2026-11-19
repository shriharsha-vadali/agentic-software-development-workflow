"""
tests/unit/test_intelligence.py
-------------------------------
Unit tests for the Codebase Intelligence & RAG engine (app/intelligence/).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.intelligence.context_builder import ContextBuilder
from app.intelligence.indexer import CodebaseIndexer
from app.intelligence.parser import PythonASTParser
from app.intelligence.retriever import CodebaseRetriever
from app.intelligence.workspace import WorkspaceManager


class TestWorkspaceManager:
    """Verify workspace isolation, safe path resolution, and file scanning."""

    def test_workspace_isolation_and_safe_path(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(base_workspaces_dir=tmp_path)
        ws = mgr.create_workspace(job_id=123)
        assert ws.exists()
        assert ws == (tmp_path / "job_123").resolve()

        # Safe resolution inside workspace
        safe_file = mgr.resolve_safe_path(123, "src/main.py")
        assert safe_file == (ws / "src" / "main.py").resolve()

        # Path traversal attack must be blocked
        with pytest.raises(ValueError, match="Path traversal detected"):
            mgr.resolve_safe_path(123, "../../etc/passwd")

    def test_file_discovery_and_filtering(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(base_workspaces_dir=tmp_path)
        ws = mgr.create_workspace(job_id=456)

        # Create eligible files
        (ws / "app.py").write_text("print('hello')", encoding="utf-8")
        (ws / "README.md").write_text("# Project", encoding="utf-8")

        # Create ignored files and directories
        git_dir = ws / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

        pycache_dir = ws / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "app.cpython-311.pyc").write_bytes(b"dummy")

        (ws / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        files = mgr.discover_files(ws)
        rel_files = [str(f.relative_to(ws)).replace("\\", "/") for f in files]

        assert "app.py" in rel_files
        assert "README.md" in rel_files
        assert not any(".git" in f for f in rel_files)
        assert not any("__pycache__" in f for f in rel_files)
        assert not any(f.endswith(".png") for f in rel_files)

    def test_detect_project_type(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(base_workspaces_dir=tmp_path)
        ws = mgr.create_workspace(job_id=789)

        (ws / "pyproject.toml").write_text("[project]\nname='test'", encoding="utf-8")
        (ws / "tests").mkdir()

        info = mgr.detect_project_type(ws)
        assert info["primary_language"] == "python"
        assert "pytest" in info["test_frameworks"]
        assert info["has_tests"] is True


class TestPythonASTParser:
    """Verify AST parsing of Python source."""

    def test_parse_functions_classes_and_methods(self) -> None:
        code = '''"""Module docstring."""
import os
from math import sqrt

def calculate_area(radius: float) -> float:
    """Calculate circle area."""
    return 3.14 * radius * radius

class Shape:
    """Shape base class."""
    def area(self) -> float:
        return 0.0

    async def async_compute(self):
        pass
'''
        parser = PythonASTParser()
        parsed = parser.parse_source(code, "shapes.py")

        assert parsed.docstring == "Module docstring."
        assert "os" in parsed.imports
        assert "math.sqrt" in parsed.imports

        sym_names = [s.name for s in parsed.symbols]
        assert "calculate_area" in sym_names
        assert "Shape" in sym_names
        assert "Shape.area" in sym_names
        assert "Shape.async_compute" in sym_names

        calc_func = next(s for s in parsed.symbols if s.name == "calculate_area")
        assert calc_func.kind == "function"
        assert calc_func.parameters == ["radius"]
        assert calc_func.docstring == "Calculate circle area."

    def test_parse_syntax_error_handled_gracefully(self) -> None:
        bad_code = "def invalid_syntax(:"
        parser = PythonASTParser()
        parsed = parser.parse_source(bad_code, "broken.py")

        assert parsed.parse_error is not None
        assert "SyntaxError" in parsed.parse_error


class TestCodeChunkerAndRetrieval:
    """Verify chunking, indexing, and context retrieval."""

    def test_chunking_and_retrieval(self, tmp_path: Path) -> None:
        # Prepare sample repo
        (tmp_path / "calculator.py").write_text(
            """def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
""",
            encoding="utf-8",
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_calculator.py").write_text(
            """import pytest
from calculator import add

def test_add():
    assert add(2, 3) == 5
""",
            encoding="utf-8",
        )

        indexer = CodebaseIndexer()
        index = indexer.index_workspace(tmp_path)
        assert index.total_chunks >= 3

        retriever = CodebaseRetriever(index)
        results = retriever.retrieve("Fix bug in multiply function", top_k=5)

        assert len(results) > 0
        top_chunk, score = results[0]
        assert top_chunk.symbol_name == "multiply"
        assert score > 1.0

        # Query about tests should boost test chunk
        test_results = retriever.retrieve("Run pytest test_add failure", top_k=5)
        assert any(c.kind == "test" for c, _ in test_results)

    def test_context_builder_budget(self) -> None:
        from app.intelligence.chunker import CodeChunk

        chunks = [
            (
                CodeChunk(
                    file_path="app.py",
                    start_line=1,
                    end_line=5,
                    content="print('short chunk')",
                    kind="function",
                    symbol_name="run",
                ),
                3.5,
            )
        ]
        builder = ContextBuilder(max_context_chars=500)
        context = builder.build_context(chunks, project_info={"primary_language": "python"})

        assert "### Codebase Context" in context
        assert "app.py" in context
        assert "print('short chunk')" in context
