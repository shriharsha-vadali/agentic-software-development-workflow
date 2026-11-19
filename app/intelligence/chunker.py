"""
app/intelligence/chunker.py
---------------------------
AST-aware code chunking for repository understanding and context retrieval.
Preserves line spans, file metadata, symbol names, and keyword indices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.intelligence.parser import PythonASTParser

# Tokenizer pattern for extracting alphanumeric keywords for retrieval
_TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")


@dataclass
class CodeChunk:
    """A semantic chunk of source code or configuration for indexing/retrieval."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    kind: str  # "function", "class", "method", "imports", "test", "config", "file"
    symbol_name: str | None = None
    keywords: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.keywords:
            tokens = _TOKEN_PATTERN.findall(self.content.lower())
            tokens.extend(_TOKEN_PATTERN.findall(self.file_path.lower()))
            if self.symbol_name:
                tokens.extend(_TOKEN_PATTERN.findall(self.symbol_name.lower()))
            self.keywords = set(tokens)

    @property
    def header(self) -> str:
        """Formatted header identifying file path and line span."""
        if self.symbol_name:
            return f"{self.file_path} (L{self.start_line}-L{self.end_line}) [{self.symbol_name}]"
        return f"{self.file_path} (L{self.start_line}-L{self.end_line})"


class CodeChunker:
    """Produces AST-aware code chunks for Python and line-based chunks for other files."""

    def __init__(self, parser: PythonASTParser | None = None) -> None:
        self.parser = parser or PythonASTParser()

    def chunk_python(self, code: str, file_path: str) -> list[CodeChunk]:
        """Produce semantic AST-aware chunks from Python code."""
        parsed = self.parser.parse_source(code, file_path)
        lines = code.splitlines()
        chunks: list[CodeChunk] = []

        is_test_file = "test" in file_path.lower()

        # If parse failed (e.g. syntax error), fall back to whole file or line chunks
        if parsed.parse_error or not parsed.symbols:
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    content=code,
                    kind="test" if is_test_file else "file",
                )
            )
            return chunks

        # 1. Imports and module preamble chunk
        if parsed.imports:
            import_lines = [
                line
                for line in lines
                if line.strip().startswith("import ") or line.strip().startswith("from ")
            ]
            if import_lines:
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=1,
                        end_line=min(30, len(lines)),
                        content="\n".join(import_lines),
                        kind="imports",
                        symbol_name="__imports__",
                    )
                )

        # 2. Symbol chunks (functions, classes, methods)
        covered_lines: set[int] = set()
        for sym in parsed.symbols:
            # Extract symbol slice
            s_idx = max(0, sym.start_line - 1)
            e_idx = min(len(lines), sym.end_line)
            chunk_content = "\n".join(lines[s_idx:e_idx])

            kind = "test" if is_test_file else sym.kind
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    content=chunk_content,
                    kind=kind,
                    symbol_name=sym.name,
                )
            )
            covered_lines.update(range(sym.start_line, sym.end_line + 1))

        return chunks

    def chunk_text_file(
        self, content: str, file_path: str, max_lines: int = 150
    ) -> list[CodeChunk]:
        """Chunk non-Python files (e.g. pyproject.toml, json, markdown)."""
        lines = content.splitlines()
        chunks: list[CodeChunk] = []
        is_config = any(
            c in file_path.lower()
            for c in ("pyproject.toml", "package.json", "setup.py", "dockerfile")
        )

        kind = "config" if is_config else "file"

        if len(lines) <= max_lines:
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    content=content,
                    kind=kind,
                )
            )
            return chunks

        # Sliding window with overlap
        step = max_lines - 30
        for i in range(0, len(lines), step):
            window = lines[i : i + max_lines]
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=min(len(lines), i + max_lines),
                    content="\n".join(window),
                    kind=kind,
                )
            )
            if i + max_lines >= len(lines):
                break

        return chunks
