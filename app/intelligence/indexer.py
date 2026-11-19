"""
app/intelligence/indexer.py
---------------------------
Repository indexer scanning workspace files and building an in-memory code index.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.intelligence.chunker import CodeChunk, CodeChunker
from app.intelligence.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class CodebaseIndex:
    """In-memory index of code chunks from a repository workspace."""

    def __init__(self, workspace_path: Path, chunks: list[CodeChunk]) -> None:
        self.workspace_path = workspace_path
        self.chunks = chunks

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def indexed_files(self) -> set[str]:
        return {c.file_path for c in self.chunks}


class CodebaseIndexer:
    """Indexes files in a workspace into a CodebaseIndex."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        chunker: CodeChunker | None = None,
    ) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.chunker = chunker or CodeChunker()

    def index_workspace(self, workspace_path: Path) -> CodebaseIndex:
        """Scan and chunk all eligible files in the workspace."""
        files = self.workspace_manager.discover_files(workspace_path)
        chunks: list[CodeChunk] = []

        for file_path in files:
            rel_path = str(file_path.relative_to(workspace_path)).replace("\\", "/")
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as err:
                logger.warning("Failed to read %s: %s", file_path, err)
                continue

            if file_path.suffix.lower() == ".py":
                file_chunks = self.chunker.chunk_python(content, rel_path)
            else:
                file_chunks = self.chunker.chunk_text_file(content, rel_path)

            chunks.extend(file_chunks)

        logger.info(
            "Indexed workspace %s: %d files, %d chunks",
            workspace_path,
            len(files),
            len(chunks),
        )
        return CodebaseIndex(workspace_path=workspace_path, chunks=chunks)
