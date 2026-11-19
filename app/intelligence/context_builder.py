"""
app/intelligence/context_builder.py
-----------------------------------
Assembles retrieved codebase chunks and repository metadata into structured
markdown context for LLM planning and code generation.
"""

from __future__ import annotations

from typing import Any

from app.intelligence.chunker import CodeChunk

DEFAULT_MAX_CONTEXT_CHARS: int = 24_000


class ContextBuilder:
    """Formats retrieved code chunks and repo metadata into LLM prompts."""

    def __init__(self, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> None:
        self.max_context_chars = max_context_chars

    def build_context(
        self,
        ranked_chunks: list[tuple[CodeChunk, float]],
        project_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Assemble chunks into Markdown format, grouping by file and adhering to budget.
        """
        lines: list[str] = ["### Codebase Context\n"]

        if project_info:
            lines.append("#### Project Summary")
            lines.append(
                f"- **Primary Language**: {project_info.get('primary_language', 'unknown')}"
            )
            if project_info.get("languages"):
                lines.append(f"- **Languages**: {', '.join(project_info['languages'])}")
            if project_info.get("config_files"):
                lines.append(f"- **Configs**: {', '.join(project_info['config_files'])}")
            if project_info.get("test_frameworks"):
                lines.append(f"- **Test Frameworks**: {', '.join(project_info['test_frameworks'])}")
            lines.append("")

        if not ranked_chunks:
            lines.append(
                "*(No specific code chunks matched the issue query. "
                "Examine repository structure.)*\n"
            )
            return "\n".join(lines)

        lines.append("#### Relevant Source Chunks\n")

        # Group chunks by file path while preserving top-scoring order
        seen_files: dict[str, list[CodeChunk]] = {}
        for chunk, _ in ranked_chunks:
            seen_files.setdefault(chunk.file_path, []).append(chunk)

        current_chars = len("\n".join(lines))

        for file_path, chunks in seen_files.items():
            file_header = f"##### File: `{file_path}`\n"
            file_lines = [file_header]

            for c in chunks:
                chunk_header = (
                    f"```{c.file_path.split('.')[-1] if '.' in c.file_path else ''} // {c.header}"
                )
                chunk_block = f"{chunk_header}\n{c.content}\n```\n"
                if current_chars + len(chunk_block) > self.max_context_chars:
                    lines.append("*(Truncated remaining chunks due to context budget)*\n")
                    return "\n".join(lines)

                file_lines.append(chunk_block)
                current_chars += len(chunk_block)

            lines.extend(file_lines)

        return "\n".join(lines)
