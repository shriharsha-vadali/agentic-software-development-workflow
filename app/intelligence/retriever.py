"""
app/intelligence/retriever.py
-----------------------------
Codebase context retrieval using hybrid lexical scoring, symbol matching,
and path relevance weighting.
"""

from __future__ import annotations

import math
import re

from app.intelligence.chunker import CodeChunk
from app.intelligence.indexer import CodebaseIndex

_TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
_TEST_HINTS = {"test", "tests", "testing", "assert", "failure", "fail", "pytest", "error"}
_CONFIG_HINTS = {"config", "dependency", "package", "install", "requirement", "settings", "env"}


class CodebaseRetriever:
    """Retrieves the most relevant code chunks for an issue query."""

    def __init__(self, index: CodebaseIndex) -> None:
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        min_score: float = 0.5,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Rank and return top-k chunks matching the query string.

        Parameters
        ----------
        query : str
            Combined issue title, body, and comments.
        top_k : int
            Maximum number of chunks to return.
        min_score : float
            Minimum relevance threshold.

        Returns
        -------
        list[tuple[CodeChunk, float]]
            List of (chunk, score) tuples sorted by descending score.
        """
        query_tokens = set(_TOKEN_PATTERN.findall(query.lower()))
        if not query_tokens:
            return [(chunk, 1.0) for chunk in self.index.chunks[:top_k]]

        query_wants_tests = bool(query_tokens & _TEST_HINTS)
        query_wants_config = bool(query_tokens & _CONFIG_HINTS)

        scored: list[tuple[CodeChunk, float]] = []

        for chunk in self.index.chunks:
            # Overlap between query tokens and chunk keywords
            overlap = query_tokens & chunk.keywords
            if not overlap:
                continue

            score = float(len(overlap))

            # 1. Symbol bonus: if symbol name matches any query token directly
            if chunk.symbol_name:
                sym_lower = chunk.symbol_name.lower()
                for q in query_tokens:
                    if q in sym_lower:
                        score += 3.0

            # 2. File path bonus: if filename or directory is mentioned in query
            path_lower = chunk.file_path.lower()
            for q in query_tokens:
                if q in path_lower:
                    score += 2.0

            # 3. Test priority bonus
            if query_wants_tests and chunk.kind == "test":
                score += 2.5

            # 4. Config priority bonus
            if query_wants_config and chunk.kind == "config":
                score += 2.0

            # Normalize slightly by chunk length to avoid favoring huge chunks purely by token count
            length_penalty = math.log1p(max(1, chunk.end_line - chunk.start_line + 1))
            final_score = score / (1.0 + 0.1 * length_penalty)

            if final_score >= min_score:
                scored.append((chunk, round(final_score, 3)))

        # Sort descending by score, then ascending by file_path and start_line
        scored.sort(key=lambda item: (-item[1], item[0].file_path, item[0].start_line))

        return scored[:top_k]
