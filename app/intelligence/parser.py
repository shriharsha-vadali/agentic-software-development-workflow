"""
app/intelligence/parser.py
--------------------------
AST-based parsing of Python source code.
Extracts functions, classes, methods, imports, docstrings, and exact line spans.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Represents a code symbol (function, method, class) extracted from AST."""

    name: str
    kind: str  # "function", "async_function", "class", "method"
    start_line: int
    end_line: int
    docstring: str | None = None
    parent_class: str | None = None
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None


@dataclass
class ParsedModule:
    """Represents the parsed structure of a single Python source file."""

    file_path: str
    docstring: str | None = None
    imports: list[str] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    total_lines: int = 0
    parse_error: str | None = None


class PythonASTParser:
    """Parser using Python's standard `ast` module to analyze Python files."""

    def parse_source(self, code: str, file_path: str = "") -> ParsedModule:
        """Parse source code string into a structured ParsedModule."""
        lines = code.splitlines()
        total_lines = len(lines)

        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as err:
            logger.debug("Syntax error parsing %s: %s", file_path, err)
            return ParsedModule(
                file_path=file_path,
                total_lines=total_lines,
                parse_error=f"SyntaxError: {err.msg} at line {err.lineno}",
            )

        module_docstring = ast.get_docstring(tree)
        imports: list[str] = []
        symbols: list[SymbolInfo] = []

        for node in tree.body:
            # Collect imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    imports.append(f"{mod}.{alias.name}" if mod else alias.name)

            # Collect top-level functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                params = [arg.arg for arg in node.args.args]
                end_lineno = getattr(node, "end_lineno", node.lineno)
                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind=kind,
                        start_line=node.lineno,
                        end_line=end_lineno,
                        docstring=ast.get_docstring(node),
                        parameters=params,
                    )
                )

            # Collect classes and their methods
            elif isinstance(node, ast.ClassDef):
                class_end = getattr(node, "end_lineno", node.lineno)
                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="class",
                        start_line=node.lineno,
                        end_line=class_end,
                        docstring=ast.get_docstring(node),
                    )
                )
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_kind = (
                            "async_method" if isinstance(item, ast.AsyncFunctionDef) else "method"
                        )
                        method_params = [arg.arg for arg in item.args.args]
                        method_end = getattr(item, "end_lineno", item.lineno)
                        symbols.append(
                            SymbolInfo(
                                name=f"{node.name}.{item.name}",
                                kind=method_kind,
                                start_line=item.lineno,
                                end_line=method_end,
                                docstring=ast.get_docstring(item),
                                parent_class=node.name,
                                parameters=method_params,
                            )
                        )

        return ParsedModule(
            file_path=file_path,
            docstring=module_docstring,
            imports=imports,
            symbols=symbols,
            total_lines=total_lines,
        )

    def parse_file(self, path: Path, relative_to: Path | None = None) -> ParsedModule:
        """Read and parse a file from disk."""
        disp_path = str(path.relative_to(relative_to)) if relative_to else str(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            return ParsedModule(file_path=disp_path, parse_error=str(err))

        return self.parse_source(content, file_path=disp_path)
