"""
app.intelligence package
------------------------
Codebase understanding engine: workspace isolation, AST parsing,
chunking, indexing, retrieval (RAG), and context building.
"""

from app.intelligence.chunker import CodeChunk, CodeChunker
from app.intelligence.coder import CodeGenerationResult, CodeGenerator, FileChange
from app.intelligence.context_builder import ContextBuilder
from app.intelligence.indexer import CodebaseIndex, CodebaseIndexer
from app.intelligence.parser import ParsedModule, PythonASTParser, SymbolInfo
from app.intelligence.planner import ImplementationPlan, ImplementationStep, PlanningEngine
from app.intelligence.retriever import CodebaseRetriever
from app.intelligence.workspace import WorkspaceManager

__all__ = [
    "CodeChunk",
    "CodeChunker",
    "CodeGenerationResult",
    "CodeGenerator",
    "CodebaseIndex",
    "CodebaseIndexer",
    "CodebaseRetriever",
    "ContextBuilder",
    "FileChange",
    "ImplementationPlan",
    "ImplementationStep",
    "ParsedModule",
    "PlanningEngine",
    "PythonASTParser",
    "SymbolInfo",
    "WorkspaceManager",
]
