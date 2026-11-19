"""
app/intelligence/coder.py
-------------------------
AI Code Generation Engine.
Takes an approved implementation plan, generates validated file modifications,
enforces workspace security boundaries, applies changes, and generates diffs.
"""

from __future__ import annotations

import difflib
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.llm import BaseLLMClient
from app.intelligence.planner import ImplementationPlan
from app.intelligence.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class FileChange(BaseModel):
    """An individual file creation, modification, or deletion."""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(..., description="Relative file path within workspace.")
    action: Literal["create", "modify", "delete"] = Field(
        ..., description="Action to perform on the target file."
    )
    content: str = Field(
        default="", description="New or updated content for create/modify actions."
    )


class CodeGenerationResult(BaseModel):
    """Structured code generation output from the LLM."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(..., description="Summary of changes implemented.")
    changes: list[FileChange] = Field(
        default_factory=list, description="List of file changes to apply."
    )


SYSTEM_CODER_PROMPT = """You are a Principal Software Engineer.
Your task is to implement an approved plan by producing precise, minimal code changes.

CRITICAL SECURITY & INTEGRITY RULES:
1. Only produce code for files inside the project workspace. Never reference absolute
   paths or parent traversal (../).
2. For "create" or "modify", provide the COMPLETE, WORKING file content.
3. Follow the project's existing code style, naming conventions, and imports.
4. Write thorough tests for any new or modified functionality.
5. Return ONLY structured output matching the provided schema.
"""


class CodeGenerator:
    """Orchestrates LLM code generation, path validation, and file writing."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        workspace_manager: WorkspaceManager | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.workspace_manager = workspace_manager or WorkspaceManager()

    def build_prompt(
        self,
        plan: ImplementationPlan,
        codebase_context: str,
        repair_context: str | None = None,
    ) -> str:
        """Assemble the prompt for code generation."""
        steps_text = "\n".join(
            f"{s.step_number}. {s.description} (Target: {', '.join(s.file_paths)})"
            for s in plan.steps
        )

        repair_section = ""
        if repair_context:
            repair_section = (
                f"\n### Test Failure Context / Repair Required\n"
                f"The previous implementation failed automated tests:\n"
                f"```\n{repair_context}\n```\n"
                f"Analyze the failures and provide the necessary corrections.\n"
            )

        files_create = ", ".join(plan.files_to_create) if plan.files_to_create else "None"
        files_modify = ", ".join(plan.files_to_modify) if plan.files_to_modify else "None"

        return (
            f"### Approved Implementation Plan\n"
            f"- **Summary**: {plan.summary}\n"
            f"- **Problem**: {plan.problem_statement}\n"
            f"- **Technical Approach**: {plan.technical_approach}\n"
            f"- **Files to Create**: {files_create}\n"
            f"- **Files to Modify**: {files_modify}\n"
            f"- **Steps**:\n{steps_text}\n"
            f"- **Test Plan**: {plan.test_plan}\n"
            f"{repair_section}\n"
            f"{codebase_context}\n\n"
            f"Generate all necessary file changes to implement this plan."
        )

    async def generate_code(
        self,
        plan: ImplementationPlan,
        codebase_context: str,
        repair_context: str | None = None,
    ) -> CodeGenerationResult:
        """Call LLM to produce structured code changes."""
        prompt = self.build_prompt(plan, codebase_context, repair_context=repair_context)
        return await self.llm_client.generate_structured(
            prompt=prompt,
            response_model=CodeGenerationResult,
            system_prompt=SYSTEM_CODER_PROMPT,
        )

    def validate_and_apply_changes(
        self,
        job_id: int,
        changes: list[FileChange],
    ) -> tuple[list[str], str]:
        """
        Validate paths and apply changes inside the job's isolated workspace.

        Returns
        -------
        tuple[list[str], str]
            List of modified file paths and the unified diff.
        """
        workspace = self.workspace_manager.get_job_workspace(job_id)
        if not workspace.exists():
            raise FileNotFoundError(f"Workspace for job {job_id} does not exist at {workspace}")

        modified_files: list[str] = []
        diff_lines: list[str] = []

        for change in changes:
            clean_rel = change.path.replace("\\", "/").strip().lstrip("/")
            if clean_rel.startswith(".git") or ".." in clean_rel:
                raise ValueError(f"Prohibited path access attempt: {change.path}")

            target_path = self.workspace_manager.resolve_safe_path(job_id, clean_rel)
            rel_display = str(target_path.relative_to(workspace)).replace("\\", "/")

            old_content = ""
            if target_path.exists():
                old_content = target_path.read_text(encoding="utf-8", errors="replace")

            if change.action == "delete":
                if target_path.exists():
                    target_path.unlink()
                    logger.info("Deleted file %s in workspace %d", rel_display, job_id)
                    modified_files.append(rel_display)
                    # Unified diff for deletion
                    diff = difflib.unified_diff(
                        old_content.splitlines(keepends=True),
                        [],
                        fromfile=f"a/{rel_display}",
                        tofile="/dev/null",
                    )
                    diff_lines.extend(diff)
            elif change.action in ("create", "modify"):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                new_content = change.content
                # Normalize line endings to LF
                new_content_lf = new_content.replace("\r\n", "\n")
                target_path.write_text(new_content_lf, encoding="utf-8")
                logger.info(
                    "Wrote file %s in workspace %d (%d bytes)",
                    rel_display,
                    job_id,
                    len(new_content_lf),
                )
                modified_files.append(rel_display)

                # Unified diff
                diff = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content_lf.splitlines(keepends=True),
                    fromfile=f"a/{rel_display}",
                    tofile=f"b/{rel_display}",
                )
                diff_lines.extend(diff)

        full_diff = "".join(diff_lines)
        return modified_files, full_diff
