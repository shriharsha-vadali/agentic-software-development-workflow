# app.adapters package
#
# Contains adapters for all external services.
# Each adapter encapsulates one external dependency so the rest of the
# application never imports httpx, the GitHub SDK, or the OpenAI SDK directly.
#
#   github.py  — GitHub REST API client (fetch issues, comments, repos)
#   llm.py     — LLM service interface and OpenAI-compatible implementation

from app.adapters.github import (
    GitHubAdapterError,
    GitHubAPIResponseError,
    GitHubAuthError,
    GitHubBranch,
    GitHubClient,
    GitHubClientProtocol,
    GitHubComment,
    GitHubIssue,
    GitHubLabel,
    GitHubNetworkError,
    GitHubPermissionError,
    GitHubPullRequest,
    GitHubRateLimitError,
    GitHubRepository,
    GitHubResourceNotFoundError,
    GitHubUser,
    GitHubValidationError,
    get_github_client,
    validate_repo_params,
)
from app.adapters.llm import (
    BaseLLMClient,
    LLMAdapterError,
    LLMAuthError,
    LLMResponseError,
    LLMTimeoutError,
    MockLLMClient,
    OpenAILLMClient,
    get_llm_client,
)

__all__ = [
    "BaseLLMClient",
    "GitHubAdapterError",
    "GitHubAPIResponseError",
    "GitHubAuthError",
    "GitHubBranch",
    "GitHubClient",
    "GitHubClientProtocol",
    "GitHubComment",
    "GitHubIssue",
    "GitHubLabel",
    "GitHubNetworkError",
    "GitHubPermissionError",
    "GitHubPullRequest",
    "GitHubRateLimitError",
    "GitHubRepository",
    "GitHubResourceNotFoundError",
    "GitHubUser",
    "GitHubValidationError",
    "validate_repo_params",
    "LLMAdapterError",
    "LLMAuthError",
    "LLMResponseError",
    "LLMTimeoutError",
    "MockLLMClient",
    "OpenAILLMClient",
    "get_github_client",
    "get_llm_client",
]
