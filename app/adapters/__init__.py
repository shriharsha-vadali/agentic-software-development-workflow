# app.adapters package
#
# Contains adapters for all external services.
# Each adapter encapsulates one external dependency so the rest of the
# application never imports httpx, the GitHub SDK, or the OpenAI SDK directly.
#
#   github.py  — GitHub REST API client (fetch issues, clone repos, open PRs)
#   llm.py     — LLM service interface and OpenAI-compatible implementation

