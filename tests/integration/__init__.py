# tests.integration package
#
# Integration tests that exercise the full HTTP layer.
# Uses httpx.AsyncClient pointed at the FastAPI app with a real (in-memory)
# SQLite database, but with GitHub and LLM adapters mocked.

