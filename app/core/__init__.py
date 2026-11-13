# app.core package
#
# Contains cross-cutting infrastructure used by every other layer.
# Nothing in here should import from app.api, app.adapters, or app.services —
# dependencies only flow inward toward this package, never outward.
#
# Modules:
#   config.py  — Pydantic Settings: load, validate and expose all env vars

