# CLAUDE.md

## Project

Interview preparation project for SAP AI / Signavio / LeanIX PM role at SAP Munich.
Goal: build demonstrable process mining + AI tools to showcase during technical interview.

## Stack

- Python 3.11
- FastAPI — REST API layer
- pandas — data manipulation
- pydantic v2 — data validation and settings
- pm4py — process mining algorithms (discovery, conformance, enhancement)

## Code Style

- Type hints are **mandatory** on all function signatures and class attributes.
- Use pydantic models for all data contracts (request/response bodies, config, domain objects).
- Docstrings in English; one-line summary + optional extended description. No redundant comments.
- No comments that describe *what* code does — only *why* when non-obvious.
- Default to no comments; only add when there is a hidden constraint or surprising behaviour.

## Testing

- Framework: pytest
- Minimum coverage: 70% (enforced via `pytest --cov` + `--cov-fail-under=70`).
- Test files live in `tests/` mirroring the `src/` layout.
- Prefer integration tests over mocks where feasible.

## File Creation Rule

Whenever a new source file is created, update `README.md` to reflect the change in project structure or goals.

## Project Layout

```
interview-prep/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/
│   └── interview_prep/
│       ├── api/          # FastAPI routers
│       ├── models/       # Pydantic domain models
│       ├── services/     # Business logic
│       └── mining/       # pm4py wrappers
└── tests/
    └── interview_prep/
```
