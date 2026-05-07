# Interview Prep — SAP AI / Signavio / LeanIX PM Role

Technical portfolio project built to demonstrate process mining and AI capabilities for the SAP Munich PM interview.

## Goals

1. **Process Mining** — ingest event logs, run discovery algorithms (Alpha, Heuristics, Inductive Miner via pm4py), and expose results via REST API.
2. **Conformance Checking** — compare real process traces against a reference model and surface deviations.
3. **AI-Assisted Insights** — integrate LLM calls to generate natural-language summaries of process bottlenecks and improvement suggestions.
4. **Demonstrable API** — FastAPI service that can be shown live or via Postman/Swagger during the interview.

## Stack

| Layer | Library |
|---|---|
| API | FastAPI 0.111+ |
| Data | pandas 2.x |
| Validation | pydantic v2 |
| Process Mining | pm4py 2.7+ |
| AI Insights | Anthropic Claude (anthropic SDK 0.40+) |
| Testing | pytest + pytest-cov |
| Runtime | Python 3.11 |

## Project Structure

```
interview-prep/
├── CLAUDE.md                   # AI assistant instructions
├── README.md                   # This file
├── pyproject.toml              # Dependencies and tool config
├── src/
│   └── interview_prep/
│       ├── __init__.py
│       ├── api/                # FastAPI routers
│       │   ├── __init__.py
│       │   └── router.py       # /discover, /variants, /bottlenecks, /insights endpoints
│       ├── main.py             # FastAPI app (CORS, /docs, router mount)
│       ├── models/             # Pydantic domain models
│       │   └── __init__.py     # InsightReport response model
│       ├── services/           # Business logic layer
│       │   └── __init__.py     # InsightReport response model
│       └── mining/             # pm4py wrappers and utilities
│           ├── __init__.py
│           ├── parser.py       # XES/CSV event log parser + validation
│           └── discovery.py    # Process discovery, variant analysis, bottlenecks
└── tests/
    ├── data/
    │   └── sample_orders.xes   # Bundled 20-case fixture for tests
    └── interview_prep/
        ├── test_parser.py      # 31 tests, 97% coverage
        ├── test_discovery.py   # 33 tests, 100% coverage on discovery.py
        └── test_api.py         # 36 tests — includes 3 mocked Anthropic tests
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn interview_prep.api.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

## Running Tests

```bash
pytest --cov=src --cov-fail-under=70
```

## Key Demos

| Demo | Description |
|---|---|
| `/api/v1/discover` | Upload an XES/CSV event log and get a discovered process model |
| `/api/v1/variants` | List all trace variants with frequency and average duration |
| `/api/v1/bottlenecks` | Top-N activity bottlenecks with wait time and rework rate |
| `/api/v1/conformance` | Check traces against a reference BPMN model |
| `/api/v1/insights` | Upload XES/CSV → runs bottleneck + variant analysis → returns an AI executive insight report (requires ANTHROPIC_API_KEY) |
