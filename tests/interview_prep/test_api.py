"""Tests for the FastAPI process-mining endpoints."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from interview_prep.main import app

SAMPLE_XES = Path(__file__).parent.parent / "data" / "sample_orders.xes"

_CSV_CONTENT = textwrap.dedent("""\
    case_id,activity,timestamp,resource
    C1,Create Order,2024-03-01 09:00:00,Alice
    C1,Approve Order,2024-03-01 11:00:00,Bob
    C1,Ship Order,2024-03-01 15:00:00,Carol
    C2,Create Order,2024-03-02 09:00:00,Bob
    C2,Approve Order,2024-03-02 12:00:00,Alice
    C2,Ship Order,2024-03-02 17:00:00,Bob
    C3,Create Order,2024-03-03 08:00:00,Carol
    C3,Approve Order,2024-03-03 10:00:00,Bob
    C3,Ship Order,2024-03-03 14:00:00,Alice
""")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def xes_bytes() -> bytes:
    return SAMPLE_XES.read_bytes()


@pytest.fixture(scope="module")
def csv_bytes() -> bytes:
    return _CSV_CONTENT.encode()


def _xes_file(data: bytes) -> dict:
    return {"file": ("log.xes", data, "application/xml")}


def _csv_file(data: bytes) -> dict:
    return {"file": ("log.csv", data, "text/csv")}


# ---------------------------------------------------------------------------
# POST /api/v1/discover
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_xes_returns_200(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post("/api/v1/discover", files=_xes_file(xes_bytes))
        assert r.status_code == 200

    def test_xes_response_shape(self, client: TestClient, xes_bytes: bytes) -> None:
        body = client.post("/api/v1/discover", files=_xes_file(xes_bytes)).json()
        assert set(body) == {"algorithm", "num_activities", "num_transitions", "fitness_score"}

    def test_xes_default_algorithm(self, client: TestClient, xes_bytes: bytes) -> None:
        body = client.post("/api/v1/discover", files=_xes_file(xes_bytes)).json()
        assert body["algorithm"] == "inductive"

    def test_xes_fitness_score_is_one(self, client: TestClient, xes_bytes: bytes) -> None:
        body = client.post("/api/v1/discover", files=_xes_file(xes_bytes)).json()
        assert body["fitness_score"] == 1.0

    def test_xes_num_activities(self, client: TestClient, xes_bytes: bytes) -> None:
        body = client.post("/api/v1/discover", files=_xes_file(xes_bytes)).json()
        assert body["num_activities"] == 3  # Create / Approve / Ship

    def test_alpha_algorithm_query_param(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post("/api/v1/discover", files=_xes_file(xes_bytes), params={"algorithm": "alpha"})
        assert r.status_code == 200
        assert r.json()["algorithm"] == "alpha"

    def test_heuristics_algorithm_query_param(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post("/api/v1/discover", files=_xes_file(xes_bytes), params={"algorithm": "heuristics"})
        assert r.status_code == 200
        assert r.json()["algorithm"] == "heuristics"

    def test_csv_returns_200(self, client: TestClient, csv_bytes: bytes) -> None:
        r = client.post("/api/v1/discover", files=_csv_file(csv_bytes))
        assert r.status_code == 200

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        r = client.post("/api/v1/discover")
        assert r.status_code == 422

    def test_unsupported_extension_returns_400(self, client: TestClient) -> None:
        r = client.post("/api/v1/discover", files={"file": ("log.txt", b"data", "text/plain")})
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]

    def test_unknown_algorithm_returns_400(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post(
            "/api/v1/discover",
            files=_xes_file(xes_bytes),
            params={"algorithm": "magic"},
        )
        assert r.status_code == 400
        assert "Unknown algorithm" in r.json()["detail"]

    def test_malformed_csv_returns_400(self, client: TestClient) -> None:
        bad_csv = b"col_a,col_b\n1,foo\n2,bar\n"
        r = client.post("/api/v1/discover", files=_csv_file(bad_csv))
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/v1/variants
# ---------------------------------------------------------------------------

class TestVariants:
    def test_xes_returns_200(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post("/api/v1/variants", files=_xes_file(xes_bytes))
        assert r.status_code == 200

    def test_returns_list(self, client: TestClient, xes_bytes: bytes) -> None:
        body = client.post("/api/v1/variants", files=_xes_file(xes_bytes)).json()
        assert isinstance(body, list)

    def test_variant_fields(self, client: TestClient, xes_bytes: bytes) -> None:
        variants = client.post("/api/v1/variants", files=_xes_file(xes_bytes)).json()
        assert len(variants) >= 1
        v = variants[0]
        assert set(v) == {"activities", "frequency", "avg_duration_seconds"}

    def test_single_variant_linear_log(self, client: TestClient, xes_bytes: bytes) -> None:
        variants = client.post("/api/v1/variants", files=_xes_file(xes_bytes)).json()
        assert len(variants) == 1
        assert variants[0]["frequency"] == 20

    def test_activities_is_list(self, client: TestClient, xes_bytes: bytes) -> None:
        v = client.post("/api/v1/variants", files=_xes_file(xes_bytes)).json()[0]
        assert isinstance(v["activities"], list)
        assert v["activities"] == ["Create Order", "Approve Order", "Ship Order"]

    def test_csv_returns_200(self, client: TestClient, csv_bytes: bytes) -> None:
        r = client.post("/api/v1/variants", files=_csv_file(csv_bytes))
        assert r.status_code == 200

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        r = client.post("/api/v1/variants")
        assert r.status_code == 422

    def test_unsupported_extension_returns_400(self, client: TestClient) -> None:
        r = client.post("/api/v1/variants", files={"file": ("log.json", b"{}", "application/json")})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/v1/bottlenecks
# ---------------------------------------------------------------------------

class TestBottlenecks:
    def test_xes_returns_200(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post("/api/v1/bottlenecks", files=_xes_file(xes_bytes))
        assert r.status_code == 200

    def test_returns_list(self, client: TestClient, xes_bytes: bytes) -> None:
        body = client.post("/api/v1/bottlenecks", files=_xes_file(xes_bytes)).json()
        assert isinstance(body, list)

    def test_bottleneck_fields(self, client: TestClient, xes_bytes: bytes) -> None:
        bns = client.post("/api/v1/bottlenecks", files=_xes_file(xes_bytes)).json()
        assert len(bns) >= 1
        b = bns[0]
        assert set(b) == {"activity", "avg_wait_time_seconds", "frequency", "rework_rate"}

    def test_default_top_n_at_most_5(self, client: TestClient, xes_bytes: bytes) -> None:
        bns = client.post("/api/v1/bottlenecks", files=_xes_file(xes_bytes)).json()
        assert len(bns) <= 5

    def test_custom_top_n(self, client: TestClient, xes_bytes: bytes) -> None:
        bns = client.post(
            "/api/v1/bottlenecks", files=_xes_file(xes_bytes), params={"top_n": 1}
        ).json()
        assert len(bns) == 1

    def test_sorted_by_wait_time_desc(self, client: TestClient, xes_bytes: bytes) -> None:
        bns = client.post("/api/v1/bottlenecks", files=_xes_file(xes_bytes)).json()
        waits = [b["avg_wait_time_seconds"] for b in bns]
        assert waits == sorted(waits, reverse=True)

    def test_csv_returns_200(self, client: TestClient, csv_bytes: bytes) -> None:
        r = client.post("/api/v1/bottlenecks", files=_csv_file(csv_bytes))
        assert r.status_code == 200

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        r = client.post("/api/v1/bottlenecks")
        assert r.status_code == 422

    def test_invalid_top_n_returns_422(self, client: TestClient, xes_bytes: bytes) -> None:
        # top_n=0 violates ge=1 constraint → FastAPI returns 422
        r = client.post(
            "/api/v1/bottlenecks", files=_xes_file(xes_bytes), params={"top_n": 0}
        )
        assert r.status_code == 422

    def test_unsupported_extension_returns_400(self, client: TestClient) -> None:
        r = client.post("/api/v1/bottlenecks", files={"file": ("log.parquet", b"data", "application/octet-stream")})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# App-level
# ---------------------------------------------------------------------------

class TestApp:
    def test_docs_available(self, client: TestClient) -> None:
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema_has_three_endpoints(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        paths = list(schema["paths"].keys())
        assert "/api/v1/discover" in paths
        assert "/api/v1/variants" in paths
        assert "/api/v1/bottlenecks" in paths

    def test_cors_header_present(self, client: TestClient, xes_bytes: bytes) -> None:
        r = client.post(
            "/api/v1/discover",
            files=_xes_file(xes_bytes),
            headers={"Origin": "http://localhost:3000"},
        )
        assert r.headers.get("access-control-allow-origin") == "*"
