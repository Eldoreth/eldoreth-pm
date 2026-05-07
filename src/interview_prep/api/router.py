"""FastAPI router exposing the process-mining modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from interview_prep.mining.discovery import (
    Bottleneck,
    ProcessModel,
    Variant,
    discover_process,
    get_bottlenecks,
    get_variants,
)
from interview_prep.mining.parser import parse_csv, parse_xes

router = APIRouter(prefix="/api/v1", tags=["process-mining"])

_SUPPORTED = {".xes", ".csv"}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

async def _parse_upload(file: UploadFile) -> pd.DataFrame:
    """Write an uploaded file to a tempfile, parse it, return a DataFrame.

    Raises:
        HTTPException 400: unsupported extension or parse failure.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or '(none)'}'. "
                "Upload a .xes or .csv event log."
            ),
        )

    content = await file.read()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)
        if suffix == ".xes":
            df = parse_xes(tmp_path)
        else:
            df = parse_csv(tmp_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/discover",
    response_model=ProcessModel,
    summary="Discover a process model",
    description=(
        "Upload an XES or CSV event log to discover a Petri-net process model. "
        "Choose `algorithm` from **inductive** (default), **alpha**, or **heuristics**."
    ),
)
async def discover(
    file: UploadFile = File(..., description="XES or CSV event log"),
    algorithm: str = Query(
        default="inductive",
        description="Discovery algorithm: inductive | alpha | heuristics",
    ),
) -> ProcessModel:
    df = await _parse_upload(file)
    try:
        return discover_process(df, algorithm=algorithm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/variants",
    response_model=list[Variant],
    summary="Extract trace variants",
    description=(
        "Upload an XES or CSV event log to get all unique trace variants "
        "with frequency and average case duration."
    ),
)
async def variants(
    file: UploadFile = File(..., description="XES or CSV event log"),
) -> list[Variant]:
    df = await _parse_upload(file)
    return get_variants(df)


@router.post(
    "/bottlenecks",
    response_model=list[Bottleneck],
    summary="Identify process bottlenecks",
    description=(
        "Upload an XES or CSV event log to get the top-N activity bottlenecks "
        "ranked by average wait time. Each result includes wait time, frequency, "
        "and rework rate."
    ),
)
async def bottlenecks(
    file: UploadFile = File(..., description="XES or CSV event log"),
    top_n: int = Query(default=5, ge=1, description="Number of bottlenecks to return"),
) -> list[Bottleneck]:
    df = await _parse_upload(file)
    return get_bottlenecks(df, top_n=top_n)
