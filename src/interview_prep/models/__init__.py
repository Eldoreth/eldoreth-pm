"""Pydantic domain models for API response contracts."""

from __future__ import annotations

from pydantic import BaseModel


class InsightReport(BaseModel):
    """AI-generated process mining insight report."""

    executive_summary: str
    top_findings: list[str]
    recommended_actions: list[str]
    estimated_value: str
