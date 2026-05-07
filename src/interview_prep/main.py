"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interview_prep.api.router import router

app = FastAPI(
    title="Process Mining API",
    description=(
        "Upload XES or CSV event logs to discover process models, "
        "extract trace variants, and identify bottlenecks. "
        "Built for the SAP AI / Signavio / LeanIX PM interview demo."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
