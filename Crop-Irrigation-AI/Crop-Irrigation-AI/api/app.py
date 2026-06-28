"""
api/app.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application factory.
─────────────────────────────────────────────────────────────────────────────
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils import logger
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.project_name} v{settings.project_version}")
    # Pre-warm model singletons here if needed
    yield
    logger.info("Shutting down API server")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        version=settings.project_version,
        description=(
            "AI-powered satellite-based crop irrigation advisory system. "
            "Fuses Sentinel-1 SAR, Sentinel-2 optical, and ERA5 weather data "
            "to generate zone-level irrigation recommendations."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_handler(request, exc):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()
