from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _split_origins(value: str) -> list[str]:
    origins: list[str] = []
    for raw_origin in value.split(","):
        origin = raw_origin.strip().rstrip("/")
        if origin:
            origins.append(origin)
    return origins


@dataclass(slots=True)
class Settings:
    app_name: str = "WasmBoost API"
    api_prefix: str = "/api"
    cors_origins: list[str] = field(
        default_factory=lambda: _split_origins(
            os.getenv(
                "WASMBOOST_CORS_ORIGINS",
                os.getenv(
                    "DATAPILOT_CORS_ORIGINS",
                    "http://localhost:3000,http://127.0.0.1:3000",
                ),
            )
        )
    )
    cors_origin_regex: str = os.getenv(
        "WASMBOOST_CORS_ORIGIN_REGEX",
        os.getenv("DATAPILOT_CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app"),
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker_url: str = os.getenv(
        "CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    celery_result_backend: str = os.getenv(
        "CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    celery_task_always_eager: bool = os.getenv(
        "CELERY_TASK_ALWAYS_EAGER", "false"
    ).lower() in {"1", "true", "yes"}


settings = Settings()
