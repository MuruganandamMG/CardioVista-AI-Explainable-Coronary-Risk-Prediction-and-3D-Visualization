"""Local inference API serving one frozen bundle."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .predict import load_bundle, predict_record
from .schema import InputError


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: dict[str, Any]
    include_explanations: bool = False


def create_app(artifact_dir: Path | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        if artifact_dir is None:
            raise RuntimeError("Set CARDIO_ARTIFACT_DIR to a trusted saved bundle directory")
        application.state.bundle = load_bundle(artifact_dir)
        yield

    application = FastAPI(title="CardioVista AI", version="0.1.0", lifespan=lifespan)

    @application.get("/health")
    def health():
        bundle = application.state.bundle
        return {"status": "ready", "model_version": bundle["metadata"]["model_version"], "artifact_hash": bundle["artifact_hash"]}

    @application.get("/schema")
    def schema():
        return application.state.bundle["schema"]

    @application.post("/predict")
    def predict(request: PredictionRequest):
        try:
            return predict_record(application.state.bundle, request.features, request.include_explanations)
        except InputError as error:
            raise HTTPException(status_code=422, detail=error.errors) from error

    return application


app = create_app(Path(os.environ["CARDIO_ARTIFACT_DIR"]) if os.environ.get("CARDIO_ARTIFACT_DIR") else None)
