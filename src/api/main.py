"""
FastAPI application — Fake News Detection API + dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import torch

from .inference import predict, get_metrics

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(
    title="Fake News Detection API",
    description="Multilingual ensemble: LightGBM + XLM-RoBERTa",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ── Schemas ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000,
                      description="News article text to classify")


# ── Routes ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/metrics")
async def metrics():
    try:
        return get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def classify(req: PredictRequest):
    try:
        return predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
