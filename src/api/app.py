import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import prometheus_client

from src.api.predictor import get_predictor

app = FastAPI(
    title="LeafLens API",
    description="Plant disease classifier powered by CLIP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus metrics ---
Instrumentator().instrument(app).expose(app)

PREDICTION_COUNTER = Counter(
    "leaflens_predictions_total",
    "Total predictions made",
    ["predicted_class"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    "leaflens_confidence_score",
    "Confidence score distribution",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

INFERENCE_LATENCY = Histogram(
    "leaflens_inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=[100, 200, 500, 1000, 2000, 5000]
)

LOW_CONFIDENCE_COUNTER = Counter(
    "leaflens_low_confidence_predictions_total",
    "Predictions with confidence below 50%"
)


# --- Response schema ---
class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    top_k: list
    latency_ms: float


# --- Routes ---
@app.get("/")
def root():
    return {"message": "LeafLens API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/classes")
def get_classes():
    predictor = get_predictor()
    return {
        "total": len(predictor.classes),
        "classes": [c.replace("_", " ") for c in predictor.classes]
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    start = time.time()
    image_bytes = await file.read()

    try:
        predictor = get_predictor()
        result = predictor.predict(image_bytes, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 2)
    result["latency_ms"] = latency_ms

    # Record metrics
    PREDICTION_COUNTER.labels(predicted_class=result["prediction"]).inc()
    CONFIDENCE_HISTOGRAM.observe(result["confidence"])
    INFERENCE_LATENCY.observe(latency_ms)
    if result["confidence"] < 50:
        LOW_CONFIDENCE_COUNTER.inc()

    return result