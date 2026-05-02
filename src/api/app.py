import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    # Validate file type
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

    result["latency_ms"] = round((time.time() - start) * 1000, 2)
    return result