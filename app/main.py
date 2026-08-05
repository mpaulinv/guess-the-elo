# app/main.py - FastAPI prediction API for the bracket-MoE Elo predictor
"""
Endpoints:
  GET  /health          -> {"status": "ok"} once the model is loaded
  POST /predict          -> multipart file upload (a .pgn file), field name "file"
  POST /predict/text     -> JSON body {"pgn": "<raw pgn text>"}
  GET  /docs             -> interactive OpenAPI docs (auto-generated, free with FastAPI)

Two separate routes instead of one overloaded endpoint (file-or-JSON) because
FastAPI can't cleanly mix an UploadFile param with a JSON-body Pydantic model
on the same endpoint - a mixed-type endpoint gets its other params treated as
form fields, not parsed as JSON. Splitting into two typed routes is also
just cleaner REST design than the Flask version's manual
"if file in request.files elif is_json" branching.

Run locally:
  "C:\\Users\\mario\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m uvicorn main:app --reload --app-dir app
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import EloPredictor, PGNParseError

# Vite's dev server default; the deployed frontend's real origin gets added
# once it exists (Cloud Run URL, custom domain, etc.) - kept as an explicit
# allowlist rather than "*" since credentials-less GET/POST from a browser
# to a public API is fine to allow broadly, but naming the known origins is
# still better practice than a wildcard once this goes past local dev.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", str(REPO_ROOT / "models" / "bracket_moe_gpu_epoch065.pt"))
VOCAB_PATH = os.environ.get("VOCAB_PATH", str(REPO_ROOT / "data" / "processed" / "nn_bracket_moe" / "vocab.json"))

predictor: Optional[EloPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # eager load at startup, not lazy-on-first-request: if the checkpoint is
    # missing/corrupt, fail loudly at deploy time instead of on a user's
    # first request. FastAPI's lifespan hook is the built-in place for this.
    global predictor
    print(f"Loading model from {CHECKPOINT_PATH} ...")
    predictor = EloPredictor(CHECKPOINT_PATH, VOCAB_PATH)
    print("Model loaded.")
    yield
    predictor = None


app = FastAPI(title="Chess Elo Predictor", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PGNRequest(BaseModel):
    pgn: str


class PredictionResponse(BaseModel):
    white_elo: int
    black_elo: int
    white_bucket_range: list[int]
    black_bucket_range: list[int]
    white_confidence: float
    black_confidence: float
    plies_used: int
    clock_coverage: float
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ok")


def _run_prediction(pgn_text: str) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        result = predictor.predict(pgn_text)
    except PGNParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictionResponse(**result)


@app.post("/predict", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    pgn_bytes = await file.read()
    return _run_prediction(pgn_bytes.decode("utf-8", errors="replace"))


@app.post("/predict/text", response_model=PredictionResponse)
def predict_text(body: PGNRequest):
    return _run_prediction(body.pgn)
