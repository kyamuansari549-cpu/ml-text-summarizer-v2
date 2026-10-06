"""FastAPI backend for the abstractive text summarizer.

Run:
    uvicorn main:app --reload --port 8000

Env:
    MODEL_NAME       HF model id (default: facebook/bart-large-cnn)
    MAX_INPUT_CHARS  input cap to avoid OOM (default: 10000)
"""

from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ocr_service import ALLOWED_TYPES, MAX_IMAGE_BYTES, extract_text
from summarizer_service import get_summarizer, model_info, summarize_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup so the first request isn't slow.
    get_summarizer()
    yield


app = FastAPI(title="Text Summarizer API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    max_length: int = Field(default=150, ge=10, le=500)
    min_length: int = Field(default=30, ge=5, le=200)


class SummarizeResponse(BaseModel):
    summary: str
    original_words: int
    summary_words: int
    model: str


class SummarizeImageResponse(SummarizeResponse):
    extracted_text: str


@app.get("/api/health")
def health():
    return {"status": "ok", **model_info()}


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if req.min_length >= req.max_length:
        raise HTTPException(
            status_code=422, detail="min_length must be smaller than max_length"
        )
    try:
        summary = summarize_text(req.text, req.max_length, req.min_length)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # model errors -> 500, never leak internals
        raise HTTPException(status_code=500, detail=f"Summarization failed: {type(exc).__name__}")
    return SummarizeResponse(
        summary=summary,
        original_words=len(req.text.split()),
        summary_words=len(summary.split()),
        model=model_info()["model"],
    )


@app.post("/api/summarize-image", response_model=SummarizeImageResponse)
async def summarize_image(
    file: UploadFile = File(...),
    max_length: int = Form(150),
    min_length: int = Form(30),
    lang: str = Form("eng"),
):
    """OCR an uploaded photo (JPG/PNG/WebP), then summarize the text found."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422, detail="Only JPG, PNG or WebP images are supported"
        )
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")
    if min_length >= max_length:
        raise HTTPException(
            status_code=422, detail="min_length must be smaller than max_length"
        )
    try:
        text = extract_text(data, lang=lang)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not text:
        raise HTTPException(status_code=422, detail="No text found in the image")
    try:
        summary = summarize_text(text, max_length, min_length)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()  # full error lands in the uvicorn console
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {type(exc).__name__}: {exc}",
        )
    return SummarizeImageResponse(
        summary=summary,
        original_words=len(text.split()),
        summary_words=len(summary.split()),
        model=model_info()["model"],
        extracted_text=text,
    )
