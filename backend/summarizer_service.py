"""Abstractive summarization service built on Hugging Face transformers.

Default model is BART (facebook/bart-large-cnn) -- the classic
abstractive summarizer. Set MODEL_NAME=t5-small (or t5-base) to use T5
instead; T5 needs a "summarize: " task prefix, which is handled here.

Note: the model is loaded directly (AutoModelForSeq2SeqLM) instead of via
transformers' pipeline("summarization"), because newer transformers versions
removed the 'summarization' pipeline task entirely -- and the 'text-generation'
fallback loads BART as a decoder-only model that just echoes the input.
"""

import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-cnn")

# BART caps input at 1024 tokens; truncate longer texts gracefully.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "10000"))
MAX_INPUT_TOKENS = 1024

_tokenizer = None
_model = None


def _is_t5():
    return "t5" in MODEL_NAME.lower()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def get_model():
    global _model
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model


def get_summarizer():
    """Load model + tokenizer once (used as the startup warmup hook)."""
    get_tokenizer()
    get_model()
    return True


def summarize_text(text, max_length=150, min_length=30):
    """Return an abstractive summary of *text*.

    Unlike the v1 TF-IDF version, this generates NEW sentences instead of
    picking existing ones.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is empty")
    text = text[:MAX_INPUT_CHARS]

    # T5 is a text-to-text model: it needs to be told the task.
    model_input = f"summarize: {text}" if _is_t5() else text

    tokenizer = get_tokenizer()
    model = get_model()

    inputs = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,   # deterministic output
            num_beams=4,       # beam search: better summaries, still deterministic
        )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if not summary:
        raise RuntimeError("Model returned an empty summary")
    return summary


def model_info():
    return {"model": MODEL_NAME, "is_t5": _is_t5()}
