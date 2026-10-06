# Text Summarizer v2 — Abstractive (BART / T5)

v1 (TF-IDF) text me se sentences **chunta** tha. Ye version **naye shabdon me
summary likhta hai** — Hugging Face transformers (BART / T5) ke saath.

Backend aur frontend alag-alag hain:

```
ml-text-summarizer-v2/
├── backend/    # FastAPI + transformers (port 8000)
└── frontend/   # React + Vite (port 5173)
```

## Backend chalao

```bash
cd backend
pip install -r requirements.txt
# Pehli baar model download hoga (~1.6 GB for BART). Halka model chahiye to:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn main:app --port 8000
```

Pehli request se pehle model startup pe load ho jata hai (`/api/health` pe check karo).

**Model badalna ho to:**

```bash
# T5-small (~240 MB, fast, thoda halka quality)
MODEL_NAME=t5-small uvicorn main:app --port 8000
# T5-base (~850 MB, beech ka rasta)
MODEL_NAME=t5-base uvicorn main:app --port 8000
```

## Frontend chalao (dusre terminal me)

```bash
cd frontend
npm install
npm run dev
```

Browser me kholo: http://localhost:5173

Backend agar 8000 ke alawa kisi port pe hai to:

```bash
VITE_API_URL=http://localhost:8001 npm run dev
```

## Photo se summary (OCR)

Document/notes ki photo upload karo → Tesseract text nikalega → BART/T5 summary
banayega. Endpoint: `POST /api/summarize-image` (multipart: `file`, `max_length`,
`min_length`, `lang` — lang default `eng`, Hindi ke liye `hin` ya `eng+hin`).

**Zaroori:** sirf Python package kaafi nahi — Tesseract ka binary bhi install
karna padega:

- **Windows:** https://github.com/UB-Mannheim/tesseract/wiki se installer lo.
  Install ke baad ya to PATH me add karo, ya phir backend chalate waqt:
  ```
  set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
- **Linux:** `sudo apt install tesseract-ocr tesseract-ocr-hin`

Frontend me "Photo upload karo" tab se JPG/PNG/WebP (max 10 MB) do.

## API

- `GET /api/health` → `{"status": "ok", "model": "..."}`
- `POST /api/summarize` → body: `{"text": "...", "max_length": 150, "min_length": 30}`

```bash
curl -X POST http://localhost:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Lamba text yahan...", "max_length": 120, "min_length": 25}'
```

## Notes

- Bahut lamba input (default 10,000 chars se zyada) truncate ho jata hai —
  BART max 1024 tokens leta hai. `MAX_INPUT_CHARS` env se badal sakte ho.
- `do_sample=False` hai, isliye output deterministic hai (same input → same summary).
