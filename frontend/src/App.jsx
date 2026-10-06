import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SAMPLE =
  "Artificial intelligence is transforming the way software is built and used " +
  "across every industry. Machine learning models can now write code, answer " +
  "questions, and even create images from text descriptions. However, these " +
  "models need large amounts of data and computing power to train. " +
  "Researchers are working on making them smaller and more efficient so they " +
  "can run on everyday devices. In the coming years, AI assistants will " +
  "become a normal part of how students learn and how developers work.";

export default function App() {
  const [mode, setMode] = useState("text"); // "text" | "photo"
  const [text, setText] = useState("");
  const [photo, setPhoto] = useState(null);
  const [preview, setPreview] = useState("");
  const [maxLen, setMaxLen] = useState(120);
  const [minLen, setMinLen] = useState(25);
  const [summary, setSummary] = useState("");
  const [extracted, setExtracted] = useState("");
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const wordCount = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);

  function reset() {
    setSummary(""); setExtracted(""); setStats(null); setError("");
  }

  function onPhotoChange(e) {
    const f = e.target.files?.[0];
    setPhoto(f || null);
    setPreview(f ? URL.createObjectURL(f) : "");
    reset();
  }

  async function handleSummarize() {
    if (minLen >= maxLen) {
      setError("Min length must be smaller than max length.");
      return;
    }
    setLoading(true);
    reset();
    try {
      let res;
      if (mode === "text") {
        if (!text.trim()) throw new Error("Paste some text first.");
        res = await fetch(`${API_URL}/api/summarize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, max_length: maxLen, min_length: minLen }),
        });
      } else {
        if (!photo) throw new Error("Choose a photo first (JPG/PNG/WebP).");
        const form = new FormData();
        form.append("file", photo);
        form.append("max_length", String(maxLen));
        form.append("min_length", String(minLen));
        form.append("lang", "eng");
        res = await fetch(`${API_URL}/api/summarize-image`, {
          method: "POST",
          body: form,
        });
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Something went wrong.");
      setSummary(data.summary);
      setExtracted(data.extracted_text || "");
      setStats({ original: data.original_words, summary: data.summary_words });
    } catch (e) {
      setError(
        e.message.includes("Failed to fetch")
          ? "Couldn't reach the backend. Is it running? (uvicorn main:app --port 8000)"
          : e.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>Text Summarizer</h1>
        <p className="sub">Abstractive summaries with BART / T5 — paste text or upload a photo.</p>
      </header>

      <div className="tabs">
        <button className={mode === "text" ? "active" : ""} onClick={() => { setMode("text"); reset(); }}>
          Write text
        </button>
        <button className={mode === "photo" ? "active" : ""} onClick={() => { setMode("photo"); reset(); }}>
          Upload photo
        </button>
      </div>

      <section className="card">
        {mode === "text" ? (
          <>
            <div className="row between">
              <label>Paste your text here</label>
              <button className="link" onClick={() => setText(SAMPLE)}>
                Try a sample
              </button>
            </div>
            <textarea
              rows={10}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste a long article, notes, anything..."
            />
            <div className="meta">{wordCount(text)} words</div>
          </>
        ) : (
          <>
            <label>Photo of a document / notes (JPG, PNG, WebP — max 10 MB)</label>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={onPhotoChange}
              className="fileinput"
            />
            {preview && (
              <img src={preview} alt="upload preview" className="preview" />
            )}
            <div className="meta">A straight, well-lit photo reads best.</div>
          </>
        )}

        <div className="controls">
          <div>
            <label>Max length: {maxLen}</label>
            <input
              type="range" min={30} max={400} step={10}
              value={maxLen} onChange={(e) => setMaxLen(+e.target.value)}
            />
          </div>
          <div>
            <label>Min length: {minLen}</label>
            <input
              type="range" min={10} max={150} step={5}
              value={minLen} onChange={(e) => setMinLen(+e.target.value)}
            />
          </div>
        </div>

        <div className="row">
          <button className="primary" onClick={handleSummarize} disabled={loading}>
            {loading ? "Working..." : "Summarize"}
          </button>
          <button
            className="ghost"
            onClick={() => { setText(""); setPhoto(null); setPreview(""); reset(); }}
          >
            Clear
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {summary && (
        <section className="card result">
          <h2>Summary</h2>
          <p>{summary}</p>
          {stats && (
            <div className="meta">
              {stats.original} words → {stats.summary} words
              {" "}({Math.round((1 - stats.summary / stats.original) * 100)}% shorter)
            </div>
          )}
          {extracted && (
            <details className="extracted">
              <summary>Text read by OCR (click to view)</summary>
              <p>{extracted}</p>
            </details>
          )}
        </section>
      )}

      <footer>Backend: FastAPI + transformers + Tesseract OCR · start the backend first, then this page</footer>
    </div>
  );
}
