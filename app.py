from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.joblib")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib not found. Run train.py first.")

MODEL = joblib.load(MODEL_PATH)
app = FastAPI(title="Iris Demo", version="1.0.0")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Iris Predictor</title>
  <style>
    :root {
      --bg1: #0f172a;
      --bg2: #1e293b;
      --card: rgba(255, 255, 255, 0.92);
      --card-border: rgba(255, 255, 255, 0.35);
      --text: #0f172a;
      --muted: #64748b;
      --primary: #6366f1;
      --primary-dark: #4f46e5;
      --success: #10b981;
      --shadow: 0 24px 70px rgba(15, 23, 42, 0.28);
      --radius: 28px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(99,102,241,0.40), transparent 30%),
        radial-gradient(circle at top right, rgba(16,185,129,0.25), transparent 28%),
        linear-gradient(135deg, var(--bg1), var(--bg2));
      overflow-x: hidden;
    }

    .page {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 28px;
    }

    .shell {
      width: 100%;
      max-width: 1080px;
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 24px;
      align-items: stretch;
    }

    .hero, .card {
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
    }

    .hero {
      color: white;
      padding: 34px;
      background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 620px;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.18);
      font-size: 0.85rem;
      letter-spacing: 0.02em;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 0 6px rgba(52, 211, 153, 0.14);
    }

    h1 {
      margin: 18px 0 12px;
      font-size: clamp(2.4rem, 4vw, 4.4rem);
      line-height: 0.98;
      letter-spacing: -0.05em;
    }

    .lead {
      max-width: 56ch;
      font-size: 1.05rem;
      line-height: 1.7;
      color: rgba(255,255,255,0.82);
      margin: 0;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-top: 28px;
    }

    .stat {
      padding: 16px;
      border-radius: 20px;
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.14);
    }

    .stat .label {
      display: block;
      font-size: 0.8rem;
      color: rgba(255,255,255,0.68);
      margin-bottom: 6px;
    }

    .stat .value {
      font-size: 1.15rem;
      font-weight: 700;
    }

    .footer-note {
      margin-top: 24px;
      font-size: 0.92rem;
      color: rgba(255,255,255,0.7);
      line-height: 1.6;
    }

    .card {
      background: var(--card);
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .card h2 {
      margin: 0;
      font-size: 1.5rem;
      letter-spacing: -0.03em;
    }

    .card p.sub {
      margin: -6px 0 0;
      color: var(--muted);
      line-height: 1.6;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
      margin-top: 6px;
    }

    .field {
      background: rgba(248,250,252,0.95);
      border: 1px solid #e2e8f0;
      border-radius: 18px;
      padding: 14px 14px 12px;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .field:focus-within {
      transform: translateY(-1px);
      border-color: rgba(99,102,241,0.55);
      box-shadow: 0 10px 24px rgba(99,102,241,0.12);
    }

    label {
      display: block;
      margin-bottom: 8px;
      color: #334155;
      font-size: 0.88rem;
      font-weight: 600;
    }

    input {
      width: 100%;
      border: none;
      outline: none;
      background: transparent;
      font-size: 1rem;
      color: var(--text);
      padding: 0;
    }

    .actions {
      display: flex;
      gap: 12px;
      margin-top: 4px;
    }

    button {
      width: 100%;
      border: none;
      border-radius: 18px;
      padding: 14px 16px;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      color: white;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 14px 30px rgba(79, 70, 229, 0.28);
      transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 18px 36px rgba(79, 70, 229, 0.34);
    }

    button:active {
      transform: translateY(0);
      opacity: 0.96;
    }

    .result {
      margin-top: 4px;
      border-radius: 22px;
      background: linear-gradient(180deg, #ffffff, #f8fafc);
      border: 1px solid #e2e8f0;
      padding: 18px;
    }

    .result-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }

    .result-title {
      margin: 0;
      font-size: 1rem;
      color: #334155;
      font-weight: 600;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-radius: 999px;
      font-weight: 700;
      background: rgba(16, 185, 129, 0.12);
      color: #047857;
      border: 1px solid rgba(16, 185, 129, 0.18);
    }

    .badge.error {
      background: rgba(239, 68, 68, 0.10);
      color: #b91c1c;
      border-color: rgba(239, 68, 68, 0.18);
    }

    .bars {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 110px 1fr 54px;
      gap: 10px;
      align-items: center;
    }

    .bar-label {
      font-size: 0.92rem;
      color: #334155;
      font-weight: 600;
    }

    .bar-track {
      height: 12px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #6366f1, #22c55e);
      width: 0%;
      transition: width 0.45s ease;
    }

    .bar-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: #475569;
      font-weight: 700;
      font-size: 0.92rem;
    }

    .json {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #0f172a;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 18px;
      padding: 16px;
      margin-top: 16px;
      font-size: 0.92rem;
      line-height: 1.6;
    }

    @media (max-width: 900px) {
      .shell {
        grid-template-columns: 1fr;
      }
      .hero {
        min-height: auto;
      }
    }

    @media (max-width: 640px) {
      .page { padding: 14px; }
      .hero, .card { padding: 20px; border-radius: 22px; }
      .grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 1fr; }
      .bar-value { text-align: left; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      <section class="hero">
        <div>
          <div class="eyebrow"><span class="dot"></span> Live ML demo on EC2</div>
          <h1>Iris Species<br/>Predictor</h1>
          <p class="lead">
            A clean, browser-based machine learning demo that predicts iris species from flower measurements.
            Built for a polished AWS + Docker showcase.
          </p>

          <div class="stats">
            <div class="stat">
              <span class="label">Model</span>
              <span class="value">Logistic Regression</span>
            </div>
            <div class="stat">
              <span class="label">Backend</span>
              <span class="value">FastAPI</span>
            </div>
            <div class="stat">
              <span class="label">Deployment</span>
              <span class="value">Docker on EC2</span>
            </div>
          </div>
        </div>

        <div class="footer-note">
          Tip: Try values like <strong>5.1, 3.5, 1.4, 0.2</strong> for setosa or <strong>6.3, 3.3, 6.0, 2.5</strong> for virginica.
        </div>
      </section>

      <section class="card">
        <div>
          <h2>Make a prediction</h2>
          <p class="sub">Enter the four measurements and the model will return the predicted class plus confidence scores.</p>
        </div>

        <div class="grid">
          <div class="field">
            <label for="sepal_length">Sepal length</label>
            <input id="sepal_length" type="number" step="0.1" value="5.1">
          </div>
          <div class="field">
            <label for="sepal_width">Sepal width</label>
            <input id="sepal_width" type="number" step="0.1" value="3.5">
          </div>
          <div class="field">
            <label for="petal_length">Petal length</label>
            <input id="petal_length" type="number" step="0.1" value="1.4">
          </div>
          <div class="field">
            <label for="petal_width">Petal width</label>
            <input id="petal_width" type="number" step="0.1" value="0.2">
          </div>
        </div>

        <button onclick="predict()">Predict species</button>

        <div class="result">
          <div class="result-top">
            <p class="result-title">Result</p>
            <span id="status" class="badge">Ready</span>
          </div>

          <div id="resultText" class="json">Prediction output will appear here.</div>
          <div id="bars" class="bars"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const targets = ["setosa", "versicolor", "virginica"];

    function setStatus(text, isError = false) {
      const status = document.getElementById("status");
      status.textContent = text;
      status.className = isError ? "badge error" : "badge";
    }

    function renderBars(probabilities) {
      const bars = document.getElementById("bars");
      bars.innerHTML = "";

      targets.forEach((name) => {
        const value = probabilities?.[name] ?? 0;
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-label">${name}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.round(value * 100)}%"></div></div>
          <div class="bar-value">${(value * 100).toFixed(1)}%</div>
        `;
        bars.appendChild(row);
      });
    }

    async function predict() {
      setStatus("Predicting...");
      const payload = {
        sepal_length: parseFloat(document.getElementById("sepal_length").value),
        sepal_width: parseFloat(document.getElementById("sepal_width").value),
        petal_length: parseFloat(document.getElementById("petal_length").value),
        petal_width: parseFloat(document.getElementById("petal_width").value),
      };

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || "Prediction failed");
        }

        setStatus("Prediction complete");
        document.getElementById("resultText").textContent = JSON.stringify(data, null, 2);
        renderBars(data.probabilities);
      } catch (err) {
        setStatus("Error", true);
        document.getElementById("resultText").textContent = err.message;
        document.getElementById("bars").innerHTML = "";
      }
    }

    renderBars({ setosa: 0, versicolor: 0, virginica: 0 });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        x = np.array(
            [
                [
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width,
                ]
            ]
        )
        pred = int(MODEL.predict(x)[0])
        probs = MODEL.predict_proba(x)[0]

        return {
            "predicted_class": TARGET_NAMES[pred],
            "probabilities": {
                name: float(prob) for name, prob in zip(TARGET_NAMES, probs)
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
