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
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Iris Predictor</title>
  <style>
    body {
      margin: 0;
      font-family: Inter, system-ui, Arial, sans-serif;
      background: linear-gradient(135deg, #eef2ff, #f8fafc);
      color: #0f172a;
    }
    .wrap {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .card {
      width: 100%;
      max-width: 760px;
      background: white;
      border-radius: 24px;
      box-shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
      padding: 32px;
    }
    h1 { margin: 0 0 8px; font-size: 2rem; }
    p { margin: 0 0 24px; color: #475569; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }
    label { display: block; font-size: .9rem; margin-bottom: 6px; color: #334155; }
    input {
      width: 100%;
      box-sizing: border-box;
      padding: 12px 14px;
      border: 1px solid #cbd5e1;
      border-radius: 14px;
      font-size: 1rem;
    }
    button {
      margin-top: 20px;
      width: 100%;
      border: none;
      border-radius: 14px;
      padding: 14px;
      background: #4f46e5;
      color: white;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: #4338ca; }
    .result {
      margin-top: 20px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 16px;
      white-space: pre-wrap;
    }
    @media (max-width: 640px) {
      .grid { grid-template-columns: 1fr; }
      .card { padding: 20px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Iris Species Predictor</h1>
      <p>Enter flower measurements and get a prediction from the model running on EC2.</p>

      <div class="grid">
        <div><label>Sepal length</label><input id="sepal_length" type="number" step="0.1" value="5.1"></div>
        <div><label>Sepal width</label><input id="sepal_width" type="number" step="0.1" value="3.5"></div>
        <div><label>Petal length</label><input id="petal_length" type="number" step="0.1" value="1.4"></div>
        <div><label>Petal width</label><input id="petal_width" type="number" step="0.1" value="0.2"></div>
      </div>

      <button onclick="predict()">Predict</button>
      <div id="result" class="result">Result will appear here.</div>
    </div>
  </div>

  <script>
    async function predict() {
      const payload = {
        sepal_length: parseFloat(document.getElementById("sepal_length").value),
        sepal_width: parseFloat(document.getElementById("sepal_width").value),
        petal_length: parseFloat(document.getElementById("petal_length").value),
        petal_width: parseFloat(document.getElementById("petal_width").value),
      };

      const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      document.getElementById("result").textContent =
        JSON.stringify(data, null, 2);
    }
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

