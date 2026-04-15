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
  <title>Iris Species Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Crimson+Pro:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --parchment:   #f4ede0;
      --parchment-d: #e9dcc9;
      --ink:         #1c1a14;
      --ink-muted:   #4a4535;
      --ink-faint:   #7a7260;
      --green:       #2a4a2e;
      --green-light: #3d6b43;
      --purple:      #5c3d8f;
      --purple-light:#7b5ea7;
      --gold:        #a07c2a;
      --gold-light:  #c9a248;
      --rule:        rgba(44, 36, 12, 0.18);
    }

    html, body {
      height: 100%;
    }

    body {
      font-family: 'Crimson Pro', Georgia, serif;
      background: var(--parchment);
      color: var(--ink);
      min-height: 100vh;
      background-image:
        repeating-linear-gradient(0deg, transparent, transparent 31px, rgba(44,36,12,0.04) 31px, rgba(44,36,12,0.04) 32px),
        repeating-linear-gradient(90deg, transparent, transparent 31px, rgba(44,36,12,0.03) 31px, rgba(44,36,12,0.03) 32px);
      overflow-x: hidden;
    }

    .page {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 36px 24px;
    }

    .shell {
      width: 100%;
      max-width: 1060px;
      display: grid;
      grid-template-columns: 1fr 1.1fr;
      gap: 0;
      border: 1.5px solid var(--rule);
      background: var(--parchment);
      position: relative;
    }

    /* outer decorative double-border */
    .shell::before {
      content: '';
      position: absolute;
      inset: -6px;
      border: 1px solid var(--rule);
      pointer-events: none;
      z-index: 0;
    }
    .shell::after {
      content: '';
      position: absolute;
      inset: -11px;
      border: 0.5px solid rgba(44,36,12,0.10);
      pointer-events: none;
      z-index: 0;
    }

    /* ── LEFT PANEL ── */
    .left {
      border-right: 1.5px solid var(--rule);
      display: flex;
      flex-direction: column;
      padding: 0;
      position: relative;
      overflow: hidden;
    }

    .left-header {
      padding: 28px 28px 20px;
      border-bottom: 1px solid var(--rule);
    }

    .classification {
      font-family: 'Space Mono', monospace;
      font-size: 0.68rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--green);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .classification::before {
      content: '';
      display: inline-block;
      width: 20px;
      height: 1px;
      background: var(--green);
      flex-shrink: 0;
    }

    .title {
      font-family: 'Cormorant Garamond', serif;
      font-size: clamp(2.6rem, 4vw, 4rem);
      font-weight: 300;
      line-height: 0.92;
      letter-spacing: -0.02em;
      color: var(--ink);
      margin-bottom: 4px;
    }
    .title em {
      font-style: italic;
      font-weight: 300;
      color: var(--purple);
    }

    .subtitle {
      font-family: 'Space Mono', monospace;
      font-size: 0.72rem;
      color: var(--ink-faint);
      letter-spacing: 0.06em;
      margin-top: 12px;
    }

    /* Iris SVG illustration area */
    .illustration {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px 28px;
      position: relative;
    }

    .iris-svg {
      width: 100%;
      max-width: 280px;
    }

    /* specimen data strip */
    .specimen-strip {
      border-top: 1px solid var(--rule);
      padding: 16px 28px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0;
    }
    .spec-item {
      padding: 0 12px 0 0;
    }
    .spec-item + .spec-item {
      padding-left: 12px;
      border-left: 1px solid var(--rule);
    }
    .spec-label {
      font-family: 'Space Mono', monospace;
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--ink-faint);
      display: block;
      margin-bottom: 3px;
    }
    .spec-value {
      font-family: 'Cormorant Garamond', serif;
      font-size: 1.05rem;
      font-weight: 500;
      color: var(--green);
    }

    .footer-tip {
      border-top: 1px solid var(--rule);
      padding: 12px 28px;
      font-size: 0.85rem;
      color: var(--ink-faint);
      font-style: italic;
    }
    .footer-tip strong {
      font-style: normal;
      font-family: 'Space Mono', monospace;
      font-size: 0.78rem;
      color: var(--ink-muted);
    }

    /* ── RIGHT PANEL ── */
    .right {
      display: flex;
      flex-direction: column;
      padding: 28px;
      gap: 20px;
      background: linear-gradient(160deg, var(--parchment) 0%, #ede6d4 100%);
    }

    .form-header {
      border-bottom: 1px solid var(--rule);
      padding-bottom: 14px;
    }
    .form-title {
      font-family: 'Cormorant Garamond', serif;
      font-size: 1.5rem;
      font-weight: 400;
      letter-spacing: -0.01em;
      color: var(--ink);
      margin-bottom: 3px;
    }
    .form-sub {
      font-size: 0.92rem;
      color: var(--ink-faint);
      line-height: 1.5;
    }

    .measurements-label {
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--gold);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .measurements-label::after {
      content: '';
      flex: 1;
      height: 1px;
      background: linear-gradient(90deg, var(--gold), transparent);
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .field {
      background: rgba(255,255,255,0.5);
      border: 1px solid var(--rule);
      padding: 12px 14px 10px;
      position: relative;
      transition: border-color 0.2s, background 0.2s;
      cursor: text;
    }
    .field:focus-within {
      border-color: rgba(92, 61, 143, 0.45);
      background: rgba(255,255,255,0.75);
    }
    .field:focus-within .field-corner {
      border-color: var(--purple-light);
    }

    /* bracket-style corner decorations */
    .field-corner {
      position: absolute;
      width: 8px;
      height: 8px;
      border-color: var(--gold-light);
      transition: border-color 0.2s;
    }
    .field-corner.tl { top: -1px; left: -1px; border-top: 1.5px solid; border-left: 1.5px solid; }
    .field-corner.tr { top: -1px; right: -1px; border-top: 1.5px solid; border-right: 1.5px solid; }
    .field-corner.bl { bottom: -1px; left: -1px; border-bottom: 1.5px solid; border-left: 1.5px solid; }
    .field-corner.br { bottom: -1px; right: -1px; border-bottom: 1.5px solid; border-right: 1.5px solid; }

    label {
      display: block;
      font-family: 'Space Mono', monospace;
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--ink-muted);
      margin-bottom: 5px;
    }

    input[type=number] {
      width: 100%;
      border: none;
      outline: none;
      background: transparent;
      font-family: 'Space Mono', monospace;
      font-size: 1.1rem;
      color: var(--ink);
      padding: 0;
      -moz-appearance: textfield;
    }
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; }

    .unit {
      position: absolute;
      bottom: 12px;
      right: 14px;
      font-family: 'Space Mono', monospace;
      font-size: 0.6rem;
      color: var(--ink-faint);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .predict-btn {
      width: 100%;
      border: 1.5px solid var(--green);
      background: var(--green);
      color: var(--parchment);
      font-family: 'Space Mono', monospace;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      padding: 14px;
      cursor: pointer;
      transition: background 0.2s, color 0.2s, transform 0.12s;
      position: relative;
      overflow: hidden;
    }
    .predict-btn::before {
      content: '→';
      position: absolute;
      right: 16px;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0.6;
      font-size: 1rem;
    }
    .predict-btn:hover {
      background: var(--green-light);
      border-color: var(--green-light);
    }
    .predict-btn:active {
      transform: scale(0.99);
    }

    /* ── RESULT PANEL ── */
    .result-panel {
      border: 1px solid var(--rule);
      background: rgba(255,255,255,0.45);
      padding: 16px 18px;
      position: relative;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .result-label {
      font-family: 'Space Mono', monospace;
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--ink-faint);
    }

    .badge {
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 4px 10px;
      border: 1px solid;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .badge.ready  { color: var(--ink-faint); border-color: var(--rule); }
    .badge.ok     { color: var(--green); border-color: rgba(42,74,46,0.35); background: rgba(42,74,46,0.06); }
    .badge.error  { color: #8b2020; border-color: rgba(139,32,32,0.3); background: rgba(139,32,32,0.06); }
    .badge.working { color: var(--gold); border-color: rgba(160,124,42,0.35); }
    .badge-dot {
      width: 5px; height: 5px; border-radius: 50%;
      background: currentColor;
      flex-shrink: 0;
    }

    .prediction-name {
      font-family: 'Cormorant Garamond', serif;
      font-size: 2rem;
      font-weight: 400;
      font-style: italic;
      color: var(--purple);
      letter-spacing: -0.01em;
      min-height: 2.4rem;
    }
    .prediction-name.empty { color: var(--ink-faint); opacity: 0.4; font-size: 1.1rem; font-style: normal; }

    /* probability bars */
    .bars { display: grid; gap: 8px; }
    .bar-row {
      display: grid;
      grid-template-columns: 80px 1fr 42px;
      gap: 8px;
      align-items: center;
    }
    .bar-name {
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      color: var(--ink-muted);
      text-transform: lowercase;
    }
    .bar-track {
      height: 6px;
      background: rgba(44,36,12,0.10);
      position: relative;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      transition: width 0.55s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .bar-fill.setosa     { background: var(--purple); }
    .bar-fill.versicolor { background: var(--green); }
    .bar-fill.virginica  { background: var(--gold); }
    .bar-pct {
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      color: var(--ink-muted);
      text-align: right;
    }

    .raw-json {
      font-family: 'Space Mono', monospace;
      font-size: 0.67rem;
      line-height: 1.7;
      color: var(--ink-muted);
      border-top: 1px solid var(--rule);
      padding-top: 10px;
      word-break: break-all;
      white-space: pre-wrap;
    }

    /* decorative page number */
    .page-number {
      position: absolute;
      bottom: 8px;
      right: 12px;
      font-family: 'Space Mono', monospace;
      font-size: 0.6rem;
      color: rgba(44,36,12,0.2);
    }

    /* ── RESPONSIVE ── */
    @media (max-width: 820px) {
      .shell { grid-template-columns: 1fr; }
      .left { border-right: none; border-bottom: 1.5px solid var(--rule); }
      .illustration { padding: 16px 24px; }
      .iris-svg { max-width: 220px; }
    }
    @media (max-width: 520px) {
      .page { padding: 16px 12px; }
      .shell::before, .shell::after { display: none; }
      .grid { grid-template-columns: 1fr; }
      .specimen-strip { grid-template-columns: 1fr; gap: 8px; }
      .spec-item + .spec-item { padding-left: 0; border-left: none; border-top: 1px solid var(--rule); padding-top: 8px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">

      <!-- ── LEFT: HERO ── -->
      <section class="left">
        <div class="left-header">
          <div class="classification">Iridaceae · Iridoideae · Iris L.</div>
          <h1 class="title">Iris<br><em>Species</em><br>Predictor</h1>
          <p class="subtitle">Logistic Regression · FastAPI · Docker on EC2</p>
        </div>

        <div class="illustration">
          <!-- Detailed botanical iris SVG -->
          <svg class="iris-svg" viewBox="0 0 260 320" xmlns="http://www.w3.org/2000/svg" fill="none">
            <!-- stem -->
            <path d="M130 310 L130 195" stroke="#2a4a2e" stroke-width="2.5" stroke-linecap="round"/>
            <!-- leaf left -->
            <path d="M130 270 Q90 240 75 200 Q105 215 130 245" fill="#3d6b43" opacity="0.7"/>
            <!-- leaf right -->
            <path d="M130 250 Q170 225 185 185 Q155 205 130 230" fill="#2a4a2e" opacity="0.6"/>

            <!-- falls (lower petals) -->
            <path d="M130 195 Q90 185 60 165 Q70 195 95 205 Q110 210 130 215Z" fill="#7b5ea7" opacity="0.85"/>
            <path d="M130 195 Q170 185 200 165 Q190 195 165 205 Q150 210 130 215Z" fill="#7b5ea7" opacity="0.85"/>
            <path d="M130 195 Q130 175 130 145 Q120 170 115 185 Q122 192 130 195Z" fill="#5c3d8f" opacity="0.9"/>

            <!-- standards (upper petals) -->
            <path d="M130 195 Q105 170 100 135 Q115 160 125 178 Q127 186 130 195Z" fill="#9b7ecf" opacity="0.8"/>
            <path d="M130 195 Q155 170 160 135 Q145 160 135 178 Q133 186 130 195Z" fill="#9b7ecf" opacity="0.8"/>
            <path d="M130 195 Q130 165 130 125 Q124 155 124 175 Q127 185 130 195Z" fill="#c4a8e8" opacity="0.7"/>

            <!-- beard / signal details -->
            <path d="M108 188 Q114 192 120 191" stroke="#c9a248" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M150 188 Q145 192 140 191" stroke="#c9a248" stroke-width="1.5" stroke-linecap="round"/>

            <!-- veining on falls -->
            <path d="M130 210 Q102 207 75 190" stroke="#5c3d8f" stroke-width="0.6" opacity="0.4" stroke-linecap="round"/>
            <path d="M130 210 Q158 207 185 190" stroke="#5c3d8f" stroke-width="0.6" opacity="0.4" stroke-linecap="round"/>

            <!-- small secondary bud -->
            <path d="M130 195 Q148 188 158 175 Q148 188 140 193Z" fill="#b89fd6" opacity="0.5"/>

            <!-- crosshair target ring (scientific illustration marker) -->
            <circle cx="130" cy="195" r="52" stroke="rgba(92,61,143,0.12)" stroke-width="0.8" stroke-dasharray="4 4"/>
            <circle cx="130" cy="195" r="78" stroke="rgba(92,61,143,0.07)" stroke-width="0.5" stroke-dasharray="3 6"/>

            <!-- measurement lines (calipers feel) -->
            <line x1="44" y1="165" x2="44" y2="225" stroke="rgba(92,61,143,0.2)" stroke-width="0.75"/>
            <line x1="40" y1="165" x2="48" y2="165" stroke="rgba(92,61,143,0.2)" stroke-width="0.75"/>
            <line x1="40" y1="225" x2="48" y2="225" stroke="rgba(92,61,143,0.2)" stroke-width="0.75"/>
            <text x="36" y="198" font-family="monospace" font-size="6" fill="rgba(92,61,143,0.4)" text-anchor="middle" transform="rotate(-90 36 198)">sepal</text>

            <!-- petal measurement lines -->
            <line x1="216" y1="135" x2="216" y2="195" stroke="rgba(160,124,42,0.25)" stroke-width="0.75"/>
            <line x1="212" y1="135" x2="220" y2="135" stroke="rgba(160,124,42,0.25)" stroke-width="0.75"/>
            <line x1="212" y1="195" x2="220" y2="195" stroke="rgba(160,124,42,0.25)" stroke-width="0.75"/>
            <text x="224" y="168" font-family="monospace" font-size="6" fill="rgba(160,124,42,0.45)" text-anchor="middle" transform="rotate(90 224 168)">petal</text>
          </svg>
        </div>

        <div class="specimen-strip">
          <div class="spec-item">
            <span class="spec-label">Model</span>
            <span class="spec-value">Logistic Reg.</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Classes</span>
            <span class="spec-value">3 Species</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Features</span>
            <span class="spec-value">4 Measures</span>
          </div>
        </div>

        <div class="footer-tip">
          Try <strong>5.1, 3.5, 1.4, 0.2</strong> for setosa or <strong>6.3, 3.3, 6.0, 2.5</strong> for virginica.
        </div>
      </section>

      <!-- ── RIGHT: FORM + RESULT ── -->
      <section class="right">
        <div class="form-header">
          <h2 class="form-title">Morphometric Measurements</h2>
          <p class="form-sub">Input the four floral measurements to classify the specimen.</p>
        </div>

        <div class="measurements-label">Sepal &amp; petal dimensions</div>

        <div class="grid">
          <div class="field">
            <div class="field-corner tl"></div><div class="field-corner tr"></div>
            <div class="field-corner bl"></div><div class="field-corner br"></div>
            <label for="sepal_length">Sepal Length</label>
            <input id="sepal_length" type="number" step="0.1" value="5.1">
            <span class="unit">cm</span>
          </div>
          <div class="field">
            <div class="field-corner tl"></div><div class="field-corner tr"></div>
            <div class="field-corner bl"></div><div class="field-corner br"></div>
            <label for="sepal_width">Sepal Width</label>
            <input id="sepal_width" type="number" step="0.1" value="3.5">
            <span class="unit">cm</span>
          </div>
          <div class="field">
            <div class="field-corner tl"></div><div class="field-corner tr"></div>
            <div class="field-corner bl"></div><div class="field-corner br"></div>
            <label for="petal_length">Petal Length</label>
            <input id="petal_length" type="number" step="0.1" value="1.4">
            <span class="unit">cm</span>
          </div>
          <div class="field">
            <div class="field-corner tl"></div><div class="field-corner tr"></div>
            <div class="field-corner bl"></div><div class="field-corner br"></div>
            <label for="petal_width">Petal Width</label>
            <input id="petal_width" type="number" step="0.1" value="0.2">
            <span class="unit">cm</span>
          </div>
        </div>

        <button class="predict-btn" onclick="predict()">Classify Specimen</button>

        <!-- Result -->
        <div class="result-panel">
          <div class="result-header">
            <span class="result-label">Classification result</span>
            <span id="badge" class="badge ready"><span class="badge-dot"></span>Awaiting input</span>
          </div>

          <div id="predName" class="prediction-name empty">Iris sp.</div>

          <div class="bars" id="bars">
            <div class="bar-row">
              <div class="bar-name">setosa</div>
              <div class="bar-track"><div class="bar-fill setosa" id="bar-setosa"></div></div>
              <div class="bar-pct" id="pct-setosa">—</div>
            </div>
            <div class="bar-row">
              <div class="bar-name">versicolor</div>
              <div class="bar-track"><div class="bar-fill versicolor" id="bar-versicolor"></div></div>
              <div class="bar-pct" id="pct-versicolor">—</div>
            </div>
            <div class="bar-row">
              <div class="bar-name">virginica</div>
              <div class="bar-track"><div class="bar-fill virginica" id="bar-virginica"></div></div>
              <div class="bar-pct" id="pct-virginica">—</div>
            </div>
          </div>

          <pre id="rawJson" class="raw-json" style="display:none;"></pre>
          <div class="page-number">fig. 1</div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const species = ["setosa", "versicolor", "virginica"];

    function setBadge(text, state = "ready") {
      const b = document.getElementById("badge");
      b.className = `badge ${state}`;
      b.innerHTML = `<span class="badge-dot"></span>${text}`;
    }

    function setResult(name, probs) {
      const nameEl = document.getElementById("predName");
      if (name) {
        nameEl.className = "prediction-name";
        nameEl.textContent = "Iris " + name;
      } else {
        nameEl.className = "prediction-name empty";
        nameEl.textContent = "Iris sp.";
      }
      species.forEach(s => {
        const v = probs ? (probs[s] ?? 0) : 0;
        const pct = probs ? (v * 100).toFixed(1) + "%" : "—";
        document.getElementById(`bar-${s}`).style.width = probs ? Math.round(v * 100) + "%" : "0%";
        document.getElementById(`pct-${s}`).textContent = pct;
      });
    }

    async function predict() {
      setBadge("Analysing…", "working");
      setResult(null, null);
      document.getElementById("rawJson").style.display = "none";

      const payload = {
        sepal_length: parseFloat(document.getElementById("sepal_length").value),
        sepal_width:  parseFloat(document.getElementById("sepal_width").value),
        petal_length: parseFloat(document.getElementById("petal_length").value),
        petal_width:  parseFloat(document.getElementById("petal_width").value),
      };

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        setBadge("Classified", "ok");
        setResult(data.prediction, data.probabilities);

        const raw = document.getElementById("rawJson");
        raw.textContent = JSON.stringify(data, null, 2);
        raw.style.display = "block";
      } catch (err) {
        setBadge("Error", "error");
        const raw = document.getElementById("rawJson");
        raw.textContent = err.message;
        raw.style.display = "block";
      }
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
