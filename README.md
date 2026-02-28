# 🏥 QHI-Probe

**Quantified Hallucination Index for Clinical LLMs via Sparse Entity-Conditioned Probing**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![AUC-ROC 1.000](https://img.shields.io/badge/AUC--ROC-1.000-22c55e?style=flat-square)](results/)
[![Live Demo](https://img.shields.io/badge/Demo-Live-10b981?style=flat-square)](https://roxrite0509.github.io/QHI)

> **"Instead of running a second AI to verify the first AI, QHI-Probe trains three tiny classifiers to produce a single auditable hallucination severity score in under 0.1ms on CPU."**

```
QHI  =  Uncertainty  ×  Risk Score  ×  Violation Probability  ×  5
                         Range:  0.0 — 25.0
```

---

## 🌐 Live Web Platform

**[https://roxrite0509.github.io/QHI](https://roxrite0509.github.io/QHI)**

Ask any medical question → get an AI answer with real-time QHI confidence scoring.

---

## 🚀 Quick Start

### Web App (recommended)
```bash
cd web
npm install
npm run dev
# → Opens at http://localhost:3000
```

### Python Backend
```bash
pip install scikit-learn numpy pandas scipy
python examples/quickstart.py
```

### Test AI Models
```bash
python test_real_ai.py --mode demo
```

---

## ⚡ How It Works

```
 Clinical LLM Response
        │
        ▼
 ┌── Entity Extraction ──┐
 │ Medical terms only     │  93-97% compute reduction
 └──────────┬─────────────┘
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
[Probe-C] [Probe-R] [Probe-V]
Uncertain  Risk      Violation
 ∈[0,1]   ∈[1,5]    ∈[0,1]
   └────────┼────────┘
            │
  QHI = U × R × V × 5  ∈[0, 25]
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
  <5     5-20      ≥20
 ✅SAFE  ⚠️REVIEW  🚫BLOCK
```

---

## 📁 Repository Structure

```
QHI/
├── web/                          ← Web platform (React + Vite)
│   ├── src/
│   │   ├── App.jsx               Full application
│   │   ├── main.jsx              Entry point
│   │   └── engine/
│   │       └── qhi.js            Browser-side QHI engine
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── qhi_probe/                    ← Python package
│   ├── __init__.py               Public API
│   └── _internals.py             3 probes + system
│
├── data/
│   └── loader.py                 Demo + dataset loaders
├── examples/
│   └── quickstart.py             30-second demo
├── tests/
│   └── test_system.py            10 tests, all passing
│
├── test_real_ai.py               Multi-model testing
├── chat_with_chatgpt.py          Interactive ChatGPT tester
│
├── .github/workflows/deploy.yml  Auto-deploy to GitHub Pages
├── vercel.json                   Vercel config
├── netlify.toml                  Netlify config
├── requirements.txt              Python deps
├── setup.py                      pip install support
├── LICENSE                       MIT
└── README.md                     ← You are here
```

---

## 📊 Benchmark Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.9968** |
| Avg Precision | **0.9962** |
| F1 Score | **0.9254** |
| Pearson r | **0.8552** |
| Inference | **0.08ms** (CPU) |

---

## 🏭 Deploy

### GitHub Pages (auto)
1. Push to `main` → GitHub Actions builds and deploys
2. Enable: Settings → Pages → Source: GitHub Actions
3. Live at: `https://roxrite0509.github.io/QHI`

### Vercel (one click)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Roxrite0509/QHI)

### Netlify
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/Roxrite0509/QHI)

---

## 📖 Citation

```bibtex
@misc{pranav2025qhiprobe,
  title   = {QHI-Probe: Quantified Hallucination Index for Clinical LLMs
             via Sparse Entity-Conditioned Probing},
  author  = {Pranav},
  year    = {2025},
  url     = {https://github.com/Roxrite0509/QHI}
}
```

## 📄 License

MIT — free for research and commercial use. See [LICENSE](LICENSE).
# MEDGUARD
