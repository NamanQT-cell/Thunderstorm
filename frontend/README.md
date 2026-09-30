# Vidyut — React frontend

A React + Vite rebuild of the nowcasting dashboard, styled to match the
supplied Figma design (white ground, olive accent, serif display type,
hairline-divided sections, laptop-mockup hero). It talks to the model
through a small JSON API and never modifies any file under `../src`
other than adding the new, additive `../src/api.py`.

## Run it

**1. Start the API** (wraps the existing, untouched pipeline):
```bash
cd ../src
pip install -r ../requirements.txt --break-system-packages   # adds fastapi/uvicorn
uvicorn api:app --reload --port 8000
```

**2. Start the frontend**, in a second terminal:
```bash
cd frontend
npm install
cp .env.example .env      # points VITE_API_BASE_URL at the API above
npm run dev
```
Open the local URL Vite prints (usually http://localhost:5173).

## How it connects to the model

- `src/lib/api.js` calls `GET /api/frame` and `GET /api/config` on the
  FastAPI server for every slider/dropdown change in the dashboard
  section (`DashboardSplit.jsx`) and on first paint (`Hero.jsx`).
- `src/api.py`, on the backend, imports `preprocessing.py`,
  `ingest_lightning.py`, `labels.py` and `model.py` exactly as
  `dashboard.py` already does — nothing in those files changed.
- If a trained checkpoint exists at `outputs/models/model.pt`
  (produced by `model.py`'s `save_model()`), the API loads
  `ThunderstormCNN` and returns real predictions, flagged
  `"usedRealModel": true` in every response and shown in the dashboard
  badge as "trained model". Until then, responses use the same
  illustrative formula `dashboard.py` uses, flagged `false` and shown
  as "demo formula" — so the UI is always honest about which mode it's
  in, and needs zero code changes on the day you drop in a real model.
- If the API isn't running at all, `src/lib/api.js` quietly falls back
  to `src/lib/offlineSynthetic.js`, a client-side stand-in, so the page
  still renders something on its own — every response in that mode is
  flagged `offline: true` and a small note appears under the hero.

## Structure

```
frontend/
├── src/
│   ├── App.jsx                 <- composes every section
│   ├── components/             <- one file per page section
│   ├── lib/
│   │   ├── api.js              <- fetches from src/api.py
│   │   ├── canvas.js           <- shared canvas drawing helpers
│   │   └── offlineSynthetic.js <- fallback used only if the API is down
│   └── index.css
└── index.html
```

## Build for deployment
```bash
npm run build
```
Outputs static files to `frontend/dist/`. Point `VITE_API_BASE_URL` at
your deployed API's URL before building.
