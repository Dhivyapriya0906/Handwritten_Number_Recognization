# Handwritten Number Recognition

Handwritten Number Recognition is a **fully static AI web app** built with HTML, TailwindCSS, JavaScript, and TensorFlow.js.  
It runs completely in the browser (no backend), supports image upload and camera capture, and predicts handwritten digits (0-9).

## Features

- Browser-only digit prediction with TensorFlow.js
- Upload image and camera capture (`getUserMedia`)
- Image preprocessing to 28x28 grayscale
- Predicted digit + confidence score
- Clean, minimal, responsive UI

## Final project structure

```text
Handwritten_Number_Recognition/
├── index.html
├── model/
│   ├── model.json
│   ├── group1-shard1of1
│   ├── group2-shard1of1
│   ├── group3-shard1of2
│   ├── group3-shard2of2
│   └── group4-shard1of1
├── public/
│   ├── script.js
│   └── style.css
├── package.json
└── README.md
```

Note: The TensorFlow.js model uses multiple weight shard files referenced by `model/model.json`.

## Production-safe paths used

- CSS: `/public/style.css`
- JS: `/public/script.js`
- Model: `/model/model.json`

These absolute root paths make deployment stable on Vercel static hosting.

## Run locally

Use any static server from project root:

```bash
npm run dev
```

Or open `index.html` directly in a browser (camera permissions can be more reliable through a local server).

## Deploy on Vercel (Static, No Backend)

### Option 1: Vercel Dashboard

1. Push project to GitHub.
2. In Vercel, click **Add New -> Project**.
3. Import your repository.
4. Framework preset: **Other**.
5. Build command: leave empty.
6. Output directory: leave empty (root static files).
7. Deploy.

### Option 2: Vercel CLI

```bash
npx vercel
```

When prompted, keep defaults for a static project.

## Why this fixes Vercel 404 issues

- `index.html` is at the root and acts as the entry point.
- No custom rewrite config is used, so Vercel serves static files directly.
- Model and asset paths are rooted from `/`, so they resolve correctly in production.

