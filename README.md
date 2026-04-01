# Handwriting Recognition AI (Vercel-ready)

A **pure frontend** handwriting recognition web app that runs fully in the browser using **TensorFlow.js**.

## Features

- Upload an image of a handwritten digit
- Capture from camera using `navigator.mediaDevices.getUserMedia()`
- Preprocess to **28×28 grayscale** (MNIST-style)
- Predict **0–9** with a confidence score
- Loading spinner while predicting
- Clean responsive UI (TailwindCSS via CDN)

## Project structure

```
handwriting-recognition/
├── index.html
├── vercel.json
├── package.json
├── public/
│   ├── style.css
│   ├── script.js
│   └── assets/
├── model/
│   ├── model.json
│   ├── group1-shard1of1
│   ├── group2-shard1of1
│   ├── group3-shard1of2
│   ├── group3-shard2of2
│   └── group4-shard1of1
└── README.md
```

Note: TensorFlow.js models can store weights as multiple shard files (instead of a single `weights.bin`). The app loads `./model/model.json`, which references the shard files in `model/`.

## Run locally

Open `index.html` directly OR run a simple static server:

```bash
cd handwriting-recognition
npm run dev
```

## Deploy to Vercel (no backend)

### Option A: Deploy from GitHub (recommended)

- Push this folder to a GitHub repository.
- In Vercel, click **Add New → Project** and import the repo.
- **Framework Preset**: Other
- **Build Command**: leave empty
- **Output Directory**: `.` (project root)
- Deploy.

### Option B: Vercel CLI

```bash
cd handwriting-recognition
npx vercel
```

## Notes for best accuracy

- Try to keep the digit centered and high-contrast.
- The preprocessing auto-inverts images when it detects a bright background (common for photos of black ink on white paper).

