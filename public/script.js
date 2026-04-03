let model = null;
let cameraStream = null;
const VALID_CONFIDENCE_THRESHOLD = 0.1;
const NO_DIGIT_MIN_FOREGROUND_PIXELS = 110; // heuristic for digit component area on temp canvas

// Higher-res preprocessing improves thresholding and centering.
const PREP_SIZE = 56; // temp canvas size (final model input is 28x28)
const ADAPTIVE_WINDOW_SIZE = 15; // must be odd
const ADAPTIVE_THRESHOLD_C = 0.03; // subtract from local mean

// Contrast boosting to help thin strokes survive thresholding.
const CONTRAST_GAMMA = 0.75; // <1 increases contrast

const MORPH_PASSES = 1; // opening+closing pass count

const els = {};

function $(id) {
  return document.getElementById(id);
}

function setModelStatus(text, ok) {
  els.modelStatus.textContent = text;
  els.modelDot.className =
    "h-2 w-2 rounded-full " + (ok ? "bg-slate-900" : "bg-slate-300");
}

function setError(msg) {
  if (!msg) {
    els.errorBox.classList.add("hidden");
    els.errorBox.textContent = "";
    return;
  }
  els.errorBox.textContent = msg;
  els.errorBox.classList.remove("hidden");
}

function setLoading(loading) {
  if (loading) {
    els.spinner.classList.remove("hidden");
    els.spinner.classList.add("inline-flex");
  } else {
    els.spinner.classList.add("hidden");
    els.spinner.classList.remove("inline-flex");
  }
}

function setPrediction(digit, confidence, hasDigit = true) {
  if (digit === null || digit === undefined) {
    els.predDigit.textContent = "—";
    els.predDigit.className =
      "mt-1 text-6xl font-bold tracking-tight text-slate-900";
    els.confText.textContent = "—";
    els.confBar.style.width = "0%";
    return;
  }

  // Confidence bar always reflects the model's highest probability.
  const pct = Math.max(0, Math.min(1, confidence)) * 100;
  els.confText.textContent = `${pct.toFixed(1)}%`;
  els.confBar.style.width = `${pct.toFixed(1)}%`;

  if (!hasDigit) {
    els.predDigit.textContent = "No digit detected";
    els.predDigit.className =
      "mt-2 text-base font-semibold leading-snug text-slate-700";
    return;
  }

  if (confidence < VALID_CONFIDENCE_THRESHOLD) {
    els.predDigit.textContent = "Not a valid handwritten number";
    els.predDigit.className =
      "mt-2 text-base font-semibold leading-snug text-slate-700";
    return;
  }

  els.predDigit.textContent = String(digit);
  els.predDigit.className =
    "mt-1 text-6xl font-bold tracking-tight text-slate-900";
}

function showPreviewFromDataUrl(dataUrl) {
  els.previewImg.src = dataUrl;
  els.previewImg.classList.remove("hidden");
  els.previewEmpty.classList.add("hidden");
}

function clearAll() {
  setError("");
  setPrediction(null, 0);
  els.previewImg.src = "";
  els.previewImg.classList.add("hidden");
  els.previewEmpty.classList.remove("hidden");
  els.fileInput.value = "";
}

async function ensureTfReady() {
  if (!window.tf) throw new Error("TensorFlow.js failed to load.");
  await tf.ready();
  try {
    await tf.setBackend("webgl");
  } catch {
    // ignore, fallback to default
  }
  await tf.ready();
}

async function loadModel() {
  setModelStatus("Loading model…", false);
  await ensureTfReady();
  model = await tf.loadLayersModel("/model/model.json");
  setModelStatus("Model ready", true);
}

function centerCropToSquare(drawCtx, imgW, imgH) {
  const side = Math.min(imgW, imgH);
  const sx = Math.floor((imgW - side) / 2);
  const sy = Math.floor((imgH - side) / 2);
  drawCtx.__crop = { sx, sy, side };
  return drawCtx.__crop;
}

function ensurePrepCanvas() {
  if (!els.prepCanvas) {
    els.prepCanvas = document.createElement("canvas");
    els.prepCanvas.width = PREP_SIZE;
    els.prepCanvas.height = PREP_SIZE;
    els.prepCtx = els.prepCanvas.getContext("2d", { willReadFrequently: true });
  }
  return { canvas: els.prepCanvas, ctx: els.prepCtx };
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function buildIntegralImage(lum, w, h) {
  // integral[y*(w+1)+x] = sum of lum in rectangle [0..x-1, 0..y-1)
  const stride = w + 1;
  const integral = new Float32Array((w + 1) * (h + 1));

  for (let y = 1; y <= h; y++) {
    let rowSum = 0;
    for (let x = 1; x <= w; x++) {
      rowSum += lum[(y - 1) * w + (x - 1)];
      integral[y * stride + x] = integral[(y - 1) * stride + x] + rowSum;
    }
  }
  return integral;
}

function adaptiveThresholdBinary(lum, w, h, windowSize, c) {
  // Adaptive mean thresholding.
  // Binary pixel = 1 if lum(x,y) >= mean(localWindow) - c else 0.
  const half = (windowSize / 2) | 0;
  const bin = new Uint8Array(w * h);

  const integral = buildIntegralImage(lum, w, h);
  const stride = w + 1;

  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - half);
    const y1 = Math.min(h - 1, y + half);
    const iy0 = y0;
    const iy1 = y1 + 1;

    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - half);
      const x1 = Math.min(w - 1, x + half);
      const ix0 = x0;
      const ix1 = x1 + 1;

      const sum =
        integral[iy1 * stride + ix1] -
        integral[iy0 * stride + ix1] -
        integral[iy1 * stride + ix0] +
        integral[iy0 * stride + ix0];

      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mean = sum / area;

      const idx = y * w + x;
      bin[idx] = lum[idx] >= mean - c ? 1 : 0;
    }
  }

  return bin;
}

function erode3x3(bin, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let keep = 1;
      for (let dy = -1; dy <= 1 && keep; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
            keep = 0;
            break;
          }
          if (bin[ny * w + nx] === 0) {
            keep = 0;
            break;
          }
        }
      }
      out[y * w + x] = keep;
    }
  }
  return out;
}

function dilate3x3(bin, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let any = 0;
      for (let dy = -1; dy <= 1 && !any; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          if (bin[ny * w + nx] === 1) {
            any = 1;
            break;
          }
        }
      }
      out[y * w + x] = any;
    }
  }
  return out;
}

function morphologicalOpenClose(bin, w, h, passes) {
  // Opening removes small noise. Closing fills small holes.
  let out = bin;
  for (let i = 0; i < passes; i++) {
    out = dilate3x3(erode3x3(out, w, h), w, h); // open
    out = erode3x3(dilate3x3(out, w, h), w, h); // close
  }
  return out;
}

function filterLargestComponent(bin, w, h, minArea) {
  // Select the "best" connected component:
  // - Prefer components with large area
  // - Prefer components closer to the canvas center
  // This helps avoid picking notebook lines/text blobs that are large but off-center.
  const n = w * h;
  const visited = new Uint8Array(n);

  let bestScore = -Infinity;
  let bestArea = 0;
  let bestIdxs = null;

  const qx = new Int32Array(n);
  const qy = new Int32Array(n);

  const cx0 = (w - 1) / 2;
  const cy0 = (h - 1) / 2;

  const dirs = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];

  for (let i = 0; i < n; i++) {
    if (bin[i] === 0 || visited[i] === 1) continue;

    const startX = i % w;
    const startY = (i / w) | 0;

    let head = 0;
    let tail = 0;
    qx[tail] = startX;
    qy[tail] = startY;
    tail++;

    visited[i] = 1;

    let area = 0;
    const idxs = [];

    let sumX = 0;
    let sumY = 0;

    let minX = w;
    let minY = h;
    let maxX = -1;
    let maxY = -1;

    while (head < tail) {
      const x = qx[head];
      const y = qy[head];
      head++;

      const idx = y * w + x;
      idxs.push(idx);
      area++;
      sumX += x;
      sumY += y;

      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;

      for (let d = 0; d < dirs.length; d++) {
        const nx = x + dirs[d][0];
        const ny = y + dirs[d][1];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nidx = ny * w + nx;
        if (visited[nidx] === 1 || bin[nidx] === 0) continue;
        visited[nidx] = 1;
        qx[tail] = nx;
        qy[tail] = ny;
        tail++;
      }
    }

    if (area < minArea) continue;

    const centroidX = sumX / area;
    const centroidY = sumY / area;
    const dx = centroidX - cx0;
    const dy = centroidY - cy0;
    const dist2 = dx * dx + dy * dy;

    // Mild penalty for extremely thin line-like components (reduces notebook lines).
    // Kept gentle to not break digit "1".
    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const thinScore = Math.min(bw, bh) <= 2 ? 1 : 0;

    // Higher is better.
    // area dominates, center distance is a tie-breaker.
    const centerWeight = 0.08;
    const score = area - centerWeight * dist2 - thinScore * 20;

    if (score > bestScore || (score === bestScore && area > bestArea)) {
      bestScore = score;
      bestArea = area;
      bestIdxs = idxs;
    }
  }

  if (!bestIdxs || bestArea < minArea) {
    return { binFiltered: new Uint8Array(n), hasDigit: false };
  }

  const binFiltered = new Uint8Array(n);
  for (let i = 0; i < bestIdxs.length; i++) {
    binFiltered[bestIdxs[i]] = 1;
  }
  return { binFiltered, hasDigit: true };
}

function bboxFromBinary(bin, w, h) {
  let minX = w,
    minY = h,
    maxX = -1,
    maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y * w + x] === 1) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  return { minX, minY, maxX, maxY };
}

function imageSourceToTensor(source) {
  // Improved preprocessing pipeline for better accuracy:
  // - Use higher-res (PREP_SIZE) preprocessing
  // - Proper luminance grayscale
  // - Contrast normalization + gamma to boost thin strokes
  // - Adaptive thresholding for robust black/white separation
  // - Noise removal (morphology + keep largest connected component)
  // - Compute bbox and re-center before mapping to 28x28
  // - Output is normalized to [0, 1] (0/1 after thresholding)

  const { canvas: prepCanvas, ctx } = ensurePrepCanvas();

  // Disable smoothing to avoid blur when downscaling.
  ctx.imageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled = false;

  ctx.clearRect(0, 0, PREP_SIZE, PREP_SIZE);

  let srcW = 0;
  let srcH = 0;
  if (source instanceof HTMLVideoElement) {
    srcW = source.videoWidth || 0;
    srcH = source.videoHeight || 0;
  } else {
    srcW = source.naturalWidth || source.width || 0;
    srcH = source.naturalHeight || source.height || 0;
  }

  if (srcW <= 0 || srcH <= 0) {
    const empty = new Float32Array(28 * 28);
    return { tensor: tf.tensor4d(empty, [1, 28, 28, 1]), hasDigit: false };
  }

  // Crop to square at PREP_SIZE (so we can threshold/locate digit before resizing).
  const { sx, sy, side } = centerCropToSquare(ctx, srcW, srcH);
  ctx.drawImage(source, sx, sy, side, side, 0, 0, PREP_SIZE, PREP_SIZE);

  const { data } = ctx.getImageData(0, 0, PREP_SIZE, PREP_SIZE);
  const nPrep = PREP_SIZE * PREP_SIZE;
  const lum = new Float32Array(nPrep);

  // Corner stats for inversion + contrast normalization.
  const cornerBand = 6;
  let cornerSum = 0;
  let cornerSumSq = 0;
  let cornerCount = 0;

  for (let i = 0; i < nPrep; i++) {
    const x = i % PREP_SIZE;
    const y = (i / PREP_SIZE) | 0;

    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const L = (0.299 * r + 0.587 * g + 0.114 * b) / 255; // grayscale luminance
    lum[i] = L;

    const inCorner =
      (x < cornerBand || x >= PREP_SIZE - cornerBand) &&
      (y < cornerBand || y >= PREP_SIZE - cornerBand);
    if (inCorner) {
      cornerSum += L;
      cornerSumSq += L * L;
      cornerCount++;
    }
  }

  // Invert if background is bright (typical notebook/paper), so the digit becomes bright.
  const bgMean = cornerCount ? cornerSum / cornerCount : 0.5;
  const cornerSumBefore = cornerSum;
  const cornerSumSqBefore = cornerSumSq;

  if (bgMean > 0.5) {
    for (let i = 0; i < nPrep; i++) lum[i] = 1 - lum[i];

    // Update corner stats after inversion:
    // sum(1-L)=N - sum(L)
    // sum((1-L)^2)=N -2*sum(L)+sum(L^2)
    cornerSum = cornerCount - cornerSumBefore;
    cornerSumSq = cornerCount - 2 * cornerSumBefore + cornerSumSqBefore;
  }

  const meanBg = cornerCount ? cornerSum / cornerCount : 0.0;
  const varBg = cornerCount
    ? cornerSumSq / cornerCount - meanBg * meanBg
    : 0.0;
  const stdBg = Math.sqrt(Math.max(varBg, 1e-6));
  const denom = stdBg * 2 + 1e-6;

  // Contrast enhancement: normalize around background + gamma.
  for (let i = 0; i < nPrep; i++) {
    const v = (lum[i] - meanBg) / denom; // spread out digit from background
    const stretched = clamp01(v);
    lum[i] = Math.pow(stretched, CONTRAST_GAMMA);
  }

  // Adaptive thresholding to produce crisp black/white.
  const bin = adaptiveThresholdBinary(
    lum,
    PREP_SIZE,
    PREP_SIZE,
    ADAPTIVE_WINDOW_SIZE,
    ADAPTIVE_THRESHOLD_C
  );

  // Remove notebook lines/text specks and clean up strokes.
  const cleaned = morphologicalOpenClose(bin, PREP_SIZE, PREP_SIZE, MORPH_PASSES);

  // Keep the largest connected component (likely the digit).
  const cc = filterLargestComponent(
    cleaned,
    PREP_SIZE,
    PREP_SIZE,
    NO_DIGIT_MIN_FOREGROUND_PIXELS
  );

  if (!cc.hasDigit) {
    const empty = new Float32Array(28 * 28);
    return { tensor: tf.tensor4d(empty, [1, 28, 28, 1]), hasDigit: false };
  }

  // Center based on the digit bbox.
  const bbox = bboxFromBinary(cc.binFiltered, PREP_SIZE, PREP_SIZE);
  if (bbox.maxX < 0 || bbox.maxY < 0) {
    const empty = new Float32Array(28 * 28);
    return { tensor: tf.tensor4d(empty, [1, 28, 28, 1]), hasDigit: false };
  }

  const bw = bbox.maxX - bbox.minX + 1;
  const bh = bbox.maxY - bbox.minY + 1;
  const squareSide = Math.max(bw, bh);
  const digitSizeRatio = squareSide / PREP_SIZE;

  // Reject tiny components (often noise) so the digit occupies enough of the canvas.
  if (digitSizeRatio < 0.22) {
    const empty = new Float32Array(28 * 28);
    return { tensor: tf.tensor4d(empty, [1, 28, 28, 1]), hasDigit: false };
  }

  const cx = (bbox.minX + bbox.maxX) / 2;
  const cy = (bbox.minY + bbox.maxY) / 2;

  // Map a square around the bbox into the 28x28 input with margins.
  // Adaptive padding: if the digit is small, reduce padding so it occupies more of the 28x28 input.
  const padding = digitSizeRatio < 0.32 ? 3 : 4;
  const targetSize = 28 - padding * 2;

  let squareX0 = Math.round(cx - squareSide / 2);
  let squareY0 = Math.round(cy - squareSide / 2);

  const clampedSide = Math.min(squareSide, PREP_SIZE);
  squareX0 = Math.max(0, Math.min(PREP_SIZE - clampedSide, squareX0));
  squareY0 = Math.max(0, Math.min(PREP_SIZE - clampedSide, squareY0));

  const scale = clampedSide / targetSize;
  const out = new Float32Array(28 * 28);
  let digitPixels = 0;

  // Nearest-neighbor mapping: no smoothing.
  for (let oy = 0; oy < targetSize; oy++) {
    const yOut = padding + oy;
    const srcY = squareY0 + Math.min(clampedSide - 1, Math.floor(oy * scale));
    for (let ox = 0; ox < targetSize; ox++) {
      const xOut = padding + ox;
      const srcX = squareX0 + Math.min(clampedSide - 1, Math.floor(ox * scale));
      const v = cc.binFiltered[srcY * PREP_SIZE + srcX];
      out[yOut * 28 + xOut] = v;
      if (v === 1) digitPixels++;
    }
  }

  // If the mapped digit is still extremely sparse, treat it as "no digit".
  // This improves reliability for faint/thin digits and reduces false predictions from noise.
  const minMappedPixels = targetSize * targetSize * 0.06; // ~6% of target square
  const hasDigit = digitPixels >= minMappedPixels;

  if (!hasDigit) {
    const empty = new Float32Array(28 * 28);
    return { tensor: tf.tensor4d(empty, [1, 28, 28, 1]), hasDigit: false };
  }

  return { tensor: tf.tensor4d(out, [1, 28, 28, 1]), hasDigit: true };
}

async function predictFromSource(source) {
  if (!model) throw new Error("Model not loaded yet.");
  setError("");
  setLoading(true);

  try {
    const { tensor: input, hasDigit } = imageSourceToTensor(source);
    const output = model.predict(input);

    const probs = await output.data();
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    setPrediction(bestIdx, probs[bestIdx], hasDigit);

    input.dispose();
    output.dispose();
  } finally {
    setLoading(false);
  }
}

async function onFileSelected(file) {
  if (!file) return;
  clearAll();

  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsDataURL(file);
  });

  showPreviewFromDataUrl(dataUrl);

  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  await predictFromSource(img);
}

function openCameraModal() {
  els.cameraModal.classList.remove("hidden");
  els.cameraModal.classList.add("flex");
}

function closeCameraModal() {
  els.cameraModal.classList.add("hidden");
  els.cameraModal.classList.remove("flex");
}

async function startCamera() {
  setError("");
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    els.video.srcObject = cameraStream;
    await els.video.play();
  } catch (e) {
    setError(
      "Camera access failed. Please allow camera permissions, or use Upload Image."
    );
    throw e;
  }
}

function stopCamera() {
  if (!cameraStream) return;
  for (const track of cameraStream.getTracks()) track.stop();
  cameraStream = null;
  els.video.srcObject = null;
}

async function captureFromCamera() {
  if (!els.video.srcObject) return;

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = els.video.videoWidth;
  tmpCanvas.height = els.video.videoHeight;
  const ctx = tmpCanvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0);

  const dataUrl = tmpCanvas.toDataURL("image/png");
  showPreviewFromDataUrl(dataUrl);

  await predictFromSource(els.video);

  closeCameraModal();
  stopCamera();
}

async function init() {
  els.fileInput = $("fileInput");
  els.uploadBtn = $("uploadBtn");
  els.cameraBtn = $("cameraBtn");
  els.clearBtn = $("clearBtn");

  els.previewEmpty = $("previewEmpty");
  els.previewImg = $("previewImg");

  els.predDigit = $("predDigit");
  els.confText = $("confText");
  els.confBar = $("confBar");
  els.spinner = $("spinner");
  els.errorBox = $("errorBox");

  els.modelStatus = $("modelStatus");
  els.modelDot = $("modelDot");

  els.cameraModal = $("cameraModal");
  els.video = $("video");
  els.captureBtn = $("captureBtn");
  els.closeCameraBtn = $("closeCameraBtn");

  els.workCanvas = $("workCanvas");

  els.uploadBtn.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", e =>
    onFileSelected(e.target.files && e.target.files[0])
  );

  els.clearBtn.addEventListener("click", clearAll);

  els.cameraBtn.addEventListener("click", async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera is not supported in this browser. Please use Upload Image.");
      return;
    }
    openCameraModal();
    try {
      await startCamera();
    } catch {
      closeCameraModal();
      stopCamera();
    }
  });

  els.closeCameraBtn.addEventListener("click", () => {
    closeCameraModal();
    stopCamera();
  });

  els.cameraModal.addEventListener("click", e => {
    if (e.target === els.cameraModal) {
      closeCameraModal();
      stopCamera();
    }
  });

  els.captureBtn.addEventListener("click", captureFromCamera);

  setPrediction(null, 0);
  setLoading(false);

  try {
    await loadModel();
  } catch (e) {
    console.error(e);
    setModelStatus("Model failed to load", false);
    setError(
      "Model failed to load. Make sure the `model/` folder is deployed and accessible."
    );
  }
}

window.addEventListener("DOMContentLoaded", init);

