let model = null;
let cameraStream = null;

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

function setPrediction(digit, confidence) {
  if (digit === null || digit === undefined) {
    els.predDigit.textContent = "—";
    els.confText.textContent = "—";
    els.confBar.style.width = "0%";
    return;
  }
  els.predDigit.textContent = String(digit);
  const pct = Math.max(0, Math.min(1, confidence)) * 100;
  els.confText.textContent = `${pct.toFixed(1)}%`;
  els.confBar.style.width = `${pct.toFixed(1)}%`;
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
  model = await tf.loadLayersModel("./model/model.json");
  setModelStatus("Model ready", true);
}

function centerCropToSquare(drawCtx, imgW, imgH) {
  const side = Math.min(imgW, imgH);
  const sx = Math.floor((imgW - side) / 2);
  const sy = Math.floor((imgH - side) / 2);
  drawCtx.__crop = { sx, sy, side };
  return drawCtx.__crop;
}

function imageSourceToTensor(source) {
  const canvas = els.workCanvas;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let w = 0;
  let h = 0;
  if (source instanceof HTMLVideoElement) {
    w = source.videoWidth;
    h = source.videoHeight;
  } else {
    w = source.naturalWidth || source.width;
    h = source.naturalHeight || source.height;
  }

  const { sx, sy, side } = centerCropToSquare(ctx, w, h);
  ctx.drawImage(source, sx, sy, side, side, 0, 0, 28, 28);

  const { data } = ctx.getImageData(0, 0, 28, 28);

  let sum = 0;
  const gray = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = data[i * 4 + 0];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const v = (r + g + b) / (3 * 255);
    gray[i] = v;
    sum += v;
  }

  const mean = sum / gray.length;
  const shouldInvert = mean > 0.55;
  if (shouldInvert) {
    for (let i = 0; i < gray.length; i++) gray[i] = 1 - gray[i];
  }

  // Light denoise/contrast: clamp and normalize-ish
  // (kept gentle so it doesn't destroy strokes)
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    gray[i] = v < 0.05 ? 0 : v;
  }

  return tf.tensor4d(gray, [1, 28, 28, 1]);
}

async function predictFromSource(source) {
  if (!model) throw new Error("Model not loaded yet.");
  setError("");
  setLoading(true);

  try {
    const input = imageSourceToTensor(source);
    const output = model.predict(input);

    const probs = await output.data();
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    setPrediction(bestIdx, probs[bestIdx]);

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
