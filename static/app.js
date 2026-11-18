const input = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const fileList = document.querySelector("#file-list");
const processButton = document.querySelector("#process-button");
const errorMessage = document.querySelector("#error-message");
const resultsSection = document.querySelector("#results-section");
const resultsGrid = document.querySelector("#results-grid");
const openCameraButton = document.querySelector("#open-camera-button");
const cameraPanel = document.querySelector("#camera-panel");
const cameraPreview = document.querySelector("#camera-preview");
const captureButton = document.querySelector("#capture-button");
const closeCameraButton = document.querySelector("#close-camera-button");

const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
const maxFileSize = 10 * 1024 * 1024;
const maxFiles = 10;
let selectedFiles = [];
let previewUrls = new Map();
let cameraStream = null;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function formatSize(bytes) {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function imageUrl(file) {
  if (!previewUrls.has(file)) previewUrls.set(file, URL.createObjectURL(file));
  return previewUrls.get(file);
}

function addFiles(fileCollection) {
  clearError();
  const incoming = Array.from(fileCollection);
  const invalid = incoming.find((file) => !allowedTypes.has(file.type));
  if (invalid) {
    showError(`${invalid.name} is not a supported image. Choose a JPG, PNG, WEBP, or BMP file.`);
    return;
  }
  const oversized = incoming.find((file) => file.size > maxFileSize);
  if (oversized) {
    showError(`${oversized.name} is larger than 10 MB. Choose a smaller image.`);
    return;
  }

  const combined = [...selectedFiles, ...incoming];
  if (combined.length > maxFiles) {
    showError(`Choose no more than ${maxFiles} images at a time.`);
    return;
  }
  selectedFiles = combined;
  renderFiles();
}

function renderFiles() {
  fileList.replaceChildren();
  selectedFiles.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "file-item";

    const image = document.createElement("img");
    image.className = "file-thumb";
    image.src = imageUrl(file);
    image.alt = "";

    const meta = document.createElement("div");
    meta.className = "file-meta";
    const name = document.createElement("div");
    name.className = "file-name";
    name.textContent = file.name;
    const size = document.createElement("div");
    size.className = "file-size";
    size.textContent = formatSize(file.size);
    meta.append(name, size);

    const remove = document.createElement("button");
    remove.className = "remove-file";
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.addEventListener("click", () => {
      const [removed] = selectedFiles.splice(index, 1);
      URL.revokeObjectURL(previewUrls.get(removed));
      previewUrls.delete(removed);
      renderFiles();
    });

    row.append(image, meta, remove);
    fileList.append(row);
  });
  processButton.disabled = selectedFiles.length === 0;
  openCameraButton.disabled = selectedFiles.length >= maxFiles;
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraPreview.srcObject = null;
  cameraPanel.hidden = true;
  captureButton.disabled = true;
}

async function openCamera() {
  clearError();
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Camera access requires a supported browser and a secure connection (HTTPS or localhost).");
    return;
  }

  openCameraButton.disabled = true;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    cameraPreview.srcObject = cameraStream;
    cameraPanel.hidden = false;
    await cameraPreview.play();
    captureButton.disabled = false;
  } catch (error) {
    stopCamera();
    if (error instanceof Error && error.name === "NotAllowedError") {
      showError("Camera access was denied. Allow camera permission in your browser and try again.");
    } else if (error instanceof Error && error.name === "NotFoundError") {
      showError("No camera was found on this device.");
    } else {
      showError("Could not start the camera. Check that it is connected and not being used by another app.");
    }
  } finally {
    openCameraButton.disabled = selectedFiles.length >= maxFiles;
  }
}

async function capturePhoto() {
  if (!cameraStream || !cameraPreview.videoWidth || !cameraPreview.videoHeight) return;

  captureButton.disabled = true;
  try {
    const scale = Math.min(1, 2560 / cameraPreview.videoWidth, 2560 / cameraPreview.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(cameraPreview.videoWidth * scale);
    canvas.height = Math.round(cameraPreview.videoHeight * scale);
    canvas.getContext("2d").drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) throw new Error("The camera image could not be created.");

    const filename = `camera-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
    stopCamera();
    addFiles([new File([blob], filename, { type: "image/jpeg" })]);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The camera image could not be created.");
    captureButton.disabled = false;
  }
}

function renderResults(results) {
  resultsGrid.replaceChildren();
  results.forEach((result, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${index * 55}ms`;

    const file = document.createElement("div");
    file.className = "result-file";
    const thumbnail = document.createElement("img");
    thumbnail.className = "result-thumb";
    thumbnail.src = imageUrl(selectedFiles[index]);
    thumbnail.alt = "";
    const meta = document.createElement("div");
    meta.style.minWidth = "0";
    const name = document.createElement("div");
    name.className = "result-name";
    name.textContent = result.filename;
    const time = document.createElement("div");
    time.className = "result-time";
    time.textContent = `${result.processing_time_ms} ms`;
    meta.append(name, time);
    file.append(thumbnail, meta);

    const readingBlock = document.createElement("div");
    readingBlock.className = "reading-block";
    const reading = document.createElement("div");
    reading.className = result.detected_reading === null ? "reading not-detected" : "reading";
    reading.textContent = result.detected_reading === null
      ? "Not detected"
      : (result.raw_text || String(result.detected_reading));
    const confidence = document.createElement("div");
    confidence.className = "confidence";
    const confidenceValue = document.createElement("strong");
    confidenceValue.textContent = `${Math.round(result.confidence_score * 100)}%`;
    confidence.append("Confidence ", confidenceValue);
    readingBlock.append(reading, confidence);

    card.append(file, readingBlock);
    resultsGrid.append(card);
  });
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.querySelector("#browse-button").addEventListener("click", () => input.click());
openCameraButton.addEventListener("click", openCamera);
captureButton.addEventListener("click", capturePhoto);
closeCameraButton.addEventListener("click", stopCamera);
dropZone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  input.click();
});
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});
input.addEventListener("change", () => {
  addFiles(input.files);
  input.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-over");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-over");
  });
}
dropZone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

processButton.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  clearError();
  processButton.disabled = true;
  processButton.classList.add("is-loading");
  processButton.querySelector("span").textContent = "Processing…";
  resultsSection.hidden = true;

  const body = new FormData();
  selectedFiles.forEach((file) => body.append("images", file));
  try {
    const response = await fetch("/api/detect", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The images could not be processed.");
    renderResults(payload.results);
  } catch (error) {
    showError(error instanceof Error ? error.message : "The images could not be processed.");
  } finally {
    processButton.disabled = selectedFiles.length === 0;
    processButton.classList.remove("is-loading");
    processButton.querySelector("span").textContent = "Process images";
  }
});

document.querySelector("#clear-button").addEventListener("click", () => {
  stopCamera();
  for (const url of previewUrls.values()) URL.revokeObjectURL(url);
  previewUrls = new Map();
  selectedFiles = [];
  renderFiles();
  resultsGrid.replaceChildren();
  resultsSection.hidden = true;
  clearError();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

window.addEventListener("pagehide", stopCamera);
