"use strict";

const phases = ["PRE", "EARLY", "LATE"];
const maxPhaseBytes = 5 * 1024 * 1024;
const form = document.getElementById("prediction-form");
const button = document.getElementById("predict-button");
const statusMessage = document.getElementById("status-message");
const errorMessage = document.getElementById("error-message");
const resultSection = document.getElementById("result-section");
const previewUrls = new Map();

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  statusMessage.textContent = "Corrige la entrada e inténtalo de nuevo.";
}

function clearResult() {
  resultSection.hidden = true;
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

for (const phase of phases) {
  document.getElementById(phase).addEventListener("change", (event) => {
    clearResult();
    const oldUrl = previewUrls.get(phase);
    if (oldUrl) URL.revokeObjectURL(oldUrl);
    const file = event.target.files[0];
    const image = document.getElementById(`preview-${phase}`);
    const empty = document.getElementById(`empty-${phase}`);
    if (file) {
      const url = URL.createObjectURL(file);
      previewUrls.set(phase, url);
      image.src = url;
      image.hidden = false;
      empty.hidden = true;
    } else {
      image.removeAttribute("src");
      image.hidden = true;
      empty.hidden = false;
    }
    statusMessage.textContent = "Pulsa «Validar y estimar» cuando estén las tres fases.";
  });
}

async function drawEnhancement() {
  const pre = await createImageBitmap(document.getElementById("PRE").files[0]);
  const early = await createImageBitmap(document.getElementById("EARLY").files[0]);
  const scratch = document.createElement("canvas");
  scratch.width = scratch.height = 256;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.drawImage(pre, 0, 0, 256, 256);
  const preData = context.getImageData(0, 0, 256, 256).data;
  context.drawImage(early, 0, 0, 256, 256);
  const earlyData = context.getImageData(0, 0, 256, 256).data;
  const output = context.createImageData(256, 256);
  for (let pixel = 0; pixel < 256 * 256; pixel++) {
    const index = pixel * 4;
    const difference = earlyData[index] - preData[index];
    const intensity = Math.max(0, Math.min(255, Math.round(128 + difference / 2)));
    output.data[index] = intensity;
    output.data[index + 1] = intensity;
    output.data[index + 2] = intensity;
    output.data[index + 3] = 255;
  }
  document.getElementById("enhancement-map").getContext("2d").putImageData(output, 0, 0);
  pre.close();
  early.close();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearResult();
  const payload = new FormData();
  for (const phase of phases) {
    const file = document.getElementById(phase).files[0];
    if (!file) return showError(`Falta la fase ${phase}.`);
    if (!file.name.toLowerCase().endsWith(".png") || file.size > maxPhaseBytes) {
      return showError(`${phase}: selecciona un PNG de hasta 5 MB.`);
    }
    payload.append(phase, file);
  }
  button.disabled = true;
  statusMessage.textContent = "Validando imágenes y calculando la estimación…";
  try {
    const response = await fetch("/api/predict", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "No se pudo procesar la muestra.");
    document.getElementById("probability").textContent = `${(data.probability_pcr * 100).toFixed(1)} %`;
    document.getElementById("classification").textContent = `Clase estimada: pCR=${data.predicted_class}. Umbral: ${data.threshold.toFixed(3)}.`;
    document.getElementById("threshold").textContent = data.threshold.toFixed(3);
    document.getElementById("model-version").textContent = data.model_version;
    document.getElementById("model-checksum").textContent = data.model_sha256;
    document.getElementById("device").textContent = data.device;
    document.getElementById("latency").textContent = `${data.latency_ms.toFixed(1)} ms`;
    try { await drawEnhancement(); } catch { /* La predicción sigue siendo válida. */ }
    resultSection.hidden = false;
    statusMessage.textContent = data.warnings?.length
      ? `Estimación terminada. ${data.warnings.join(" ")}`
      : "Estimación terminada. Puedes cargar otra muestra sin reiniciar.";
  } catch (error) {
    showError(error.message || "No se pudo conectar con el servidor local.");
  } finally {
    button.disabled = false;
  }
});

fetch("/api/status").then((response) => response.json()).then((data) => {
  if (!data.model_configured) {
    statusMessage.textContent = "La página está lista. Falta configurar un modelo entrenado para estimar.";
  }
}).catch(() => {
  statusMessage.textContent = "Abre esta página desde el servidor local para estimar.";
});
