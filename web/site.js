"use strict";

const blocks = {
  1: {
    number: "Bloque 01",
    title: "Primeras características",
    description: "Dos convoluciones 3×3 combinan las tres fases y buscan patrones locales. MaxPool se aplica entre las dos convoluciones y reduce alto y ancho a la mitad.",
    rationale: "Hipótesis: empezar con 16 canales limita el coste sin perder la señal inicial. Su utilidad se evaluará en validación.",
    shape: "3 × 256 × 256 → 16 × 128 × 128",
    operations: "Conv–BN–ReLU → MaxPool 2×2 → Conv–BN–ReLU; kernel 3×3, padding 1, stride 1",
    parameters: "2.800",
    receptive: "8 × 8 píxeles",
  },
  2: {
    number: "Bloque 02",
    title: "Patrones intermedios",
    description: "Los 16 mapas del bloque anterior se transforman en 32. El segundo MaxPool se aplica entre las dos convoluciones de este bloque.",
    rationale: "Hipótesis: duplicar canales compensa la reducción espacial y permite representar patrones más variados; N01 compara pérdidas manteniendo la red completa fija, no el efecto aislado de este bloque.",
    shape: "16 × 128 × 128 → 32 × 64 × 64",
    operations: "Conv–BN–ReLU → MaxPool 2×2 → Conv–BN–ReLU; kernel 3×3, padding 1, stride 1",
    parameters: "13.952",
    receptive: "22 × 22 píxeles",
  },
  3: {
    number: "Bloque 03",
    title: "Contexto más amplio",
    description: "La representación pasa a 64 canales y 32 × 32 posiciones. Cada activación resume una zona mayor de la imagen original.",
    rationale: "Hipótesis: ganar contexto puede ayudar a integrar patrones espaciales y temporales. Aún no hay una ablación que demuestre su contribución por separado.",
    shape: "32 × 64 × 64 → 64 × 32 × 32",
    operations: "Conv–BN–ReLU → MaxPool 2×2 → Conv–BN–ReLU; kernel 3×3, padding 1, stride 1",
    parameters: "55.552",
    receptive: "50 × 50 píxeles",
  },
  4: {
    number: "Bloque 04",
    title: "Resumen de alto nivel",
    description: "La última etapa entrega 128 mapas de 16 × 16. Después, el promedio global resume cada mapa en un valor.",
    rationale: "Hipótesis: esta capacidad final podría captar relaciones complejas sin una capa densa grande. Su coste y posible sobreajuste se juzgarán con validación.",
    shape: "64 × 32 × 32 → 128 × 16 × 16",
    operations: "Conv–BN–ReLU → MaxPool 2×2 → Conv–BN–ReLU; kernel 3×3, padding 1, stride 1",
    parameters: "221.696",
    receptive: "106 × 106 píxeles",
  },
};

const detailFields = {
  number: document.getElementById("detail-number"),
  title: document.getElementById("detail-title"),
  description: document.getElementById("detail-description"),
  rationale: document.getElementById("detail-rationale"),
  shape: document.getElementById("detail-shape"),
  operations: document.getElementById("detail-operations"),
  parameters: document.getElementById("detail-parameters"),
  receptive: document.getElementById("detail-receptive"),
};

document.querySelectorAll("button[data-block]").forEach((button) => {
  button.addEventListener("click", () => {
    const block = blocks[button.dataset.block];
    if (!block) return;
    document.querySelectorAll("button[data-block]").forEach((node) => {
      const selected = node === button;
      node.classList.toggle("is-selected", selected);
      node.setAttribute("aria-pressed", String(selected));
    });
    for (const [key, element] of Object.entries(detailFields)) {
      element.textContent = block[key];
    }
  });
});

document.querySelectorAll(".mobile-nav a").forEach((link) => {
  link.addEventListener("click", () => {
    link.closest("details").open = false;
  });
});
