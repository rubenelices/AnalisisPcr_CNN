# Interfaz web

React + Vite. Toda la inferencia ocurre en el servidor Python ([app.py](../app.py)) con el mismo preprocesado que en el entrenamiento; la interfaz solo envía las tres fases y muestra la respuesta.

```bash
npm ci
npm run dev     # desarrollo; /api se redirige a app.py en el puerto 8000
npm run build   # compila en dist/, que app.py sirve en /
```

`VITE_API_URL` permite apuntar a un servidor en otro dominio (ver [deploy/README.md](../deploy/README.md)). Las cifras y gráficas de Resultados se generan con `python -m src.web_resultados` y `python -m src.web_graficas` en `public/resultados/`.
