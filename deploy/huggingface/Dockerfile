# Hugging Face Space (SDK Docker): interfaz React + API Flask en un solo contenedor.
# El código se descarga de GitHub y los pesos de E57 del repositorio de modelo en el arranque.
# Para desplegar otra versión del código, cambia REF y reconstruye el Space.
ARG REPO=https://github.com/rubenelices/AnalisisCNN.git
ARG REF=main

FROM node:20-slim AS interfaz
ARG REPO
ARG REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${REF} ${REPO} /src
WORKDIR /src/frontend
RUN npm ci && npm run build

FROM python:3.12-slim
ARG REPO
ARG REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
RUN git clone --depth 1 --branch ${REF} ${REPO} $HOME/app
WORKDIR $HOME/app
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r deploy/requirements.txt
COPY --from=interfaz --chown=user /src/frontend/dist frontend/dist

ENV BREAST_PCR_HF_REPO=RubenElices/BreastDCEDL
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
