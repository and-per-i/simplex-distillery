# Immagine base snella con CUDA 12.1
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Evita interazioni durante l'installazione
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Installa solo il minimo indispensabile del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installazione dipendenze Python in un unico colpo per ottimizzare i layer
# Usiamo versioni bloccate per stabilità
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    "transformers>=4.38.0" \
    "accelerate>=0.28.0" \
    datasets \
    pyarrow \
    polars \
    sentencepiece \
    einops \
    triton \
    wandb \
    scipy

# Copia solo i file necessari (grazie al .dockerignore escludiamo i gigabyte di troppo)
COPY Newclid_Transformer /app/Newclid_Transformer
COPY Newclid /app/Newclid
COPY simplex-distillery /app/simplex-distillery

# Configurazione ambiente
ENV PYTHONPATH="/app/simplex-distillery:/app/Newclid_Transformer/src:${PYTHONPATH}"

# Crea la directory risultati
RUN mkdir -p /app/distill_results

# Comando di avvio
CMD ["python3", "/app/simplex-distillery/distillation/run_distill_cloud.py"]
