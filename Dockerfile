# Usa un'immagine base con supporto CUDA 12.1 per la RTX 5090 Ti
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Evita interazioni durante l'installazione
ENV DEBIAN_FRONTEND=noninteractive

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro
WORKDIR /app

# Copia i requisiti e installa le dipendenze Python
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    transformers>=4.38.0 \
    accelerate>=0.28.0 \
    datasets \
    pyarrow \
    sentencepiece \
    einops \
    triton \
    wandb \
    scipy

# Copia il codice del Teacher (Newclid_Transformer)
COPY Newclid_Transformer /app/Newclid_Transformer
# Copia il motore simbolico (Newclid)
COPY Newclid /app/Newclid
# Copia il codice dello Student (simplex-distillery)
COPY simplex-distillery /app/simplex-distillery

# Imposta il PYTHONPATH per includere i sorgenti
ENV PYTHONPATH="/app/simplex-distillery:/app/Newclid_Transformer/src:${PYTHONPATH}"

# Directory per i risultati
RUN mkdir -p /app/distill_results

# Comando di default: lancia la distillazione massiva
CMD ["python3", "/app/simplex-distillery/distillation/run_distill_cloud.py"]
