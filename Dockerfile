# Usa un'immagine base con supporto CUDA 11.8 per compatibilità con driver vecchi
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
# Useremo una versione stabile di Transformers per evitare i bug visti in locale
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    transformers==4.38.0 \
    datasets \
    pyarrow \
    accelerate \
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

# Installa Newclid se necessario (se è un pacchetto)
# RUN cd /app/Newclid_Transformer && pip3 install -e .

# Imposta il PYTHONPATH per includere i sorgenti
ENV PYTHONPATH="/app/simplex-distillery:/app/Newclid_Transformer/src:${PYTHONPATH}"

# Directory per i risultati
RUN mkdir -p /app/distill_results

# Comando di default: lancia la distillazione massiva
# Dovrai mappare il dataset dall'esterno o includerlo nell'immagine
CMD ["python3", "/app/simplex-distillery/distillation/run_distill_cloud.py"]
