#!/bin/bash
set -e

echo "🚀 Inizializzazione ambiente AlphaGeometry per RTX 5090 Ti..."

# 1. Crea l'ambiente virtuale
python3 -m venv venv_cuda
source venv_cuda/bin/activate

# 2. Installa pip aggiornato
pip install --upgrade pip

# 3. Installa le dipendenze per CUDA 12
echo "📦 Installazione dipendenze JAX/Flax per CUDA 12..."
pip install -r requirements-cuda.txt

echo "✅ Ambiente pronto! Per attivarlo usa: source venv_cuda/bin/activate"
echo "Per lanciare l'estrazione: python distillation/extract_logits.py"
