#!/bin/bash
set -e

echo "=================================================="
echo "🚀 MASTER SETUP: AlphaGeometry & Simplex Distillery"
echo "   Hardware target: RTX 5090 Ti (CUDA 12)"
echo "=================================================="

# Assumiamo di lanciare lo script da dentro simplex-distillery
SIMPLEX_DIR=$(pwd)
WORKSPACE_DIR=$(dirname "$SIMPLEX_DIR")
AG_DIR="$WORKSPACE_DIR/alphageometry"

echo "📂 Directory di lavoro:"
echo "   - Simplex: $SIMPLEX_DIR"
echo "   - Workspace: $WORKSPACE_DIR"
echo "   - AlphaGeometry: $AG_DIR"
echo "--------------------------------------------------"

# 1. Controllo/Clonazione AlphaGeometry
if [ ! -d "$AG_DIR" ]; then
    echo "⬇️  Clonazione repository AlphaGeometry..."
    cd "$WORKSPACE_DIR"
    git clone https://github.com/and-per-i/alphageometry.git
    cd "$SIMPLEX_DIR"
else
    echo "✅ Repository AlphaGeometry già presente."
fi

# 2. Creazione Ambiente Virtuale Unificato
VENV_PATH="$WORKSPACE_DIR/venv_ag_cuda"
# 1. Installazione Python 3.10 (per compatibilità AlphaGeometry)
echo "🐍 Installazione Python 3.10..."
if ! command -v python3.10 &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# 2. Creazione ambiente virtuale con Python 3.10
echo "🏗️  Creazione ambiente virtuale (Python 3.10)..."
python3.10 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

# 3. Installazione dipendenze CUDA 12
echo "📦 Installazione dipendenze JAX/Flax (CUDA 12) e utility..."
pip install -r "$SIMPLEX_DIR/distillation/requirements-cuda.txt"

# HOTFIX: Fix JAX 0.4.35 bug (pathlib TypeError) - Dinamico per versione Python
echo "🔧 Applicato hotfix per JAX 0.4.35..."
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
sed -i 's/cuda_nvcc_path = pathlib.Path(cuda_nvcc.__file__).parent/cuda_nvcc_path = pathlib.Path(getattr(cuda_nvcc, "__file__", "x") or "x").parent/g' "$VENV_PATH/lib/python$PY_VER/site-packages/jax/_src/lib/__init__.py" 2>/dev/null || true

# 4. Libreria Meliad (Strategia Multi-Livello: HuggingFace -> Docker)
echo "⬇️  Configurazione Meliad Library..."
cd "$AG_DIR"
MELIAD_PATH="meliad_lib/meliad"

if [ ! -d "$MELIAD_PATH" ]; then
    echo "   Tentativo 1: Download da HuggingFace (Versione Congelata)..."
    if wget -q https://huggingface.co/Wauplin/alphageometry/resolve/main/meliad_lib.tar.gz; then
        tar -xzf meliad_lib.tar.gz
        rm meliad_lib.tar.gz
        echo "   ✅ Download completato da HuggingFace."
    else
        echo "   Tentativo 2: Estrazione da Docker (Installazione Docker se necessario)..."
        if ! command -v docker &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y docker.io
        fi
        sudo systemctl start docker || true
        sudo docker pull gcr.io/t5-data/alphageometry
        sudo docker create --name ag_temp_extract gcr.io/t5-data/alphageometry
        sudo docker cp ag_temp_extract:/app/meliad_lib .
        sudo docker rm ag_temp_extract
        echo "   ✅ Estrazione da Docker completata."
    fi
fi



# Assicuriamoci che meliad_lib e meliad siano pacchetti Python validi
touch meliad_lib/__init__.py
touch meliad_lib/meliad/__init__.py
> "$MELIAD_PATH/transformer/__init__.py"

# HOTFIX: Namespace assoluto per evitare conflitti di import
echo "🔧 Applicato hotfix definitivo per gli import"
sed -i 's/from transformer import \([a-zA-Z0-9_]*\)/import meliad_lib.meliad.transformer.\1 as \1/g' "$AG_DIR"/*.py 2>/dev/null || true
sed -i 's/import transformer\./import meliad_lib.meliad.transformer./g' "$AG_DIR"/*.py 2>/dev/null || true
sed -i 's/from transformer import /from meliad_lib.meliad.transformer import /g' "$MELIAD_PATH/transformer/"*.py 2>/dev/null || true




# 5. Pesi AlphaGeometry
echo "⬇️  Controllo Pesi AlphaGeometry (150M)..."
if [ ! -d "ag_ckpt_vocab" ]; then
    echo "   Scarico i pesi ufficiali dal mirror Hugging Face (più stabile di GDrive)..."
    pip install huggingface_hub
    
    # Usa il nuovo tool 'hf' per scaricare l'intero repo dentro la cartella ag_ckpt_vocab
    hf download Wauplin/alphageometry --local-dir ag_ckpt_vocab
else
    echo "✅ Pesi già presenti."
fi

echo "=================================================="
echo "🎉 SETUP COMPLETATO CON SUCCESSO!"
echo "=================================================="
echo ""
echo "Per iniziare a lavorare, attiva l'ambiente unificato:"
echo "👉 source $VENV_PATH/bin/activate"
echo ""
echo "Per l'estrazione logit (Simplex):"
echo "   cd $SIMPLEX_DIR"
echo "   $VENV_PATH/bin/python distillation/extract_logits.py"
echo ""
echo "Per la dimostrazione nativa (AlphaGeometry):"
echo "   cd $AG_DIR"
echo "   $VENV_PATH/bin/bash run_cuda.sh"
echo "=================================================="

