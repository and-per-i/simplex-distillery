#!/bin/bash
# ============================================================================
# SETUP SCRIPT PER CLOUD - RTX 5090 32GB
# ============================================================================
# 
# Questo script configura l'ambiente cloud per training ottimizzato.
#
# Usage:
#   chmod +x setup_cloud_5090.sh
#   ./setup_cloud_5090.sh
#
# ============================================================================

set -e  # Exit on error

echo "🚀 Setup Environment per RTX 5090 32GB Training"
echo "=" | tr '\n' '=' | head -c 80; echo

# ============================================================================
# 1. CHECK PREREQUISITES
# ============================================================================

echo ""
echo "📋 Step 1: Verifica Prerequisites"
echo "-" | tr '\n' '-' | head -c 70; echo

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi non trovato! Installa NVIDIA drivers."
    exit 1
fi

echo "✅ NVIDIA Drivers installati"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trovato!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python: $PYTHON_VERSION"

# Check Git LFS
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS non installato - installazione..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
fi

echo "✅ Git LFS installato"

# ============================================================================
# 2. CLONE REPOSITORY (se necessario)
# ============================================================================

echo ""
echo "📋 Step 2: Repository Setup"
echo "-" | tr '\n' '-' | head -c 70; echo

if [ ! -d "simplex-distillery" ]; then
    echo "📦 Clonazione repository..."
    git clone https://github.com/and-per-i/simplex-distillery.git
    cd simplex-distillery
else
    echo "✅ Repository già presente"
    cd simplex-distillery
    git pull origin main
fi

# Pull LFS files
echo "📦 Download file LFS..."
git lfs pull

# ============================================================================
# 3. VIRTUAL ENVIRONMENT
# ============================================================================

echo ""
echo "📋 Step 3: Virtual Environment"
echo "-" | tr '\n' '-' | head -c 70; echo

if [ ! -d "venv" ]; then
    echo "🔨 Creazione virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ Virtual environment attivato"

# ============================================================================
# 4. INSTALL DEPENDENCIES
# ============================================================================

echo ""
echo "📋 Step 4: Installazione Dipendenze"
echo "-" | tr '\n' '-' | head -c 70; echo

# Upgrade pip
pip install --upgrade pip setuptools wheel

# PyTorch con CUDA 12.1 (ottimizzato per RTX 5090)
echo "📦 Installazione PyTorch con CUDA 12.1..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Verifica PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Transformers e HuggingFace
echo "📦 Installazione Transformers..."
pip install transformers==4.45.0 accelerate==0.34.0 datasets==2.20.0

# Altre dipendenze
echo "📦 Installazione altre dipendenze..."
pip install -r requirements.txt

# Dipendenze specifiche per cloud
pip install tensorboard wandb  # Logging
pip install nvidia-ml-py3  # GPU monitoring
pip install psutil  # System monitoring

# ============================================================================
# 5. VERIFY INSTALLATION
# ============================================================================

echo ""
echo "📋 Step 5: Verifica Installazione"
echo "-" | tr '\n' '-' | head -c 70; echo

python3 -c "
import torch
import transformers
import numpy as np

print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ CUDA Capability:', torch.cuda.get_device_capability(0))
    print('✅ Total VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
"

# ============================================================================
# 6. DATASET VERIFICATION
# ============================================================================

echo ""
echo "📋 Step 6: Verifica Dataset"
echo "-" | tr '\n' '-' | head -c 70; echo

if [ -d "data" ]; then
    echo "✅ Directory data/ presente"
    ls -lh data/
else
    echo "⚠️  Directory data/ non trovata"
    mkdir -p data
    echo "📁 Creata directory data/"
fi

# ============================================================================
# 7. CHECKPOINT VERIFICATION
# ============================================================================

echo ""
echo "📋 Step 7: Verifica Checkpoint Teacher"
echo "-" | tr '\n' '-' | head -c 70; echo

if [ -d "pt_ckpt" ]; then
    echo "✅ Directory pt_ckpt/ presente"
    du -sh pt_ckpt/
    ls -lh pt_ckpt/
    
    # Verifica file essenziali
    if [ -f "pt_ckpt/params.sav" ]; then
        echo "✅ params.sav presente"
    else
        echo "❌ params.sav mancante!"
    fi
    
    if [ -f "pt_ckpt/cfg.sav" ]; then
        echo "✅ cfg.sav presente"
    else
        echo "⚠️  cfg.sav mancante"
    fi
    
    if [ -f "pt_ckpt/vocab.model" ]; then
        echo "✅ vocab.model presente"
    else
        echo "❌ vocab.model mancante!"
    fi
else
    echo "❌ Directory pt_ckpt/ non trovata!"
    echo "   Download checkpoint AlphaGeometry manualmente"
fi

# ============================================================================
# 8. CREATE NECESSARY DIRECTORIES
# ============================================================================

echo ""
echo "📋 Step 8: Creazione Directory"
echo "-" | tr '\n' '-' | head -c 70; echo

mkdir -p checkpoints
mkdir -p runs/distill_5090
mkdir -p logs
mkdir -p results

echo "✅ Directory create"

# ============================================================================
# 9. SETUP GIT LFS TRACKING
# ============================================================================

echo ""
echo "📋 Step 9: Git LFS Setup"
echo "-" | tr '\n' '-' | head -c 70; echo

# Verifica .gitattributes
if [ -f ".gitattributes" ]; then
    echo "✅ .gitattributes configurato"
    git lfs track
else
    echo "⚠️  .gitattributes non trovato"
fi

# ============================================================================
# 10. FINAL SETUP
# ============================================================================

echo ""
echo "📋 Step 10: Setup Finale"
echo "-" | tr '\n' '-' | head -c 70; echo

# Crea script di avvio rapido
cat > start_training.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python train_cloud_5090.py \
    --full_pipeline \
    --run_tests \
    --num_epochs 4 \
    --data_path data/train_sequences.txt \
    --output_dir runs/distill_5090
EOF

chmod +x start_training.sh
echo "✅ Script start_training.sh creato"

# Crea script di monitoring
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
watch -n 1 nvidia-smi
EOF

chmod +x monitor_gpu.sh
echo "✅ Script monitor_gpu.sh creato"

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "=" | tr '\n' '=' | head -c 80; echo
echo "🎉 SETUP COMPLETATO!"
echo "=" | tr '\n' '=' | head -c 80; echo
echo ""
echo "🚀 Prossimi passi:"
echo ""
echo "1. Verifica checkpoint Teacher:"
echo "   ls -lh pt_ckpt/"
echo ""
echo "2. Verifica dataset:"
echo "   ls -lh data/"
echo ""
echo "3. Test rapido:"
echo "   source venv/bin/activate"
echo "   python train_cloud_5090.py --run_tests --skip_forgia"
echo ""
echo "4. Training completo:"
echo "   ./start_training.sh"
echo ""
echo "5. Monitoring GPU (terminale separato):"
echo "   ./monitor_gpu.sh"
echo ""
echo "=" | tr '\n' '=' | head -c 80; echo
