# 🚀 Cloud Training Guide - RTX 5090 32GB

Guida completa per training su GPU cloud con NVIDIA RTX 5090 32GB.

---

## 📋 Specifiche Hardware Target

- **GPU**: NVIDIA RTX 5090 32GB VRAM
- **RAM**: 64GB+ raccomandato
- **Storage**: 500GB+ SSD
- **CUDA**: 12.1+
- **Driver**: 550.54.15+

---

## ⚡ Quick Start (Setup + Training)

### 1. Setup Automatico
```bash
# Clone repository
git clone https://github.com/and-per-i/simplex-distillery.git
cd simplex-distillery

# Setup environment completo
chmod +x setup_cloud_5090.sh
./setup_cloud_5090.sh
```

Lo script automaticamente:
- ✅ Verifica NVIDIA drivers e CUDA
- ✅ Installa Python dependencies
- ✅ Configura PyTorch con CUDA 12.1
- ✅ Setup Git LFS
- ✅ Crea directory necessarie
- ✅ Verifica checkpoint e dataset

### 2. Attiva Environment
```bash
source venv/bin/activate
```

### 3. Pre-Flight Tests
```bash
# Test configurazione Teacher Model
python train_cloud_5090.py --run_tests --skip_forgia
```

**Expected Output:**
```
🧪 PRE-FLIGHT TESTS - Verifica Teacher Model
✅ Checkpoint Integrity: PASSED
✅ Teacher Loading: PASSED
✅ Vocab Compatibility: PASSED
✅ Forward Pass: PASSED
✅ Logits Shape: PASSED
✅ Memory Footprint: PASSED
✅ Teacher-Student Compatibility: PASSED

🎉 TUTTI I TEST PASSATI - Ready for Training!
```

### 4. Training Completo
```bash
# Opzione 1: Script auto-generato
./start_training.sh

# Opzione 2: Comando diretto con controllo completo
python train_cloud_5090.py \
    --full_pipeline \
    --run_tests \
    --num_epochs 4 \
    --temperature 4.0 \
    --alpha 0.5 \
    --output_dir runs/distill_5090
```

---

## 🔧 Configurazione Ottimizzata per RTX 5090

Il training è ottimizzato specificamente per RTX 5090:

### Performance Settings
```python
# Batch Size Ottimale
per_device_train_batch_size = 32  # Massimizza VRAM usage
gradient_accumulation_steps = 2    # Effective batch = 64

# Mixed Precision (2x speedup)
fp16 = True
fp16_opt_level = "O2"

# Memory Management
gradient_checkpointing = True      # Trade-off: +20% tempo, -50% memoria

# Data Loading
dataloader_num_workers = 4
dataloader_prefetch_factor = 2
```

### Expected Performance
- **Training speed**: ~2.5 ore per epoch (dataset 10K examples)
- **VRAM usage**: ~28GB peak (lascia margine per altre operazioni)
- **Throughput**: ~40-50 samples/sec

---

## 📊 Monitoring Durante Training

### Terminal 1: Training
```bash
source venv/bin/activate
./start_training.sh
```

### Terminal 2: GPU Monitoring
```bash
./monitor_gpu.sh
```

### Terminal 3: TensorBoard
```bash
source venv/bin/activate
tensorboard --logdir runs/distill_5090/logs --port 6006
```

Accedi a: `http://localhost:6006`

---

## 🧪 Pre-Flight Tests Dettagliati

I test verificano:

### Test 1: Checkpoint Integrity
- ✅ `params.sav` esiste e dimensione corretta (~580MB)
- ✅ `cfg.sav` esiste
- ✅ `vocab.model` esiste e dimensione corretta (~14KB)

### Test 2: Teacher Loading
- ✅ Modello carica senza errori
- ✅ Parametri totali: ~151M

### Test 3: Vocab Compatibility
- ✅ Vocab size = 1024
- ✅ Tokenizzazione funzionante
- ✅ EOS token ID = 263

### Test 4: Forward Pass
- ✅ Input dummy → output valido
- ✅ Latency < 100ms

### Test 5: Logits Shape
- ✅ Shape corretta: `(batch, seq_len, 1024)`
- ✅ No NaN, no Inf
- ✅ Distribuzione valori ragionevole

### Test 6: Memory Footprint
- ✅ Teacher usa < 16GB VRAM
- ✅ Lascia spazio per Student (~12-14GB)

### Test 7: Teacher-Student Compatibility
- ✅ Batch processing funzionante
- ✅ Shape logits compatibili

---

## 📁 Struttura Files Generati

```
simplex-distillery/
├── checkpoints/
│   └── studente_inizializzato.pt    # Fase 1 output (~150MB)
├── runs/
│   └── distill_5090/
│       ├── checkpoint-500/          # Checkpoint intermedi
│       ├── checkpoint-1000/
│       ├── logs/                    # TensorBoard logs
│       └── pytorch_model.bin        # Modello finale
├── logs/
│   └── training_5090.log           # Training log file
└── results/
    └── final_metrics.json          # Metriche finali
```

---

## 🐛 Troubleshooting

### Errore: "CUDA out of memory"
**Soluzione:**
```bash
# Riduci batch size
python train_cloud_5090.py --full_pipeline --batch_size 16
```

### Errore: "params.sav not found"
**Soluzione:**
```bash
# Verifica Git LFS
git lfs pull

# Verifica file
ls -lh pt_ckpt/params.sav
```

### Pre-flight test fallisce
**Soluzione:**
```bash
# Pulisci cache GPU
python -c "import torch; torch.cuda.empty_cache()"

# Rilancia test
python train_cloud_5090.py --run_tests --skip_forgia
```

### Training lento
**Check:**
1. Verifica GPU utilization: `nvidia-smi`
2. Verifica FP16 enabled: controlla logs "FP16: True"
3. Verifica dataloader workers: aumenta a 8 se CPU potente

---

## 🎯 Pipeline Completa (4 Fasi)

### Fase 1: Forgia SVD (Automatica)
```bash
# Genera studente_inizializzato.pt
# Tempo: ~5-10 minuti
python scripts/fase1_forgia_svd.py --teacher_path pt_ckpt --output checkpoints/studente_inizializzato.pt
```

### Fase 2-3: Training + Progressive Pruning
```bash
# Training con KD + pruning graduale
# Tempo: ~10-12 ore (4 epoche)
python train_cloud_5090.py --full_pipeline --num_epochs 4
```

**Timeline Pruning:**
- Epoca 1: 12 layers attivi
- Epoca 2: Bypass layer 2, 10 → 10 layers attivi
- Epoca 3: Bypass layer 3, 9 → 8 layers attivi  
- Epoca 4: Bypass layer 5, 7 → 6 layers attivi (3 standard + 3 simpliciali)

### Fase 4: Estrazione Fisica
```bash
# Compatta modello finale (rimuove layer bypassed)
python scripts/fase4_estrazione_fisica.py \
    --checkpoint runs/distill_5090/pytorch_model.bin \
    --output checkpoints/Davide_2Simplex_40M_Finale.pt \
    --verify
```

---

## 📈 Metriche Attese

### Durante Training
```
Epoch 1: Loss ~2.5 → ~1.8
Epoch 2: Loss ~1.8 → ~1.4 (dopo pruning layer 2,10)
Epoch 3: Loss ~1.4 → ~1.2 (dopo pruning layer 3,9)
Epoch 4: Loss ~1.2 → ~1.0 (dopo pruning layer 5,7)
```

### Modello Finale
- **Parametri**: ~40M (vs 151M Teacher = 73% riduzione)
- **Dimensione file**: ~150MB
- **Layers**: 6 (3 standard + 3 simpliciali)
- **Performance**: ~85-90% del Teacher su test set

---

## 🚀 Dopo il Training

### 1. Verifica Modello Finale
```bash
python scripts/fase4_estrazione_fisica.py --checkpoint runs/distill_5090/pytorch_model.bin --output checkpoints/finale.pt --verify
```

### 2. Test Inferenza
```bash
python evaluate_teacher.py --model checkpoints/finale.pt --problem orthocenter
```

### 3. Deploy
```bash
# Copia modello finale per deployment
cp checkpoints/Davide_2Simplex_40M_Finale.pt /path/to/deployment/
```

---

## 💾 Backup e Checkpoints

### Backup Consigliati
```bash
# Durante training (ogni 500 steps)
runs/distill_5090/checkpoint-{step}/

# Finale
checkpoints/Davide_2Simplex_40M_Finale.pt

# Logs
logs/training_5090.log
runs/distill_5090/logs/  # TensorBoard
```

### Upload a Cloud Storage
```bash
# Google Cloud
gsutil -m cp -r runs/distill_5090 gs://your-bucket/

# AWS S3
aws s3 sync runs/distill_5090 s3://your-bucket/distill_5090/
```

---

## 🎓 Ottimizzazioni Avanzate

### Multi-GPU (se disponibili)
```bash
# Distributed Data Parallel
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_cloud_5090.py --full_pipeline
```

### Gradient Accumulation Personalizzato
```python
# Per batch virtuale = 128
gradient_accumulation_steps = 4  # 32 * 4 = 128
```

### Custom Learning Rate Schedule
```python
# Cosine annealing con warmup
lr_scheduler_type = "cosine"
warmup_ratio = 0.05
```

---

## 📞 Support

- **Issues**: https://github.com/and-per-i/simplex-distillery/issues
- **Docs**: `FIX_SUMMARY.md`, `QUICK_START.md`
- **Pre-flight Tests**: Sempre esegui prima del training!

---

**Training ottimizzato per RTX 5090 - Built for Speed! 🚀**
