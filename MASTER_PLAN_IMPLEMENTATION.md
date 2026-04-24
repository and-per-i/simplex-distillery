# 🎯 MASTER PLAN - Implementazione Completa

**Status**: ✅ **IMPLEMENTATO E PRONTO PER TRAINING**

---

## 📋 Overview

Implementazione completa della pipeline di distillazione avanzata che unisce **tre tecniche di frontiera**:

1. **SVD Compression** - Compressione intelligente 151M → 40M parametri
2. **Perturbed Identity (Identità Perturbata)** - K' = K + ε per layer simpliciali
3. **Progressive Pruning** - Riduzione graduale 12→6 layer senza amnesia

**Risultato**: Modello compatto (40M) con **2-Simplicial Attention** per ragionamento geometrico.

---

## 🏗️ Architettura delle 4 Fasi

### **FASE 1: LA FORGIA** 🔨
**File**: `scripts/fase1_forgia_svd.py`

**Processo**:
```python
Teacher (151M, 12 layers, dim=1024)
    ↓ SVD Compression (dimensione 1024 → 384)
    ↓ Per ogni layer:
    │   - Comprimi W_Q, W_K, W_V, W_O con SVD
    │   - Comprimi MLP proporzionalmente
    │   - Se layer simpliciale (indici 3, 7, 11):
    │       → Clona K: K' = K + rumore(1e-4)
    ↓
Student Inizializzato (40M, 12 layers compressi, dim=384)
```

**Output**: `checkpoints/studente_inizializzato.pt` (~150MB)

**Tempo stimato**: 5-10 minuti

---

### **FASE 2: SETUP DISTILLAZIONE** 🎓
**File**: `distillation/trainer_master.py`

**Componenti**:

1. **StudentModelProgressive** (`models/student_progressive.py`):
   ```python
   class SimplexAttention:
       # Attenzione 2-Simpliciale
       Attn = α * Attn(Q, K, V) + (1-α) * Attn(Q, K', V)
       # dove α è learnable
   ```

2. **KDTrainerMaster**:
   ```python
   Loss = α * CE_loss + (1-α) * KL_divergence(Student || Teacher)
   # α balancia Cross-Entropy e Knowledge Distillation
   ```

3. **Teacher Freezing**:
   ```python
   for param in teacher.parameters():
       param.requires_grad = False
   teacher.eval()  # Modalità inferenza
   ```

**Configurazione**:
- Temperature: 4.0 (smoothing delle probabilità)
- Alpha: 0.5 (bilanciamento 50/50 CE/KD)
- Batch size: 32 (ottimizzato per RTX 5090)
- FP16: Enabled (2x speedup)

---

### **FASE 3: PROGRESSIVE PRUNING** ✂️
**File**: `distillation/progressive_pruning_callback.py`

**Timeline di Spegnimento**:

| Epoca | Azione | Layers Attivi | Nota |
|-------|--------|---------------|------|
| 1 | Nessun bypass | 12/12 | Training standard |
| 2 | Bypass layer 2, 10 | 10/12 | Primi layer eliminati |
| 3 | Bypass layer 3, 9 | 8/12 | Continua pruning |
| 4 | Bypass layer 5, 7 | 6/12 | **Architettura finale** |

**Layers Finali** (6 totali):
- Layer 1 (idx 0) - Standard
- Layer 4 (idx 3) - 🔷 **Simpliciale**
- Layer 6 (idx 5) - Standard
- Layer 8 (idx 7) - 🔷 **Simpliciale**
- Layer 11 (idx 10) - Standard
- Layer 12 (idx 11) - 🔷 **Simpliciale**

**Meccanismo di Bypass**:
```python
class TransformerBlock:
    def forward(self, x):
        if self.is_bypassed:
            return x  # Shortcut completo - gradiente si ferma
        return x + self.attention(...) + self.mlp(...)
```

**Tempo training**: ~10-12 ore (4 epoche su RTX 5090)

---

### **FASE 4: ESTRAZIONE FISICA** 📦
**File**: `scripts/fase4_estrazione_fisica.py`

**Processo**:
```python
Checkpoint Training (12 layers, alcuni bypassed)
    ↓ Rimuovi fisicamente layer [1, 2, 4, 6, 8, 9]
    ↓ Rinomina layer sopravvissuti:
    │   Old Layer 1 → New Layer 1
    │   Old Layer 4 → New Layer 2 (Simpliciale)
    │   Old Layer 6 → New Layer 3
    │   Old Layer 8 → New Layer 4 (Simpliciale)
    │   Old Layer 11 → New Layer 5
    │   Old Layer 12 → New Layer 6 (Simpliciale)
    ↓
Modello Finale Compatto (6 layers fisici)
```

**Output**: `checkpoints/Davide_2Simplex_40M_Finale.pt`

**Tempo stimato**: < 1 minuto

---

## 🚀 Script di Orchestrazione

### **Master Script** - `run_master_distillation.py`
Esegue tutte le 4 fasi automaticamente:

```bash
# Pipeline completa automatica
python run_master_distillation.py --full_pipeline

# Oppure fase singola
python run_master_distillation.py --fase 1  # Solo Forgia
python run_master_distillation.py --fase 2-3  # Solo Training
python run_master_distillation.py --fase 4  # Solo Estrazione
```

### **Cloud Training** - `train_cloud_5090.py`
Ottimizzato per NVIDIA RTX 5090 32GB:

```bash
# Training completo con test pre-flight
python train_cloud_5090.py --full_pipeline --run_tests

# Skip forgia se già fatta
python train_cloud_5090.py --skip_forgia --run_tests

# Resume da checkpoint
python train_cloud_5090.py --resume_from runs/distill/checkpoint-1000
```

---

## 🧪 Pre-Flight Tests

**File**: `train_cloud_5090.py` (classe `TeacherPreFlightTests`)

### Suite di Test Automatici:

1. ✅ **Checkpoint Integrity** - Verifica file esistenti e dimensioni
2. ✅ **Teacher Loading** - Caricamento modello senza errori
3. ✅ **Vocab Compatibility** - Vocab size 1024, tokenizzazione OK
4. ✅ **Forward Pass** - Input → Output valido
5. ✅ **Logits Shape** - Shape corretta (B, T, 1024), no NaN/Inf
6. ✅ **Memory Footprint** - Teacher usa < 16GB VRAM
7. ✅ **Teacher-Student Compatibility** - Shapes logits compatibili

**Esecuzione automatica** prima del training per garantire configurazione corretta!

---

## ⚙️ Configurazione RTX 5090 Ottimizzata

### Hardware Target
- **GPU**: NVIDIA RTX 5090 32GB VRAM
- **RAM**: 64GB+ raccomandato
- **CUDA**: 12.1+
- **Driver**: 550.54.15+

### Ottimizzazioni Implementate

```python
# Batch Processing
per_device_train_batch_size = 32      # Ottimale per 32GB
gradient_accumulation_steps = 2        # Effective batch = 64

# Mixed Precision (2x Speedup)
fp16 = True
fp16_opt_level = "O2"                 # Massima ottimizzazione

# Memory Management
gradient_checkpointing = True          # -50% memoria, +20% tempo

# Data Loading
dataloader_num_workers = 4             # Parallelizza I/O
dataloader_prefetch_factor = 2         # Pre-fetch batches

# Learning Rate Schedule
lr_scheduler_type = "cosine"           # Cosine annealing
warmup_ratio = 0.05                    # 5% warmup steps
```

### Performance Attese
- **Speed**: ~2.5 ore/epoca (10K examples)
- **VRAM Peak**: ~28GB (lascia margine)
- **Throughput**: 40-50 samples/sec

---

## 📁 Struttura File Generati

```
simplex-distillery/
├── checkpoints/
│   ├── studente_inizializzato.pt          # Fase 1 (~150MB)
│   └── Davide_2Simplex_40M_Finale.pt      # Fase 4 (~150MB)
│
├── runs/distill_5090/
│   ├── checkpoint-500/                     # Checkpoint intermedi
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   ├── logs/                              # TensorBoard logs
│   │   └── events.out.tfevents.*
│   └── pytorch_model.bin                  # Modello finale training
│
├── logs/
│   └── training_5090.log                  # Log completo training
│
└── results/
    └── final_metrics.json                 # Metriche finali
```

---

## 🔧 Git LFS - Gestione File Grandi

**File**: `.gitattributes` (aggiornato)

### File tracciati automaticamente con LFS:

```gitattributes
# Model Checkpoints
*.pt, *.bin, *.pth, *.sav, *.safetensors

# Datasets
*.parquet, *.tar.gz, *.h5, *.arrow

# Vocabulary
*.model, *.vocab

# Generated Files
results_eval/** (tutti i file in questa directory)
```

### Comando Setup:
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.bin"
```

---

## 📊 Metriche Attese

### Durante Training

| Epoca | Loss Start | Loss End | Layers Attivi | Note |
|-------|------------|----------|---------------|------|
| 1 | 2.5 | 1.8 | 12/12 | Baseline |
| 2 | 1.8 | 1.4 | 10/12 | Pruning layer 2,10 |
| 3 | 1.4 | 1.2 | 8/12 | Pruning layer 3,9 |
| 4 | 1.2 | 1.0 | 6/12 | Pruning layer 5,7 → **Finale** |

### Modello Finale

```yaml
Architecture:
  Total Layers: 6 (3 standard + 3 simplicial)
  Hidden Dimension: 384
  Attention Heads: 8
  Vocab Size: 1024

Performance:
  Parameters: ~40M (vs 151M Teacher = 73% riduzione)
  File Size: ~150MB
  Accuracy: 85-90% del Teacher (atteso)
  Inference Speed: 3-4x più veloce del Teacher

Layer Configuration:
  Standard Layers: [1, 3, 5]
  Simplicial Layers: [2, 4, 6]  # Con K' perturbato
```

---

## 🚢 Deployment

### 1. Verifica Modello Finale
```bash
python scripts/fase4_estrazione_fisica.py \
    --checkpoint runs/distill_5090/pytorch_model.bin \
    --output checkpoints/finale.pt \
    --verify
```

### 2. Test Inferenza
```bash
python evaluate_teacher.py \
    --model checkpoints/Davide_2Simplex_40M_Finale.pt \
    --problem orthocenter
```

### 3. Export per Produzione
```python
# Carica modello
from models.student_progressive import StudentModelProgressive
model = StudentModelProgressive.from_pretrained("checkpoints/finale.pt")

# Inference
input_ids = tokenizer.encode("a b c = triangle a b c ;")
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=100)
```

---

## 📚 Documentazione Completa

| File | Descrizione |
|------|-------------|
| `MASTER_PLAN_IMPLEMENTATION.md` | Questo documento - Overview completa |
| `CLOUD_TRAINING_5090.md` | Guida dettagliata training cloud |
| `FIX_SUMMARY.md` | Fix tecnici tokenizer/vocab |
| `QUICK_START.md` | Quick start per distillazione base |
| `README.md` | Overview progetto |

---

## ✅ Checklist Pre-Training

Prima di avviare il training su RTX 5090:

- [ ] Git LFS configurato e file scaricati (`git lfs pull`)
- [ ] Teacher checkpoint verificato (`pt_ckpt/params.sav` presente)
- [ ] Vocab file presente (`pt_ckpt/vocab.model`)
- [ ] Dataset preparato (`data/train_sequences.txt` o `.parquet`)
- [ ] Environment setup (`./setup_cloud_5090.sh` completato)
- [ ] Pre-flight tests passati (`python train_cloud_5090.py --run_tests`)
- [ ] CUDA verficato (`nvidia-smi` mostra RTX 5090)
- [ ] TensorBoard pronto (`tensorboard --logdir runs/distill_5090/logs`)

---

## 🎯 Comandi Rapidi

### Setup Iniziale
```bash
git clone https://github.com/and-per-i/simplex-distillery.git
cd simplex-distillery
./setup_cloud_5090.sh
source venv/bin/activate
```

### Training Completo
```bash
# Automatico
./start_training.sh

# Manuale con controllo
python train_cloud_5090.py --full_pipeline --run_tests --num_epochs 4
```

### Monitoring
```bash
# Terminal 1: Training
python train_cloud_5090.py --full_pipeline

# Terminal 2: GPU
./monitor_gpu.sh

# Terminal 3: TensorBoard
tensorboard --logdir runs/distill_5090/logs
```

### Estrazione Finale
```bash
python scripts/fase4_estrazione_fisica.py \
    --checkpoint runs/distill_5090/pytorch_model.bin \
    --output checkpoints/Davide_2Simplex_40M_Finale.pt \
    --verify
```

---

## 🏆 Risultati Attesi

Al termine della pipeline completa (4 fasi):

✅ **Modello Student**: 40M parametri, 6 layers (3 simpliciali)  
✅ **Compressione**: 73% riduzione rispetto al Teacher  
✅ **Performance**: 85-90% accuracy del Teacher  
✅ **Speed**: 3-4x più veloce in inferenza  
✅ **Memoria**: 4x meno VRAM richiesta  
✅ **Architettura**: 2-Simplicial Attention per geometria  

**Innovazione**: Primo modello geometrico che unisce SVD + Identità Perturbata + Progressive Pruning!

---

## 📞 Support & Next Steps

### Issues e Bug
https://github.com/and-per-i/simplex-distillery/issues

### Prossimi Sviluppi
1. Benchmark su IMO problems
2. Fine-tuning per domini specifici
3. Quantizzazione INT8 per edge deployment
4. Multi-GPU scaling tests

---

**🎉 MASTER PLAN IMPLEMENTATO - READY FOR TRAINING! 🚀**

*Built with cutting-edge distillation techniques for geometric reasoning.*
