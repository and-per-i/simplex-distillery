# ✅ Deployment Checklist - RTX 5090 Cloud

**Quick reference per deployment su Cloud GPU**

---

## 📋 Pre-Deployment Checklist

### 1. Hardware Verification
```bash
# Verifica GPU
nvidia-smi

# Expected Output:
# GPU: NVIDIA GeForce RTX 5090
# VRAM: 32GB
# Driver Version: 550.54.15+
# CUDA Version: 12.1+
```

- [ ] GPU è RTX 5090 (o compatibile)
- [ ] VRAM ≥ 32GB
- [ ] CUDA 12.1+
- [ ] Driver NVIDIA aggiornato

### 2. System Requirements
- [ ] Ubuntu 20.04+ / Debian-based Linux
- [ ] Python 3.10+
- [ ] Git 2.30+
- [ ] Git LFS installato
- [ ] 500GB+ storage libero (per dataset + checkpoints)
- [ ] 64GB+ RAM (raccomandato)

### 3. Network
- [ ] Internet connection stabile (per git clone e download dipendenze)
- [ ] Port 6006 aperto per TensorBoard (opzionale)

---

## 🚀 Deployment Step-by-Step

### Step 1: Clone Repository
```bash
# Installa Git LFS prima del clone!
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Clone repository (with LFS files)
git clone https://github.com/and-per-i/simplex-distillery.git
cd simplex-distillery

# Verifica LFS files
git lfs ls-files
```

**Expected LFS Files:**
- ✅ `pt_ckpt/params.sav` (~580MB)
- ✅ `pt_ckpt/vocab.model` (~14KB)
- ✅ `pt_ckpt/cfg.sav` (~1KB)
- ✅ `geometry.757.vocab`
- ✅ `tokenizer/weights/geometry.757.model`

### Step 2: Run Setup Script
```bash
chmod +x setup_cloud_5090.sh
./setup_cloud_5090.sh
```

**Script Actions:**
1. ✅ Verifica NVIDIA drivers
2. ✅ Installa Python dependencies
3. ✅ Installa PyTorch con CUDA 12.1
4. ✅ Configura Git LFS
5. ✅ Crea virtual environment
6. ✅ Verifica dataset e checkpoints
7. ✅ Crea directory necessarie

### Step 3: Activate Environment
```bash
source venv/bin/activate
```

### Step 4: Pre-Flight Tests
```bash
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

- [ ] Tutti i 7 test passano
- [ ] Nessun warning critico
- [ ] Memory footprint < 16GB

### Step 5: Prepare Dataset
```bash
# Verifica dataset
ls -lh data/
```

**Required:**
- [ ] `data/train_sequences.txt` o `data/*.parquet` presente
- [ ] Dataset size > 0
- [ ] File leggibile

**Se dataset mancante:**
```bash
# Download o copia dataset
# Esempio:
# gsutil cp gs://your-bucket/train_sequences.txt data/
# aws s3 cp s3://your-bucket/train_sequences.txt data/
```

---

## 🔥 Launch Training

### Option 1: Automatic (Recommended)
```bash
./start_training.sh
```

### Option 2: Manual Control
```bash
python train_cloud_5090.py \
    --full_pipeline \
    --run_tests \
    --num_epochs 4 \
    --data_path data/train_sequences.txt \
    --output_dir runs/distill_5090
```

### Option 3: Custom Configuration
```bash
python train_cloud_5090.py \
    --full_pipeline \
    --run_tests \
    --num_epochs 6 \
    --temperature 3.5 \
    --alpha 0.6 \
    --output_dir runs/custom_run
```

---

## 📊 Monitoring

### Terminal 1: Training Progress
```bash
source venv/bin/activate
python train_cloud_5090.py --full_pipeline
```

### Terminal 2: GPU Monitoring
```bash
./monitor_gpu.sh
# Or manually:
watch -n 1 nvidia-smi
```

**What to watch:**
- GPU Utilization: should be 95-100%
- GPU Memory: should reach ~28-30GB (out of 32GB)
- Temperature: keep < 85°C
- Power: should be near TDP limit

### Terminal 3: TensorBoard
```bash
source venv/bin/activate
tensorboard --logdir runs/distill_5090/logs --port 6006 --bind_all
```

Access at: `http://<server-ip>:6006`

---

## ⏱️ Expected Timeline

| Phase | Duration | Note |
|-------|----------|------|
| **Fase 1: Forgia** | 5-10 min | SVD compression offline |
| **Fase 2-3: Training** | 10-12 hours | 4 epochs with progressive pruning |
| - Epoch 1 | 2.5 hours | 12 layers active |
| - Epoch 2 | 2.5 hours | 10 layers (pruning applied) |
| - Epoch 3 | 2.5 hours | 8 layers |
| - Epoch 4 | 2.5 hours | 6 layers (final) |
| **Fase 4: Extraction** | < 1 min | Physical compaction |
| **TOTAL** | ~12 hours | Full pipeline |

---

## 🚨 Common Issues & Solutions

### Issue 1: "CUDA out of memory"
**Symptom:** RuntimeError during training

**Solution:**
```bash
# Reduce batch size
python train_cloud_5090.py --full_pipeline --batch_size 16

# Or edit train_cloud_5090.py:
# per_device_train_batch_size = 16
```

### Issue 2: "params.sav not found"
**Symptom:** Pre-flight test fails on Checkpoint Integrity

**Solution:**
```bash
# Re-pull LFS files
git lfs pull

# Verify
ls -lh pt_ckpt/params.sav
# Should show ~580MB
```

### Issue 3: "Slow data loading"
**Symptom:** GPU utilization < 90%

**Solution:**
```bash
# Increase data workers
# Edit train_cloud_5090.py:
# dataloader_num_workers = 8  # from 4
```

### Issue 4: "Loss not decreasing"
**Symptom:** Loss stuck or increasing

**Check:**
1. Learning rate not too high/low
2. Temperature parameter (should be 3-5)
3. Alpha balance (0.3-0.7 range)
4. Dataset quality

**Solution:**
```bash
# Adjust hyperparameters
python train_cloud_5090.py \
    --full_pipeline \
    --temperature 4.5 \
    --alpha 0.4 \
    --learning_rate 3e-4
```

---

## 💾 Checkpoint Management

### During Training
Checkpoints saved every 500 steps:
```
runs/distill_5090/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── ...
```

### Backup Strategy
```bash
# Backup to cloud storage every 1000 steps
# Google Cloud:
gsutil -m rsync -r runs/distill_5090 gs://your-bucket/distill_5090/

# AWS S3:
aws s3 sync runs/distill_5090 s3://your-bucket/distill_5090/

# Or use cron for automatic backup:
# */30 * * * * cd /path/to/simplex-distillery && gsutil -m rsync -r runs/distill_5090 gs://bucket/backup/
```

---

## ✅ Post-Training Verification

### Step 1: Check Final Checkpoint
```bash
ls -lh runs/distill_5090/pytorch_model.bin
# Should exist and be ~150-200MB
```

### Step 2: Run Fase 4 (Extraction)
```bash
python scripts/fase4_estrazione_fisica.py \
    --checkpoint runs/distill_5090/pytorch_model.bin \
    --output checkpoints/Davide_2Simplex_40M_Finale.pt \
    --verify
```

**Expected Output:**
```
🎉 ESTRAZIONE COMPLETATA!
📦 File finale: checkpoints/Davide_2Simplex_40M_Finale.pt
💾 Dimensione: ~150 MB
🔢 Parametri totali: ~40M
📐 Architettura finale: 6 layers
```

### Step 3: Test Inference
```bash
python evaluate_teacher.py \
    --model checkpoints/Davide_2Simplex_40M_Finale.pt \
    --problem orthocenter
```

**Expected:** Solution found successfully

---

## 📦 Final Deliverables

After successful training, you should have:

- [ ] `checkpoints/studente_inizializzato.pt` (~150MB) - After Fase 1
- [ ] `runs/distill_5090/pytorch_model.bin` (~150MB) - After training
- [ ] `checkpoints/Davide_2Simplex_40M_Finale.pt` (~150MB) - Final model
- [ ] `runs/distill_5090/logs/` - TensorBoard logs
- [ ] `logs/training_5090.log` - Complete training log

### Recommended Backups
```bash
# Create tarball
tar -czf distill_5090_complete.tar.gz \
    checkpoints/ \
    runs/distill_5090/ \
    logs/

# Upload to cloud
# gsutil cp distill_5090_complete.tar.gz gs://your-bucket/
# aws s3 cp distill_5090_complete.tar.gz s3://your-bucket/
```

---

## 🎯 Success Criteria

Training is successful if:

- [ ] All 4 phases complete without errors
- [ ] Final loss < 1.2 (Epoch 4)
- [ ] No NaN/Inf in losses
- [ ] Final model file exists and is ~150MB
- [ ] Inference test passes
- [ ] Model has 6 layers (3 standard + 3 simplicial)
- [ ] GPU utilization was 90-100% during training
- [ ] No OOM errors occurred

---

## 📞 Troubleshooting Contacts

### If Stuck:
1. Check logs: `cat logs/training_5090.log | tail -100`
2. Check TensorBoard for loss curves
3. Verify GPU status: `nvidia-smi`
4. Check disk space: `df -h`
5. GitHub Issues: https://github.com/and-per-i/simplex-distillery/issues

### Common Exit Points:
- **Pre-flight test fails** → Check Teacher checkpoint integrity
- **OOM during training** → Reduce batch size
- **Loss exploding** → Reduce learning rate
- **Slow progress** → Check data loading (increase workers)

---

## 🚀 Quick Command Reference

```bash
# Setup
./setup_cloud_5090.sh && source venv/bin/activate

# Test
python train_cloud_5090.py --run_tests --skip_forgia

# Train
./start_training.sh

# Monitor
./monitor_gpu.sh  # Terminal 2
tensorboard --logdir runs/distill_5090/logs  # Terminal 3

# Extract
python scripts/fase4_estrazione_fisica.py \
    --checkpoint runs/distill_5090/pytorch_model.bin \
    --output checkpoints/finale.pt --verify

# Test Model
python evaluate_teacher.py --model checkpoints/finale.pt
```

---

**✅ Deployment Ready - Good Luck with Training! 🚀**
