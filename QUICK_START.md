# 🚀 Quick Start Guide - Simplex Distillery

Sistema pronto e funzionante dopo i fix di compatibilità AlphaGeometry/Newclid!

---

## ⚡ Test Rapidi

### Test Tokenizer
```bash
source venv/bin/activate
python tests/test_tokenization_roundtrip.py
```
**Expected**: ✅ Tutti i test passano

### Test Formato Prompt
```bash
python tests/test_prompt_format.py
```
**Expected**: ✅ Forward pass OK, token predictions sensati

### Test Teacher Model End-to-End
```bash
python evaluate_teacher.py --problem orthocenter
```
**Expected**: 🎉 SUCCESSO! Soluzione trovata

---

## 🏋️ Distillazione

### Opzione 1: Distillazione Locale (Consigliato per debug)
```bash
python distillation/run_distill_local.py
```

**Configurazione**:
- Teacher: AlphaGeometry da `pt_ckpt/`
- Student: 2-Simplicial Attention (piccolo, 4 layers)
- Vocab: 1024 tokens
- Device: Auto-detect (MPS/CUDA/CPU)

### Opzione 2: Distillazione Cloud (Production)
```bash
python distillation/run_distill_cloud.py
```

### Opzione 3: Training Completo con train.py
```bash
python train.py \
    --teacher pt_ckpt \
    --data_path data/train_sequences.parquet \
    --output_dir runs/kd_experiment_1 \
    --temperature 4.0 \
    --alpha 0.5 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --learning_rate 5e-4 \
    --save_steps 500 \
    --logging_steps 100
```

**Parametri chiave**:
- `--temperature`: Temperature per KD softmax (default: 4.0)
- `--alpha`: Bilanciamento CE/KD loss (default: 0.5)
  - `alpha=0.0`: Solo KD loss
  - `alpha=1.0`: Solo CE loss
  - `alpha=0.5`: Bilanciato

---

## 🔧 Configurazione Corrente

### Tokenizer
- **File**: `pt_ckpt/vocab.model`
- **Vocab size**: 1024 (757 real + 267 padding)
- **EOS token**: ID=263 (semicolon `▁;`)

### Teacher Model
- **Architettura**: Decoder-only Transformer (AlphaGeometry)
- **Checkpoint**: `pt_ckpt/params.sav`
- **Config**: `pt_ckpt/cfg.sav`
- **Vocab size**: 1024
- **Embedding dim**: 1024
- **Layers**: 12
- **Heads**: 8

### Student Model (Default)
- **Architettura**: 2-Simplicial Attention
- **Vocab size**: 1024 (allineato al teacher!)
- **Hidden size**: 256 (local) / 512 (cloud)
- **Layers**: 4 (local) / 8 (cloud)
- **Heads**: 8

---

## 📁 Struttura Dataset

### Formato Parquet (Consigliato)
Colonne richieste:
- `text`: Stringa con sequenza geometrica
- Opzionale: `teacher_logits` per distillazione offline

### Formato Testo
File `.txt` con una sequenza per riga:
```
a b c = triangle a b c ; ? perp c h a b
a b = segment a b ; c : midpoint c a b ; ? midp c a b
```

---

## 🐛 Troubleshooting

### Errore: "vocab_size mismatch"
✅ **RISOLTO!** Ora tutto usa vocab_size=1024

### Errore: "EOS token not found"
✅ **RISOLTO!** EOS token corretto = 263

### Errore: "Model does not understand Newclid format"
✅ **RISOLTO!** Formato verificato compatibile

### Performance lenta su CPU
**Soluzione**: Usa MPS (macOS M-series) o CUDA (GPU NVIDIA)
```python
device = "mps" if torch.backends.mps.is_available() else "cuda"
```

### Out of Memory
**Soluzione**: Riduci batch size
```bash
--per_device_train_batch_size 4  # invece di 8
```

---

## 📊 Monitoring Training

### TensorBoard (se abilitato)
```bash
tensorboard --logdir runs/kd_experiment_1
```

### Checkpoints
Salvati automaticamente in `--output_dir`:
- `checkpoint-{step}/`: Model checkpoint
- `trainer_state.json`: Training state
- `training_args.bin`: Hyperparameters

### Logs
- Console output: Loss, perplexity, step time
- `runs/{exp_name}/`: Full training logs

---

## 🧪 Verifica Modello Distillato

Dopo il training, testa lo student:

```python
from models.student_model import StudentForCausalLM
from tokenizer.hf_tokenizer import load_tokenizer
import torch

# Carica student checkpoint
model = StudentForCausalLM.from_pretrained("runs/kd_experiment_1")
tokenizer = load_tokenizer("pt_ckpt/vocab.model")

# Test generation
text = "a b c = triangle a b c ;"
tokens = tokenizer.encode(text, add_special_tokens=False)
input_ids = torch.tensor([tokens])

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        num_beams=4,
        eos_token_id=263  # Semicolon
    )

print(tokenizer.decode(outputs[0]))
```

---

## 📚 File Importanti

| File | Descrizione |
|------|-------------|
| `FIX_SUMMARY.md` | Dettagli tecnici di tutti i fix |
| `QUICK_START.md` | Questa guida rapida |
| `README.md` | Overview del progetto |
| `evaluate_teacher.py` | Test end-to-end del teacher |
| `train.py` | Script principale di training |
| `distillation/run_distill_local.py` | Distillazione locale semplificata |

---

## 🎯 Next Steps

1. **Test locale**: `python distillation/run_distill_local.py`
2. **Monitor**: Verifica loss decresce
3. **Evaluate**: Testa student su problemi geometrici
4. **Scale**: Passa a training completo con dataset più grande

---

**Sistema pronto al 100%!** 🚀

Per domande o problemi, vedi `FIX_SUMMARY.md` per dettagli tecnici completi.
