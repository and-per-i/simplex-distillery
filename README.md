# 🔷 Simplex Distillery

**Simplex Distillery** è un framework avanzato per la **Knowledge Distillation** applicata al ragionamento geometrico. Il progetto mira a distillare la conoscenza di modelli di linguaggio complessi (come AlphaGeometry) in architetture più efficienti basate sulla **2-Simplicial Attention**.

---

## 🚀 Panoramica

Il sistema è progettato per addestrare uno "Student" (modello compatto con attenzione simpliciale) imitando le performance di un "Teacher" (modello di grandi dimensioni o logit pre-estratti). 

### Caratteristiche Principali:
- 🧠 **2-Simplicial Attention**: Architettura custom ottimizzata per catturare relazioni geometriche complesse.
- ⚗️ **Distillazione Offline/Online**: Supporto per distillazione real-time tramite Teacher model o tramite logit estratti (Parquet).
- 🍎 **Ottimizzazione Apple Silicon**: Supporto nativo per accelerazione MPS (Metal Performance Shaders).
- 📏 **Tokenizer AlphaGeometry**: Integrazione nativa con il vocabolario SentencePiece a 757 token.
- 🏗️ **HF Ecosystem**: Pienamente compatibile con HuggingFace `Trainer`, `PretrainedConfig` e `PreTrainedModel`.

---

## 📂 Struttura del Progetto

```text
.
├── data/               # Gestione dataset e data collator custom
├── distillation/       # Core logic della Knowledge Distillation (KDTrainer, KDloss)
├── models/             # Definizioni dello Student (2-Simplex) e Teacher Wrapper
├── tokenizer/          # Integrazione con il tokenizer di AlphaGeometry
├── train.py            # Entry point principale per l'addestramento
├── inference_test.py   # Script per testare le performance del modello
└── master_setup.sh     # Script di setup per ambienti cloud/locali
```

---

## 🛠️ Installazione

1. **Clona il repository**:
   ```bash
   git clone https://github.com/and-per-i/simplex-distillery.git
   cd simplex-distillery
   ```

2. **Crea l'ambiente virtuale**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Addestramento & Distillazione

Il comando principale per avviare la distillazione è `train.py`.

### Esempio: Distillazione Online con GPT-2 come Teacher
```bash
python train.py \
    --teacher gpt2 \
    --data_path data/geometry_sequences.txt \
    --output_dir runs/kd_v1 \
    --temperature 4.0 \
    --alpha 0.5 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8
```

### Esempio: Distillazione Offline (da Parquet con logit pre-calcolati)
```bash
python train.py \
    --data_path train0901.parquet \
    --output_dir runs/offline_distill \
    --per_device_train_batch_size 16
```

---

## 🧪 Testing & Inferenza

Dopo l'addestramento, puoi verificare il modello con:
```bash
python inference_test.py --model_path runs/kd_v1 --prompt "a b c coll a b c ;"
```

---

## ⚖️ Licenza
Questo progetto è sviluppato per scopi di ricerca nel campo dell'intelligenza geometrica. Consultare il file `LICENSE` per ulteriori dettagli.
