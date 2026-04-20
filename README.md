# 🔷 Simplex Distillery

**Simplex Distillery** is an advanced framework for **Knowledge Distillation** applied to geometric reasoning. The project aims to distill the knowledge of complex language models (such as AlphaGeometry) into more efficient architectures based on **2-Simplicial Attention**.

---

## 🚀 Overview

The system is designed to train a "Student" (a compact model with simplicial attention) by mimicking the performance of a "Teacher" (a large-scale model or pre-extracted logits).

### Key Features:
- 🧠 **2-Simplicial Attention**: A custom architecture optimized for capturing complex geometric relationships.
- ⚗️ **Offline/Online Distillation**: Support for real-time distillation via a Teacher model or via pre-extracted logits (Parquet).
- 🍎 **Apple Silicon Optimization**: Native support for MPS (Metal Performance Shaders) acceleration.
- 📏 **AlphaGeometry Tokenizer**: Native integration with the 757-token SentencePiece vocabulary.
- 🏗️ **HF Ecosystem**: Fully compatible with HuggingFace `Trainer`, `PretrainedConfig`, and `PreTrainedModel`.

---

## 📂 Project Structure

```text
.
├── data/               # Dataset management and custom data collators
├── distillation/       # Core Knowledge Distillation logic (KDTrainer, KDloss)
├── models/             # Student (2-Simplex) and Teacher Wrapper definitions
├── tokenizer/          # Integration with the AlphaGeometry tokenizer
├── train.py            # Main entry point for training
├── inference_test.py   # Script for testing model performance
└── master_setup.sh     # Setup script for cloud/local environments
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/and-per-i/simplex-distillery.git
   cd simplex-distillery
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training & Distillation

The main command to start the distillation process is `train.py`.

### Example: Online Distillation with GPT-2 as Teacher
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

### Example: Offline Distillation (from Parquet with pre-calculated logits)
```bash
python train.py \
    --data_path train0901.parquet \
    --output_dir runs/offline_distill \
    --per_device_train_batch_size 16
```

---

## 🧪 Testing & Inference

After training, you can verify the model using:
```bash
python inference_test.py --model_path runs/kd_v1 --prompt "a b c coll a b c ;"
```

---

## ⚖️ License
This project is developed for research purposes in the field of geometric intelligence.