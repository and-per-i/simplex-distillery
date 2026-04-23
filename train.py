"""
train.py — Entry point per la distillazione Knowledge Distillation.

Uso:
    python train.py                          # config default (smoke test con dati sintetici)
    python train.py --teacher gpt2           # teacher da HF Hub
    python train.py --data_path data.txt     # dataset reale
    python train.py --output_dir runs/exp1   # directory di output

Per un run completo con tutte le opzioni:
    python train.py \\
        --teacher gpt2 \\
        --data_path /path/to/geometry_sequences.txt \\
        --output_dir runs/kd_v1 \\
        --temperature 4.0 \\
        --alpha 0.5 \\
        --num_train_epochs 10 \\
        --per_device_train_batch_size 8 \\
        --learning_rate 5e-4
"""

import argparse
import os
import sys

import torch
from transformers import TrainingArguments

# Aggiungi la root del progetto al PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import GeometryDataset, make_test_dataset
from data.collator import GeometryDataCollator
from distillation.kd_trainer import KDTrainer
from models.student_config import StudentConfig
from models.student_model import StudentForCausalLM
from models.teacher_wrapper import TeacherWrapper, load_teacher_from_hf
from tokenizer.hf_tokenizer import load_tokenizer
from distillation.pruning_callback import ProgressivePruningCallback


# ---------------------------------------------------------------------------
# Parsing argomenti
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation con HuggingFace Trainer"
    )

    # Teacher
    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Path o nome HF Hub del modello teacher. None = smoke test senza teacher.",
    )

    # Dati
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path al file .txt delle sequenze geometriche (una per riga). "
             "None = usa dati sintetici per test.",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Modello student
    parser.add_argument("--hidden_size", type=int, default=384) # 384 come da piano
    parser.add_argument("--num_layers", type=int, default=12)  # 12 layer da comprimere poi a 6
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=1536) # 4 * 384
    parser.add_argument("--forged_path", type=str, default=None, help="Path al file .pt inizializzato con SVD (Fase 1)")

    # KD hyperparameters
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)

    # Training
    parser.add_argument("--output_dir", type=str, default="runs/kd_distillery")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", help="Usa fp16 se CUDA disponibile")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Dispositivo
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🖥️  Device: {device}")

    # Ottimizzazione specifica per NVIDIA (Ampere/Ada/Blackwell come la 5090)
    if device == "cuda":
        torch.set_float32_matmul_precision('high') # Attiva TF32
        print("🚀 TensorFloat-32 (TF32) attivata per moltiplicazioni matriciali")

    # ------------------------------------------------------------------ #
    # 1. Tokenizer
    # ------------------------------------------------------------------ #
    tokenizer_path = os.path.join(
        os.path.dirname(__file__), "tokenizer", "weights", "geometry.757.model"
    )
    tokenizer = load_tokenizer(tokenizer_path, vocab_size=1024)
    print(f"📝 Tokenizer caricato — vocab_size: {tokenizer.vocab_size}")

    # Sanity check: il vocab_size DEVE essere 1024 (Newclid)
    assert tokenizer.vocab_size == 1024, (
        f"Vocab size atteso: 1024, trovato: {tokenizer.vocab_size}"
    )

    # ------------------------------------------------------------------ #
    # 2. Dataset
    # ------------------------------------------------------------------ #
    if args.data_path is not None:
        # Se il file è un parquet, proviamo ad attivare la modalità distillazione offline
        is_distillation = args.data_path.endswith(".parquet")
        train_dataset = GeometryDataset(
            sequences=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            is_distillation=is_distillation,
        )
    else:
        print("⚠️  Nessun data_path fornito — uso dataset sintetico per smoke test")
        train_dataset = make_test_dataset(tokenizer, n=500, max_length=args.max_seq_len)

    collator = GeometryDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # ------------------------------------------------------------------ #
    # 3. Student model
    # ------------------------------------------------------------------ #
    student_config = StudentConfig(
        vocab_size=tokenizer.vocab_size,  # = 757
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id or 0,
        bos_token_id=tokenizer.bos_token_id or 1,
        eos_token_id=tokenizer.eos_token_id or 2,
    )
    student_model = StudentForCausalLM(student_config)

    # Caricamento pesi forgiati (Fase 1)
    if args.forged_path and os.path.exists(args.forged_path):
        print(f"🔨 Caricamento pesi inizializzati (SVD) da: {args.forged_path}")
        state_dict = torch.load(args.forged_path, map_location="cpu")
        missing, unexpected = student_model.load_state_dict(state_dict, strict=False)
        print(f"   Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    # Ottimizzazione Compilazione (Torch 2.0+)
    if device == "cuda":
        print("⚡ Compilazione modello (torch.compile) attivata per performance Blackwell/Ada...")
        try:
            student_model = torch.compile(student_model)
        except Exception as e:
            print(f"⚠️  Salto compilazione: {e}")

    n_params = sum(p.numel() for p in student_model.parameters())
    print(f"🎓 Student model: {n_params:,} parametri")

    # ------------------------------------------------------------------ #
    # 4. Teacher model
    # ------------------------------------------------------------------ #
    if args.teacher is not None:
        teacher = load_teacher_from_hf(
            args.teacher,
            student_vocab_size=tokenizer.vocab_size,
            device=device,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
        )
        print(f"👨‍🏫 Teacher caricato: {args.teacher}")
    else:
        # Smoke test: usa uno student identico come teacher (distillazione a sé stesso)
        print("⚠️  Nessun teacher specificato — uso self-distillation per smoke test")
        teacher_config = StudentConfig(vocab_size=tokenizer.vocab_size)
        teacher_raw = StudentForCausalLM(teacher_config).to(device)
        teacher = TeacherWrapper(teacher_raw, student_vocab_size=tokenizer.vocab_size)

    # ------------------------------------------------------------------ #
    # 5. TrainingArguments
    # ------------------------------------------------------------------ #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, int(args.warmup_ratio * args.num_train_epochs * 500 // args.per_device_train_batch_size)),
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="no",              # disabilitato per default (no eval set)
        save_total_limit=3,
        fp16=False,
        bf16=torch.cuda.is_available(), # Forza BF16 per 5090
        dataloader_num_workers=16,       # Sfrutta i core dell'EPYC (32 core -> 16 worker è un buon sweet spot)
        dataloader_pin_memory=True,     # Velocizza trasferimento verso GPU
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=False,
    )

    # ------------------------------------------------------------------ #
    # 6. KDTrainer
    # ------------------------------------------------------------------ #
    trainer = KDTrainer(
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        # --- args standard del Trainer ---
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,   # transformers 5.x: tokenizer → processing_class
        data_collator=collator,
        callbacks=[ProgressivePruningCallback()] # Aggiunge Fase 3
    )

    # ------------------------------------------------------------------ #
    # 7. Sanity check: un batch forward prima di iniziare
    # ------------------------------------------------------------------ #
    print("\n🔍 Sanity check — un batch forward...")
    sample_batch = collator([train_dataset[i] for i in range(2)])
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    student_model.to(device)
    with torch.no_grad():
        out = student_model(**sample_batch)
    print(f"   Student logits shape: {out.logits.shape}")  # deve essere (2, S, 757)
    print(f"   CE loss: {out.loss.item():.4f}")
    print("✅ Sanity check passato!\n")

    # ------------------------------------------------------------------ #
    # 8. Training
    # ------------------------------------------------------------------ #
    print("🚀 Avvio training...")
    trainer.train()

    # ------------------------------------------------------------------ #
    # 9. Salva il modello student
    # ------------------------------------------------------------------ #
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Modello salvato in: {args.output_dir}")


if __name__ == "__main__":
    main()
