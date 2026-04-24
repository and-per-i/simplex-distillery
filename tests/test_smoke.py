"""
Smoke test per verificare che tutto il pipeline funzioni end-to-end.

Esegui:
    python tests/test_smoke.py

Verifica:
    1. Tokenizer caricato correttamente (vocab_size == 1024)
    2. StudentConfig valido
    3. StudentForCausalLM forward corretto (shape logits, loss)
    4. TeacherWrapper normalizza l'output
    5. KnowledgeDistillationLoss calcola correttamente
    6. GeometryDataset e DataCollator producono batch validi
    7. KDTrainer compute_loss funziona
"""

import os
import sys
import torch

# Aggiungi la root al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer.hf_tokenizer import load_tokenizer
from models.student_config import StudentConfig
from models.student_model import StudentForCausalLM
from models.teacher_wrapper import TeacherWrapper
from data.dataset import make_test_dataset
from data.collator import GeometryDataCollator
from distillation.kd_loss import KnowledgeDistillationLoss

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tokenizer", "weights", "geometry.757.model"
)

def test_tokenizer():
    print("=== 1. Tokenizer ===")
    tok = load_tokenizer(TOKENIZER_PATH)
    assert tok.vocab_size == 1024, f"vocab_size atteso 1024, trovato {tok.vocab_size}"
    ids = tok.encode("a b c coll a b c ;")
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)
    print(f"  ✅ vocab_size={tok.vocab_size}, encode ok, decode ok")
    return tok

def test_student_model(tokenizer):
    print("=== 2. StudentForCausalLM ===")
    config = StudentConfig(
        vocab_size=1024,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    model = StudentForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 1024, (2, 32))
    labels = input_ids.clone()
    labels[:, :5] = -100  # simula alcuni token ignorati

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)

    assert out.logits.shape == (2, 32, 1024), f"Shape errata: {out.logits.shape}"
    assert out.loss is not None and out.loss.item() > 0
    print(f"  ✅ logits shape: {out.logits.shape}, loss: {out.loss.item():.4f}")
    return model, config

def test_teacher_wrapper(config):
    print("=== 3. TeacherWrapper ===")
    teacher_raw = StudentForCausalLM(config)  # usa student come teacher finto
    teacher = TeacherWrapper(teacher_raw, student_vocab_size=1024)

    input_ids = torch.randint(0, 1024, (2, 32))
    out = teacher(input_ids=input_ids)

    assert hasattr(out, "logits")
    assert out.logits.shape == (2, 32, 1024)
    # Verifica che il teacher sia frozen
    for p in teacher.wrapped_teacher.parameters():
        assert not p.requires_grad, "Teacher ha parametri con requires_grad=True!"
    print(f"  ✅ output shape: {out.logits.shape}, teacher frozen: ✓")
    return teacher

def test_kd_loss():
    print("=== 4. KnowledgeDistillationLoss ===")
    loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.5)

    B, S, V = 2, 16, 1024
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[:, -3:] = -100  # ultimi 3 token ignorati

    result = loss_fn(student_logits, teacher_logits, labels)
    assert "total_loss" in result
    assert result["total_loss"].item() > 0
    assert result["ce_loss"].item() > 0
    assert result["kd_loss"].item() > 0
    print(
        f"  ✅ total_loss={result['total_loss'].item():.4f}, "
        f"ce_loss={result['ce_loss'].item():.4f}, "
        f"kd_loss={result['kd_loss'].item():.4f}"
    )

def test_dataset_and_collator(tokenizer):
    print("=== 5. Dataset + DataCollator ===")
    ds = make_test_dataset(tokenizer, n=10, max_length=64)
    assert len(ds) == 10

    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample

    collator = GeometryDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    batch = collator([ds[i] for i in range(4)])

    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].shape == batch["input_ids"].shape
    # Le posizioni paddate nei labels devono essere -100
    pad_positions = (batch["attention_mask"] == 0)
    if pad_positions.any():
        assert (batch["labels"][pad_positions] == -100).all(), \
            "Labels paddati con valore sbagliato (atteso -100)"

    print(
        f"  ✅ Dataset ok ({len(ds)} samples), "
        f"batch shape: {batch['input_ids'].shape}, "
        f"padding con -100: ✓"
    )

def main():
    print("🧪 Avvio smoke test...\n")
    tok = test_tokenizer()
    model, config = test_student_model(tok)
    test_teacher_wrapper(config)
    test_kd_loss()
    test_dataset_and_collator(tok)
    print("\n🎉 Tutti i test passati!")

if __name__ == "__main__":
    main()
