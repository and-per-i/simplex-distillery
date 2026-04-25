"""
Fine-Tuning Post-Distillazione con Tokenizzazione Corretta
===========================================================

Carica il modello Davide già estratto dalla Fase 4 (modello compatto
a ~6 layer attivi) e lo affina su un dataset preprocessato con formato
pulito, usando LR bassa per preservare la conoscenza strutturale acquisita.

IMPORTANTE: Eseguire DOPO la Fase 4 (Estrazione Fisica).
  - I layer bypassed dal Progressive Pruning sono già stati rimossi
  - Il fine-tuning mantiene la struttura compatta risultante
  - NON si re-abilitano layer bypassed (contraddirebbe il pruning)

Prerequisiti:
    1. Training KD completato → runs/distill_5090/pytorch_model.bin
    2. Fase 4 completata → checkpoints/studente_finale.pt
    3. Dataset preprocessato → python preprocess_dataset.py \
           --input data/parquets/<nuovo_dataset>.parquet \
           --output data/parquets/finetune_clean.parquet

Usage:
    python finetune_clean.py \
        --model_path checkpoints/studente_finale.pt \
        --data_path data/parquets/finetune_clean.parquet \
        --output_dir runs/finetune_clean \
        --num_epochs 2

"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from pathlib import Path
import logging
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from models.student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer
from data.dataset import GeometryDataset
from data.collator import GeometryDataCollator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune_clean.log')
    ]
)
logger = logging.getLogger(__name__)


def load_extracted_model(model_path: str, device: str) -> StudentModelProgressive:
    """
    Carica il modello estratto dalla Fase 4 (modello compatto).
    
    NON re-abilita i layer bypassed: il Progressive Pruning ha già
    identificato quali layer sono ridondanti. Il fine-tuning lavora
    sul modello nella sua forma compatta finale.
    """
    model = StudentModelProgressive(
        vocab_size=1024,
        dim_hidden=384,
        num_layers=12,
        simplicial_layers=[3, 7, 11]
    )

    model_path = Path(model_path)

    if model_path.is_dir():
        bin_path = model_path / "pytorch_model.bin"
        safetensors_path = model_path / "model.safetensors"
        if bin_path.exists():
            model_path = bin_path
        elif safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Pesi caricati da safetensors: {safetensors_path}")
            model.to(device)
            return model

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    logger.info(f"✅ Pesi caricati da: {model_path}")

    # Mostra stato dei layer (non modificarli: rispettiamo il pruning)
    bypassed = [i+1 for i, l in enumerate(model.layers) if l.is_bypassed]
    active = [i+1 for i, l in enumerate(model.layers) if not l.is_bypassed]
    logger.info(f"   Layer attivi:   {active}")
    logger.info(f"   Layer bypassed: {bypassed} (mantenuti spenti)")

    model.to(device)
    return model


def get_finetune_args(output_dir: str, num_epochs: int = 2) -> TrainingArguments:
    """
    TrainingArguments conservativi per fine-tuning:
    - LR molto bassa (10x rispetto a distillazione) per preservare struttura
    - Batch più piccolo per maggiore granularità degli aggiornamenti
    - No KD loss (solo CE su dati puliti)
    """
    return TrainingArguments(
        output_dir=output_dir,

        # Epoche e batch
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,      # Più piccolo per aggiornamenti fini
        gradient_accumulation_steps=4,        # Batch virtuale = 64

        # LR molto bassa per preservare conoscenza strutturale
        learning_rate=5e-5,                   # 10x più bassa della distillazione
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=0.5,                    # Clipping più aggressivo (fine-tuning)

        # Warmup breve
        warmup_steps=100,
        lr_scheduler_type="cosine",

        # FP16 (come distillazione)
        fp16=True,
        gradient_checkpointing=False,

        # Logging e salvataggio
        logging_steps=100,
        logging_first_step=True,
        save_steps=1000,
        save_total_limit=3,

        # Eval
        eval_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,

        # Report
        report_to="none",
    )


class FineTuneTrainer(Trainer):
    """
    Trainer semplice per fine-tuning: solo CE loss.
    Non c'è il teacher → nessuna KD loss.
    Il modello apprende direttamente dai token puliti.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs

        # CE loss con shift (causal LM: predici t+1 da t)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='runs/distill_5090/pytorch_model.bin',
                        help='Path al modello distillato (.bin o directory checkpoint)')
    parser.add_argument('--data_path', type=str,
                        default='data/parquets/finetune_clean.parquet',
                        help='Path al dataset preprocessato (output di preprocess_dataset.py)')
    parser.add_argument('--output_dir', type=str,
                        default='runs/finetune_clean',
                        help='Directory output fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--tokenizer_path', type=str, default='pt_ckpt/vocab.model')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n🎯 FINE-TUNING POST-DISTILLAZIONE")
    logger.info(f"   Device: {device}")
    logger.info(f"   Model:  {args.model_path}")
    logger.info(f"   Data:   {args.data_path}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Epochs: {args.num_epochs}")

    # Tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path, vocab_size=1024)
    logger.info(f"✅ Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Modello distillato
    logger.info(f"\n📦 Caricamento modello estratto dalla Fase 4...")
    model = load_extracted_model(args.model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Modello: {total_params/1e6:.2f}M parametri")

    # Dataset preprocessato
    logger.info(f"\n📚 Caricamento dataset pulito: {args.data_path}")
    dataset = GeometryDataset(
        sequences=args.data_path,
        tokenizer=tokenizer,
        max_length=512,
        add_bos=True,
        add_eos=True,
    )
    logger.info(f"✅ Dataset: {len(dataset)} campioni")

    collator = GeometryDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    # Training args
    training_args = get_finetune_args(args.output_dir, args.num_epochs)

    logger.info(f"\n⚙️  Fine-tuning config:")
    logger.info(f"   LR: {training_args.learning_rate} (vs 5e-4 distillazione)")
    logger.info(f"   Batch: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"   Obiettivo: adattare embedding a token puliti preservando struttura")

    # Trainer
    trainer = FineTuneTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info(f"\n🔥 AVVIO FINE-TUNING...")
    trainer.train()

    # Salva modello finale
    final_path = Path(args.output_dir) / "pytorch_model_finetuned.bin"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n✅ Fine-tuning completato!")
    logger.info(f"   Modello salvato: {final_path}")
    logger.info(f"\n💡 Prossimo step: python verify_learning.py")
    logger.info(f"   (modifica model_path per puntare a {final_path})")


if __name__ == '__main__':
    main()
