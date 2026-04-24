"""
FASE 2-3: Training Master con KD + Progressive Pruning
=======================================================

Implementa il training loop completo che integra:
- Knowledge Distillation (Teacher → Student)
- Progressive Pruning Callback
- StudentModelProgressive con attenzione simpliciale
- Logging avanzato

Usage:
    from distillation.trainer_master import run_distillation_master
    
    run_distillation_master(
        teacher_path='pt_ckpt',
        student_init_path='checkpoints/studente_inizializzato.pt',
        output_dir='runs/distill',
        data_path='data/train.txt',
        num_epochs=4,
        progressive_pruning=True
    )
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from pathlib import Path
import logging
from typing import Dict, Optional

from models.student_progressive import StudentModelProgressive
from models.teacher_wrapper import TeacherWrapper
from alphageo.alphageometry import get_lm
from tokenizer.hf_tokenizer import load_tokenizer
from data.dataset import GeometryDataset
from data.collator import GeometryDataCollator
from distillation.progressive_pruning_callback import ProgressivePruningCallback
from distillation.kd_loss import KnowledgeDistillationLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KDTrainerMaster(Trainer):
    """
    Trainer custom per Knowledge Distillation con Progressive Pruning.
    
    Estende Hugging Face Trainer con:
    - KD Loss (Kullback-Leibler Divergence)
    - Progressive Pruning support
    - Logging avanzato
    """
    
    def __init__(self, teacher_model=None, temperature=4.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # KD Loss
        self.kd_loss_fn = KnowledgeDistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        # Freeze teacher
        if self.teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Calcola loss combinato: CE loss + KD loss.
        
        Formula:
            Loss = α * CE(student_logits, labels) + (1-α) * KL(student_logits || teacher_logits)
        """
        labels = inputs.pop("labels")
        
        # Forward pass student
        student_outputs = model(**inputs)
        student_logits = student_outputs['logits'] if isinstance(student_outputs, dict) else student_outputs
        
        # Forward pass teacher (se disponibile)
        if self.teacher:
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
        else:
            # Fallback: solo CE loss
            loss = nn.functional.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return (loss, student_outputs) if return_outputs else loss
        
        # Calcola KD Loss
        loss_dict = self.kd_loss_fn(student_logits, teacher_logits, labels)
        total_loss = loss_dict['total_loss']
        
        # Log metriche (ogni N step)
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'kd/total_loss': total_loss.item(),
                'kd/ce_loss': loss_dict['ce_loss'].item(),
                'kd/kd_loss': loss_dict['kd_loss'].item(),
                'kd/alpha': self.alpha,
                'kd/temperature': self.temperature
            })
        
        return (total_loss, student_outputs) if return_outputs else total_loss


def load_teacher(teacher_path: str, device: str, vocab_size: int = 1024):
    """
    Carica il Teacher Model (AlphaGeometry).
    
    Args:
        teacher_path: Path al checkpoint (directory o file .sav)
        device: Device (cpu, cuda, mps)
        vocab_size: Dimensione vocabolario
    
    Returns:
        TeacherWrapper pronto per inferenza
    """
    logger.info(f"📦 Caricamento Teacher da: {teacher_path}")
    
    teacher_path = Path(teacher_path)
    
    # Se è una directory, cerca params.sav
    if teacher_path.is_dir():
        teacher_path = teacher_path / "params.sav"
    
    # Carica modello AlphaGeometry
    teacher_model = get_lm(teacher_path.parent, device)
    
    # Wrap per compatibilità Trainer
    teacher_wrapped = TeacherWrapper(teacher_model, student_vocab_size=vocab_size)
    teacher_wrapped.to(device)
    teacher_wrapped.eval()
    
    logger.info("✅ Teacher caricato e wrappato")
    return teacher_wrapped


def load_student(student_init_path: str, device: str, vocab_size: int = 1024,
                 dim_hidden: int = 384, num_layers: int = 12,
                 simplicial_layers: list = [3, 7, 11]):
    """
    Carica lo Student Model inizializzato dalla Fase 1.
    
    Args:
        student_init_path: Path a studente_inizializzato.pt
        device: Device
        vocab_size: Dimensione vocabolario
        dim_hidden: Hidden dimension
        num_layers: Numero layer
        simplicial_layers: Indici layer simpliciali
    
    Returns:
        StudentModelProgressive pronto per training
    """
    logger.info(f"📦 Caricamento Student da: {student_init_path}")
    
    # Inizializza architettura
    student = StudentModelProgressive(
        vocab_size=vocab_size,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        num_heads=8,
        simplicial_layers=simplicial_layers,
        max_seq_len=512,
        dropout=0.1
    )
    
    # Carica pesi dalla Fase 1 (se esistono)
    if Path(student_init_path).exists():
        logger.info("  Caricamento pesi da Fase 1...")
        state_dict = torch.load(student_init_path, map_location='cpu')
        
        # Carica pesi (con gestione fallback)
        try:
            student.load_state_dict(state_dict, strict=False)
            logger.info("✅ Pesi Student caricati (strict=False)")
        except Exception as e:
            logger.warning(f"⚠️  Caricamento parziale: {e}")
            logger.info("  Inizializzazione random per pesi mancanti")
    else:
        logger.warning(f"⚠️  File non trovato: {student_init_path}")
        logger.info("  Inizializzazione random completa")
    
    student.to(device)
    student.train()
    
    # Log parametri
    params = student.count_parameters()
    logger.info(f"✅ Student inizializzato:")
    logger.info(f"  - Parametri totali: {params['total']/1e6:.2f}M")
    logger.info(f"  - Attention: {params['attention']/1e6:.2f}M")
    logger.info(f"  - MLP: {params['mlp']/1e6:.2f}M")
    logger.info(f"  - Embedding: {params['embedding']/1e6:.2f}M")
    
    return student


def run_distillation_master(
    teacher_path: str,
    student_init_path: str,
    output_dir: str,
    data_path: str,
    num_epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    progressive_pruning: bool = True,
    pruning_schedule: Optional[Dict] = None,
    **kwargs
) -> bool:
    """
    FUNZIONE PRINCIPALE: Esegue training completo con KD + Progressive Pruning.
    
    Args:
        teacher_path: Path al teacher checkpoint
        student_init_path: Path allo student inizializzato (Fase 1)
        output_dir: Directory output per checkpoints
        data_path: Path al dataset
        num_epochs: Numero epoche
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Temperature per KD softmax
        alpha: Bilanciamento CE/KD (0=solo KD, 1=solo CE)
        progressive_pruning: Abilita Progressive Pruning
        pruning_schedule: Schedule custom (None = default)
        **kwargs: Altri parametri per TrainingArguments
    
    Returns:
        True se training completato con successo
    """
    logger.info("\n" + "="*80)
    logger.info("🎯 FASE 2-3: KNOWLEDGE DISTILLATION + PROGRESSIVE PRUNING")
    logger.info("="*80 + "\n")
    
    # Determina device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"💻 Device: {device.upper()}")
    
    # 1. Carica tokenizer
    vocab_path = Path(teacher_path) / "vocab.model" if Path(teacher_path).is_dir() else "pt_ckpt/vocab.model"
    tokenizer = load_tokenizer(str(vocab_path), vocab_size=1024)
    logger.info(f"✅ Tokenizer caricato: vocab_size={tokenizer.vocab_size}")
    
    # 2. Carica Teacher
    teacher = load_teacher(teacher_path, device, vocab_size=1024)
    
    # 3. Carica Student
    student = load_student(student_init_path, device, vocab_size=1024, dim_hidden=384)
    
    # 4. Carica Dataset
    logger.info(f"📚 Caricamento dataset da: {data_path}")
    dataset = GeometryDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512
    )
    logger.info(f"✅ Dataset caricato: {len(dataset)} esempi")
    
    # Data Collator
    collator = GeometryDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",  # No eval durante training per speed
        fp16=False,  # Disabilita FP16 per MPS compatibility
        dataloader_num_workers=0,  # Evita problemi su MacOS
        remove_unused_columns=False,
        report_to="none",  # Disabilita wandb/tensorboard per semplicità
        **kwargs
    )
    
    # 6. Callbacks
    callbacks = []
    if progressive_pruning:
        pruning_callback = ProgressivePruningCallback(pruning_schedule=pruning_schedule)
        callbacks.append(pruning_callback)
        logger.info("✅ Progressive Pruning attivato")
    
    # 7. Inizializza Trainer
    logger.info("\n🚀 Inizializzazione KDTrainerMaster...")
    trainer = KDTrainerMaster(
        model=student,
        teacher_model=teacher,
        temperature=temperature,
        alpha=alpha,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks
    )
    
    # 8. Training
    logger.info("\n" + "="*80)
    logger.info("🔥 AVVIO TRAINING")
    logger.info("="*80)
    logger.info(f"  - Epoche: {num_epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info(f"  - Alpha: {alpha}")
    logger.info(f"  - Progressive Pruning: {progressive_pruning}")
    logger.info("="*80 + "\n")
    
    try:
        trainer.train()
        logger.info("\n✅ Training completato con successo!")
        
        # Salva modello finale
        final_path = Path(output_dir) / "pytorch_model.bin"
        torch.save(student.state_dict(), final_path)
        logger.info(f"💾 Modello finale salvato: {final_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Training fallito: {e}")
        import traceback
        traceback.print_exc()
        return False


# Entry point per testing standalone
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, default='pt_ckpt')
    parser.add_argument('--student_init', type=str, default='checkpoints/studente_inizializzato.pt')
    parser.add_argument('--datapath', type=str, default='data/train_sequences.txt')
    parser.add_argument('--output_dir', type=str, default='runs/distill_master')
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    success = run_distillation_master(
        teacher_path=args.teacher_path,
        student_init_path=args.student_init,
        output_dir=args.output_dir,
        data_path=args.data_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    exit(0 if success else 1)
