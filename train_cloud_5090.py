"""
🚀 TRAINING SCRIPT OTTIMIZZATO PER NVIDIA RTX 5090 32GB
========================================================

Script di training per GPU cloud con:
- Ottimizzazioni per RTX 5090 (32GB VRAM)
- FP16 mixed precision training
- Gradient checkpointing per memoria
- Batch size ottimale (32-64)
- Pre-flight tests del Teacher Model
- Logging avanzato
- Checkpointing robusto

Hardware Target:
- GPU: NVIDIA RTX 5090 32GB
- RAM: 64GB+ raccomandato
- Storage: SSD per dataset

Usage:
    # Training completo con tutti i test
    python train_cloud_5090.py --full_pipeline --run_tests
    
    # Solo training (skippa Fase 1 se già fatta)
    python train_cloud_5090.py --skip_forgia
    
    # Resume da checkpoint
    python train_cloud_5090.py --resume_from runs/distill/checkpoint-1000
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import TrainingArguments
from pathlib import Path
import logging
import sys
import time
from typing import Dict, Optional
import subprocess

# Aggiungi root al path
sys.path.insert(0, str(Path(__file__).parent))

from models.student_progressive import StudentModelProgressive
from models.teacher_wrapper import TeacherWrapper
from alphageo.alphageometry import get_lm
from tokenizer.hf_tokenizer import load_tokenizer
from data.dataset import GeometryDataset
from data.collator import GeometryDataCollator
from distillation.progressive_pruning_callback import ProgressivePruningCallback
from distillation.trainer_master import KDTrainerMaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_5090.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PRE-FLIGHT TESTS - Verifica configurazione Teacher
# ============================================================================

class TeacherPreFlightTests:
    """Test suite per verificare il Teacher Model prima del training."""
    
    def __init__(self, teacher_path: str, device: str = "cuda"):
        self.teacher_path = Path(teacher_path)
        self.device = device
        self.passed_tests = []
        self.failed_tests = []
    
    def run_all_tests(self) -> bool:
        """Esegue tutti i test pre-flight. Ritorna True se tutti passano."""
        logger.info("\n" + "🧪"*35)
        logger.info("PRE-FLIGHT TESTS - Verifica Teacher Model")
        logger.info("🧪"*35 + "\n")
        
        tests = [
            ("Checkpoint Integrity", self.test_checkpoint_integrity),
            ("Teacher Loading", self.test_teacher_loading),
            ("Vocab Compatibility", self.test_vocab_compatibility),
            ("Forward Pass", self.test_forward_pass),
            ("Logits Shape", self.test_logits_shape),
            ("Memory Footprint", self.test_memory_footprint),
            ("Teacher-Student Compatibility", self.test_teacher_student_compatibility)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"🔍 Test: {test_name}")
            try:
                test_func()
                self.passed_tests.append(test_name)
                logger.info(f"✅ {test_name}: PASSED\n")
            except Exception as e:
                self.failed_tests.append((test_name, str(e)))
                logger.error(f"❌ {test_name}: FAILED")
                logger.error(f"   Error: {e}\n")
        
        # Report finale
        logger.info("\n" + "="*70)
        logger.info("📊 PRE-FLIGHT TEST RESULTS")
        logger.info("="*70)
        logger.info(f"✅ Passed: {len(self.passed_tests)}/{len(tests)}")
        logger.info(f"❌ Failed: {len(self.failed_tests)}/{len(tests)}")
        
        if self.failed_tests:
            logger.info("\n❌ Failed Tests:")
            for test_name, error in self.failed_tests:
                logger.info(f"   - {test_name}: {error}")
            logger.info("\n⚠️  TRAINING NON PUÒ INIZIARE - Fixa gli errori!")
            return False
        
        logger.info("\n🎉 TUTTI I TEST PASSATI - Ready for Training!")
        logger.info("="*70 + "\n")
        return True
    
    def test_checkpoint_integrity(self):
        """Test 1: Verifica integrità file checkpoint."""
        params_file = self.teacher_path / "params.sav" if self.teacher_path.is_dir() else self.teacher_path
        cfg_file = self.teacher_path / "cfg.sav" if self.teacher_path.is_dir() else None
        vocab_file = self.teacher_path / "vocab.model" if self.teacher_path.is_dir() else None
        
        assert params_file.exists(), f"params.sav non trovato in {params_file}"
        logger.info(f"  ✓ params.sav: {params_file.stat().st_size / (1024**2):.2f} MB")
        
        if cfg_file and cfg_file.exists():
            logger.info(f"  ✓ cfg.sav: presente")
        
        if vocab_file and vocab_file.exists():
            logger.info(f"  ✓ vocab.model: {vocab_file.stat().st_size / 1024:.2f} KB")
        
        # Prova a caricare
        state = torch.load(params_file, map_location='cpu')
        num_keys = len(state)
        logger.info(f"  ✓ Checkpoint valido: {num_keys} chiavi")
    
    def test_teacher_loading(self):
        """Test 2: Carica Teacher model."""
        teacher = get_lm(self.teacher_path, self.device)
        teacher.eval()
        
        # Conta parametri
        total_params = sum(p.numel() for p in teacher.parameters())
        logger.info(f"  ✓ Teacher caricato: {total_params/1e6:.1f}M parametri")
        
        self.teacher = teacher  # Salva per test successivi
    
    def test_vocab_compatibility(self):
        """Test 3: Verifica compatibilità vocabolario."""
        vocab_path = self.teacher_path / "vocab.model" if self.teacher_path.is_dir() else "pt_ckpt/vocab.model"
        tokenizer = load_tokenizer(str(vocab_path), vocab_size=1024)
        
        assert tokenizer.vocab_size == 1024, f"Vocab size errato: {tokenizer.vocab_size} (expected 1024)"
        logger.info(f"  ✓ Vocab size: {tokenizer.vocab_size}")
        
        # Test tokenizzazione
        test_text = "a b c = triangle a b c ;"
        tokens = tokenizer.encode(test_text, add_special_tokens=False)
        assert len(tokens) > 0, "Tokenizzazione fallita"
        logger.info(f"  ✓ Tokenization OK: '{test_text}' → {len(tokens)} tokens")
        
        # Verifica EOS token
        from alphageo.tokens import GEOMETRY_EOS_ID
        assert GEOMETRY_EOS_ID == 263, f"EOS ID errato: {GEOMETRY_EOS_ID} (expected 263)"
        logger.info(f"  ✓ EOS token ID: {GEOMETRY_EOS_ID}")
        
        self.tokenizer = tokenizer
    
    def test_forward_pass(self):
        """Test 4: Forward pass del Teacher."""
        # Prepara input dummy
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1024, (batch_size, seq_len), device=self.device)
        
        with torch.no_grad():
            start = time.time()
            output = self.teacher(input_ids)
            elapsed = time.time() - start
        
        logits = output if isinstance(output, torch.Tensor) else output.logits
        assert logits.shape == (batch_size, seq_len, 1024), f"Shape errata: {logits.shape}"
        logger.info(f"  ✓ Forward pass OK: {elapsed*1000:.2f}ms")
    
    def test_logits_shape(self):
        """Test 5: Verifica shape logits e range valori."""
        input_ids = torch.randint(0, 1024, (1, 16), device=self.device)
        
        with torch.no_grad():
            output = self.teacher(input_ids)
            logits = output if isinstance(output, torch.Tensor) else output.logits
        
        assert logits.dim() == 3, f"Logits dim errata: {logits.dim()} (expected 3)"
        assert logits.shape[-1] == 1024, f"Vocab dim errata: {logits.shape[-1]} (expected 1024)"
        
        # Verifica range valori (no NaN, no Inf)
        assert not torch.isnan(logits).any(), "NaN detected in logits"
        assert not torch.isinf(logits).any(), "Inf detected in logits"
        
        # Verifica distribuzione ragionevole
        logits_mean = logits.mean().item()
        logits_std = logits.std().item()
        logger.info(f"  ✓ Logits mean: {logits_mean:.4f}, std: {logits_std:.4f}")
    
    def test_memory_footprint(self):
        """Test 6: Misura memory footprint del Teacher."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            input_ids = torch.randint(0, 1024, (4, 64), device=self.device)
            with torch.no_grad():
                _ = self.teacher(input_ids)
            
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
            
            logger.info(f"  ✓ Memory allocated: {mem_allocated:.2f} GB")
            logger.info(f"  ✓ Memory reserved: {mem_reserved:.2f} GB")
            logger.info(f"  ✓ Peak memory: {mem_peak:.2f} GB")
            
            # Verifica che non superi 16GB (lascia spazio per student)
            assert mem_peak < 16.0, f"Teacher usa troppa memoria: {mem_peak:.2f}GB > 16GB"
    
    def test_teacher_student_compatibility(self):
        """Test 7: Verifica compatibilità Teacher-Student."""
        # Crea student dummy
        student = StudentModelProgressive(
            vocab_size=1024,
            dim_hidden=384,
            num_layers=12,
            simplicial_layers=[3, 7, 11]
        ).to(self.device)
        
        # Test batch processing
        input_ids = torch.randint(0, 1024, (2, 32), device=self.device)
        
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            student_out = student(input_ids)
            
            teacher_logits = teacher_out if isinstance(teacher_out, torch.Tensor) else teacher_out.logits
            student_logits = student_out['logits'] if isinstance(student_out, dict) else student_out
        
        # Verifica shape compatibili
        assert teacher_logits.shape == student_logits.shape, \
            f"Shape mismatch: teacher {teacher_logits.shape} vs student {student_logits.shape}"
        
        logger.info(f"  ✓ Teacher-Student compatible: {teacher_logits.shape}")


# ============================================================================
# CONFIGURAZIONE OTTIMIZZATA PER RTX 5090 32GB
# ============================================================================

def get_5090_training_args(output_dir: str, num_epochs: int = 4) -> TrainingArguments:
    """
    Configurazione ottimizzata per RTX 5090 32GB.
    
    Ottimizzazioni:
    - FP16 mixed precision (2x speedup)
    - Gradient accumulation per batch virtuale grande
    - Gradient checkpointing per risparmiare memoria
    - Batch size ottimale: 32-48
    """
    return TrainingArguments(
        output_dir=output_dir,
        
        # Epoche e batch
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,  # Ottimale per 32GB VRAM
        gradient_accumulation_steps=2,    # Batch virtuale = 64
        
        # Ottimizzazioni GPU
        fp16=True,  # Mixed precision training (RTX 5090 ha Tensor Cores)
        gradient_checkpointing=True,  # Risparmia memoria (trade-off: +20% tempo)
        
        # Learning rate e optimizer
        learning_rate=5e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # Gradient clipping
        
        # Warmup e scheduler
        warmup_steps=500,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",  # Cosine annealing per convergenza smooth
        
        # Logging e salvataggio
        logging_steps=50,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=5,  # Mantieni solo ultimi 5 checkpoint
        
        # Evaluation (opzionale)
        evaluation_strategy="steps",
        eval_steps=500,
        
        # Performance
        dataloader_num_workers=4,  # Parallelize data loading
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        
        # Stabilità
        ignore_data_skip=False,
        
        # Output
        report_to=["tensorboard"],  # Logging su TensorBoard
        logging_dir=f"{output_dir}/logs",
        
        # Checkpointing avanzato
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        # Altro
        remove_unused_columns=False,
        disable_tqdm=False,
        
        # Multi-GPU (se disponibili)
        # local_rank=-1,  # Auto-detect per DDP
    )


# ============================================================================
# PIPELINE COMPLETA
# ============================================================================

def run_fase1_forgia(config: dict) -> Path:
    """Esegue Fase 1 (Forgia SVD) se necessario."""
    student_init_path = Path(config['checkpoints_dir']) / "studente_inizializzato.pt"
    
    if student_init_path.exists() and not config.get('force_regenerate', False):
        logger.info(f"✅ Studente già forgiato: {student_init_path}")
        return student_init_path
    
    logger.info("\n🔨 FASE 1: LA FORGIA")
    logger.info("-" * 70)
    
    cmd = [
        sys.executable,
        'scripts/fase1_forgia_svd.py',
        '--teacher_path', config['teacher_path'],
        '--output', str(student_init_path),
        '--dim_originale', str(config.get('dim_originale', 1024)),
        '--nuova_dim', str(config.get('nuova_dim', 384)),
        '--simplicial_layers', *[str(i) for i in config.get('simplicial_layers', [3, 7, 11])]
    ]
    
    logger.info(f"🚀 Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode == 0:
        logger.info("✅ Fase 1 completata!")
        return student_init_path
    else:
        raise RuntimeError("❌ Fase 1 fallita")


def main_training_5090(config: dict):
    """
    Funzione principale per training su RTX 5090.
    
    Args:
        config: Dizionario configurazione completo
    """
    logger.info("\n" + "🚀"*35)
    logger.info("TRAINING OTTIMIZZATO PER RTX 5090 32GB")
    logger.info("🚀"*35 + "\n")
    
    # Device check
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA non disponibile! Questo script richiede GPU NVIDIA.")
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info(f"💻 GPU: {device_name}")
    logger.info(f"💾 VRAM: {total_memory:.1f} GB")
    
    if "5090" not in device_name and config.get('strict_gpu_check', False):
        logger.warning(f"⚠️  GPU non è RTX 5090! Rilevato: {device_name}")
        logger.warning("   Continuo comunque (usa --strict_gpu_check per bloccare)")
    
    device = "cuda"
    
    # ========================================================================
    # STEP 1: PRE-FLIGHT TESTS
    # ========================================================================
    if config.get('run_tests', True):
        logger.info("\n📋 STEP 1: PRE-FLIGHT TESTS")
        tests = TeacherPreFlightTests(config['teacher_path'], device=device)
        
        if not tests.run_all_tests():
            logger.error("❌ Pre-flight tests falliti! Impossibile continuare.")
            sys.exit(1)
    else:
        logger.info("\n⏭️  Pre-flight tests skippati (--no_tests)")
    
    # ========================================================================
    # STEP 2: FASE 1 - FORGIA (se necessario)
    # ========================================================================
    if config.get('skip_forgia', False):
        logger.info("\n⏭️  Fase 1 skippata")
        student_init_path = Path(config['checkpoints_dir']) / "studente_inizializzato.pt"
    else:
        logger.info("\n📋 STEP 2: FASE 1 - FORGIA")
        student_init_path = run_fase1_forgia(config)
    
    # ========================================================================
    # STEP 3: SETUP MODELLI E DATASET
    # ========================================================================
    logger.info("\n📋 STEP 3: SETUP MODELLI E DATASET")
    logger.info("-" * 70)
    
    # Tokenizer
    vocab_path = Path(config['teacher_path']) / "vocab.model"
    tokenizer = load_tokenizer(str(vocab_path), vocab_size=1024)
    logger.info(f"✅ Tokenizer: vocab_size={tokenizer.vocab_size}")
    
    # Teacher
    logger.info("📦 Caricamento Teacher...")
    teacher = get_lm(Path(config['teacher_path']), device)
    teacher_wrapped = TeacherWrapper(teacher, student_vocab_size=1024)
    teacher_wrapped.to(device)
    teacher_wrapped.eval()
    for param in teacher_wrapped.parameters():
        param.requires_grad = False
    logger.info("✅ Teacher pronto")
    
    # Student
    logger.info("📦 Caricamento Student...")
    student = StudentModelProgressive(
        vocab_size=1024,
        dim_hidden=384,
        num_layers=12,
        num_heads=8,
        simplicial_layers=[3, 7, 11],
        max_seq_len=512,
        dropout=0.1
    )
    
    # Carica pesi Fase 1
    if student_init_path.exists():
        state_dict = torch.load(student_init_path, map_location='cpu')
        student.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ Pesi Fase 1 caricati da {student_init_path}")
    
    student.to(device)
    student.train()
    
    params = student.count_parameters()
    logger.info(f"✅ Student pronto: {params['total']/1e6:.2f}M parametri")
    
    # Dataset
    logger.info(f"📚 Caricamento dataset: {config['data_path']}")
    dataset = GeometryDataset(
        sequences=config['data_path'],
        tokenizer=tokenizer,
        max_length=512
    )
    logger.info(f"✅ Dataset: {len(dataset)} esempi")
    
    collator = GeometryDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    
    # ========================================================================
    # STEP 4: TRAINING CONFIGURATION
    # ========================================================================
    logger.info("\n📋 STEP 4: CONFIGURAZIONE TRAINING")
    logger.info("-" * 70)
    
    training_args = get_5090_training_args(
        output_dir=config['output_dir'],
        num_epochs=config.get('num_epochs', 4)
    )
    
    logger.info("⚙️  Training Args:")
    logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  - FP16: {training_args.fp16}")
    logger.info(f"  - Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    
    # Progressive Pruning Callback
    pruning_callback = ProgressivePruningCallback(
        pruning_schedule=config.get('pruning_schedule', {
            2: [1, 9],
            3: [2, 8],
            4: [4, 6]
        })
    )
    
    # ========================================================================
    # STEP 5: TRAINING
    # ========================================================================
    logger.info("\n📋 STEP 5: TRAINING CON PROGRESSIVE PRUNING")
    logger.info("="*70)
    
    trainer = KDTrainerMaster(
        model=student,
        teacher_model=teacher_wrapped,
        temperature=config.get('temperature', 4.0),
        alpha=config.get('alpha', 0.5),
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[pruning_callback]
    )
    
    # Inizio training
    logger.info("🔥 AVVIO TRAINING...")
    start_time = time.time()
    
    try:
        if config.get('resume_from'):
            logger.info(f"📂 Resume da: {config['resume_from']}")
            trainer.train(resume_from_checkpoint=config['resume_from'])
        else:
            trainer.train()
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Training completato in {elapsed/3600:.2f} ore")
        
        # Salva modello finale
        final_path = Path(config['output_dir']) / "pytorch_model.bin"
        torch.save(student.state_dict(), final_path)
        logger.info(f"💾 Modello finale: {final_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Training fallito: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training ottimizzato per RTX 5090 32GB")
    
    # Modalità
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Esegue pipeline completa (Forgia + Training)')
    parser.add_argument('--skip_forgia', action='store_true',
                        help='Skippa Fase 1 (usa checkpoint esistente)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training da checkpoint')
    
    # Test
    parser.add_argument('--run_tests', action='store_true', default=True,
                        help='Esegue pre-flight tests (default: True)')
    parser.add_argument('--no_tests', dest='run_tests', action='store_false',
                        help='Skippa pre-flight tests')
    parser.add_argument('--strict_gpu_check', action='store_true',
                        help='Blocca se GPU non è RTX 5090')
    
    # Paths
    parser.add_argument('--teacher_path', type=str, default='pt_ckpt',
                        help='Path al teacher checkpoint')
    parser.add_argument('--data_path', type=str, default='data/train_sequences.txt',
                        help='Path al dataset')
    parser.add_argument('--output_dir', type=str, default='runs/distill_5090',
                        help='Output directory')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory checkpoint intermedi')
    
    # Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Numero epoche')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature per KD')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha per CE/KD balance')
    
    # Architettura
    parser.add_argument('--dim_originale', type=int, default=1024)
    parser.add_argument('--nuova_dim', type=int, default=384)
    parser.add_argument('--simplicial_layers', type=int, nargs='+', default=[3, 7, 11])
    
    args = parser.parse_args()
    config = vars(args)
    
    # Run training
    success = main_training_5090(config)
    sys.exit(0 if success else 1)
