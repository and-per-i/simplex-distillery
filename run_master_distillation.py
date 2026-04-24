"""
🎯 MASTER SCRIPT - Pipeline Completa di Distillazione Avanzata
================================================================

Orchestra le 4 fasi del Master Plan:
1. FORGIA: SVD + Identità Perturbata → studente_inizializzato.pt
2. SETUP: Knowledge Distillation con Teacher-Student
3. PROGRESSIVE PRUNING: Riduzione graduale 12→6 layer durante training
4. ESTRAZIONE FISICA: Compattazione finale del modello

Tecniche implementate:
- SVD Compression
- Perturbed Identity (Identità Perturbata)
- 2-Simplicial Attention
- Progressive Pruning
- Knowledge Distillation

Usage:
    # Esecuzione completa end-to-end
    python run_master_distillation.py --full_pipeline
    
    # Oppure singole fasi
    python run_master_distillation.py --fase 1  # Solo Forgia
    python run_master_distillation.py --fase 2-3  # Solo Training
    python run_master_distillation.py --fase 4  # Solo Estrazione
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import torch
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MasterPipeline:
    """Orchestratore della pipeline completa di distillazione."""
    
    def __init__(self, config: dict):
        self.config = config
        self.checkpoints_dir = Path(config.get('checkpoints_dir', 'checkpoints'))
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths dei file intermedi
        self.studente_init_path = self.checkpoints_dir / "studente_inizializzato.pt"
        self.training_output_dir = Path(config.get('training_output_dir', 'runs/distillation_master'))
        self.finale_path = self.checkpoints_dir / "Davide_2Simplex_40M_Finale.pt"
        
        self.fase_completate = set()
    
    def print_banner(self, fase: str, descrizione: str):
        """Stampa banner colorato per ogni fase."""
        logger.info("\n" + "="*80)
        logger.info(f"🎯 {fase}: {descrizione}")
        logger.info("="*80 + "\n")
    
    def fase_1_forgia(self):
        """
        FASE 1: LA FORGIA
        Genera studente_inizializzato.pt tramite SVD + Identità Perturbata
        """
        self.print_banner("FASE 1", "LA FORGIA (SVD + Identità Perturbata)")
        
        # Verifica se già esiste
        if self.studente_init_path.exists() and not self.config.get('force_regenerate', False):
            logger.info(f"✅ Studente già inizializzato: {self.studente_init_path}")
            logger.info("   Usa --force_regenerate per ricrearlo")
            self.fase_completate.add(1)
            return True
        
        # Parametri Fase 1
        teacher_path = self.config.get('teacher_path', 'pt_ckpt/params.sav')
        dim_originale = self.config.get('dim_originale', 1024)
        nuova_dim = self.config.get('nuova_dim', 384)
        simplicial_layers = self.config.get('simplicial_layers', [3, 7, 11])
        
        # Costruisci comando
        cmd = [
            sys.executable,
            'scripts/fase1_forgia_svd.py',
            '--teacher_path', str(teacher_path),
            '--output', str(self.studente_init_path),
            '--dim_originale', str(dim_originale),
            '--nuova_dim', str(nuova_dim),
            '--simplicial_layers', *[str(i) for i in simplicial_layers]
        ]
        
        logger.info(f"🚀 Esecuzione: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info("✅ FASE 1 completata con successo!")
            self.fase_completate.add(1)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FASE 1 fallita: {e}")
            return False
    
    def fase_2_3_training(self):
        """
        FASE 2-3: SETUP DISTILLAZIONE + PROGRESSIVE PRUNING
        Training con Knowledge Distillation e pruning graduale
        """
        self.print_banner("FASE 2-3", "KNOWLEDGE DISTILLATION + PROGRESSIVE PRUNING")
        
        # Verifica prerequisiti
        if not self.studente_init_path.exists():
            logger.error("❌ FASE 1 non completata! Esegui prima la Forgia.")
            return False
        
        # Importa trainer custom
        try:
            from distillation.trainer_master import run_distillation_master
            
            # Parametri training
            training_config = {
                'teacher_path': self.config.get('teacher_path', 'pt_ckpt'),
                'student_init_path': str(self.studente_init_path),
                'output_dir': str(self.training_output_dir),
                'data_path': self.config.get('data_path', 'data/train_sequences.txt'),
                'num_epochs': self.config.get('num_epochs', 4),
                'batch_size': self.config.get('batch_size', 8),
                'learning_rate': self.config.get('learning_rate', 5e-4),
                'temperature': self.config.get('temperature', 4.0),
                'alpha': self.config.get('alpha', 0.5),
                'progressive_pruning': True,
                'pruning_schedule': {
                    2: [1, 9],   # Epoca 2: layer 2, 10
                    3: [2, 8],   # Epoca 3: layer 3, 9
                    4: [4, 6]    # Epoca 4: layer 5, 7
                }
            }
            
            logger.info("🚀 Avvio training con Progressive Pruning...")
            success = run_distillation_master(**training_config)
            
            if success:
                logger.info("✅ FASE 2-3 completata con successo!")
                self.fase_completate.add(2)
                self.fase_completate.add(3)
                return True
            else:
                logger.error("❌ FASE 2-3 fallita durante il training")
                return False
                
        except ImportError as e:
            logger.error(f"❌ Impossibile importare trainer: {e}")
            logger.info("💡 Crea distillation/trainer_master.py oppure usa train.py manualmente")
            return False
    
    def fase_4_estrazione(self):
        """
        FASE 4: ESTRAZIONE FISICA
        Compatta il modello finale rimuovendo layer bypassed
        """
        self.print_banner("FASE 4", "ESTRAZIONE FISICA DEL MODELLO FINALE")
        
        # Trova ultimo checkpoint del training
        checkpoint_dir = self.training_output_dir
        if not checkpoint_dir.exists():
            logger.error("❌ Directory training non trovata! Esegui prima FASE 2-3.")
            return False
        
        # Cerca pytorch_model.bin o ultimo checkpoint
        checkpoints = list(checkpoint_dir.glob("**/pytorch_model.bin"))
        if not checkpoints:
            checkpoints = list(checkpoint_dir.glob("**/model.safetensors"))
        
        if not checkpoints:
            logger.error("❌ Nessun checkpoint trovato in output directory!")
            return False
        
        checkpoint_path = checkpoints[-1]  # Prendi l'ultimo
        logger.info(f"📦 Usando checkpoint: {checkpoint_path}")
        
        # Parametri Fase 4
        layers_to_remove = self.config.get('layers_to_remove', [1, 2, 4, 6, 8, 9])
        
        # Costruisci comando
        cmd = [
            sys.executable,
            'scripts/fase4_estrazione_fisica.py',
            '--checkpoint', str(checkpoint_path),
            '--output', str(self.finale_path),
            '--layers_to_remove', *[str(i) for i in layers_to_remove],
            '--verify'
        ]
        
        logger.info(f"🚀 Esecuzione: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info("✅ FASE 4 completata con successo!")
            self.fase_completate.add(4)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FASE 4 fallita: {e}")
            return False
    
    def run_full_pipeline(self):
        """Esegue l'intera pipeline end-to-end."""
        logger.info("\n" + "🎯"*40)
        logger.info("MASTER PIPELINE - Distillazione Avanzata Completa")
        logger.info("🎯"*40 + "\n")
        
        start_time = datetime.now()
        
        # Fase 1: Forgia
        if not self.fase_1_forgia():
            logger.error("❌ Pipeline interrotta alla FASE 1")
            return False
        
        # Fase 2-3: Training
        if not self.fase_2_3_training():
            logger.error("❌ Pipeline interrotta alla FASE 2-3")
            return False
        
        # Fase 4: Estrazione
        if not self.fase_4_estrazione():
            logger.error("❌ Pipeline interrotta alla FASE 4")
            return False
        
        # Report finale
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "🎉"*40)
        logger.info("✅ PIPELINE COMPLETATA CON SUCCESSO!")
        logger.info("🎉"*40)
        logger.info(f"\n⏱️  Tempo totale: {duration}")
        logger.info(f"\n📦 Modello finale: {self.finale_path}")
        logger.info(f"📊 Dimensione: {self.finale_path.stat().st_size / (1024**2):.2f} MB")
        logger.info("\n🚀 Prossimi passi:")
        logger.info("   1. Testa il modello con inferenza")
        logger.info("   2. Valuta su dataset di test")
        logger.info("   3. Deploy in produzione")
        logger.info("\n" + "🎉"*40 + "\n")
        
        return True
    
    def run_fase(self, fase_num: int):
        """Esegue una singola fase specifica."""
        if fase_num == 1:
            return self.fase_1_forgia()
        elif fase_num in [2, 3]:
            return self.fase_2_3_training()
        elif fase_num == 4:
            return self.fase_4_estrazione()
        else:
            logger.error(f"❌ Fase {fase_num} non valida! Usa 1, 2, 3 o 4")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline per Distillazione Avanzata (SVD + Simplex + Progressive Pruning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Pipeline completa automatica
  python run_master_distillation.py --full_pipeline
  
  # Solo fase 1 (Forgia)
  python run_master_distillation.py --fase 1
  
  # Fasi 2-3 (Training) con configurazione custom
  python run_master_distillation.py --fase 2 --num_epochs 5 --batch_size 16
  
  # Solo fase 4 (Estrazione)
  python run_master_distillation.py --fase 4
        """
    )
    
    # Modalità esecuzione
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Esegue tutte le 4 fasi in sequenza')
    parser.add_argument('--fase', type=int, choices=[1, 2, 3, 4],
                        help='Esegue solo la fase specificata')
    
    # Parametri Fase 1 (Forgia)
    parser.add_argument('--teacher_path', type=str, default='pt_ckpt/params.sav',
                        help='Path al checkpoint del teacher')
    parser.add_argument('--dim_originale', type=int, default=1024,
                        help='Dimensione hidden del teacher')
    parser.add_argument('--nuova_dim', type=int, default=384,
                        help='Dimensione hidden dello student')
    parser.add_argument('--simplicial_layers', type=int, nargs='+', default=[3, 7, 11],
                        help='Indici (0-based) dei layer simpliciali')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='Rigenera studente_init anche se esiste')
    
    # Parametri Fase 2-3 (Training)
    parser.add_argument('--data_path', type=str, default='data/train_sequences.txt',
                        help='Path al dataset di training')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Numero di epoche di training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per training')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature per KD')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha per bilanciamento CE/KD loss')
    
    # Parametri Fase 4 (Estrazione)
    parser.add_argument('--layers_to_remove', type=int, nargs='+', default=[1, 2, 4, 6, 8, 9],
                        help='Layer da rimuovere fisicamente')
    
    # Output directories
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory per checkpoint intermedi')
    parser.add_argument('--training_output_dir', type=str, default='runs/distillation_master',
                        help='Directory output del training')
    
    args = parser.parse_args()
    
    # Converti args in dizionario config
    config = vars(args)
    
    # Inizializza pipeline
    pipeline = MasterPipeline(config)
    
    # Esegui pipeline
    success = False
    if args.full_pipeline:
        success = pipeline.run_full_pipeline()
    elif args.fase is not None:
        success = pipeline.run_fase(args.fase)
    else:
        parser.print_help()
        logger.error("\n❌ Devi specificare --full_pipeline oppure --fase N")
        sys.exit(1)
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
