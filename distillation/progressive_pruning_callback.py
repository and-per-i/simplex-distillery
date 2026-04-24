"""
FASE 3: Progressive Pruning Callback
=====================================

Callback per Hugging Face Trainer che implementa il Progressive Pruning:
- Spegne gradualmente i layer durante il training
- Timeline automatica basata sulle epoche
- Logging dettagliato dello stato del pruning

Timeline di Spegnimento:
- Epoca 1: Tutti e 12 layer attivi
- Epoca 2: Bypass layer 2, 10 (rimangono 10 attivi)
- Epoca 3: Bypass layer 3, 9 (rimangono 8 attivi)
- Epoca 4: Bypass layer 5, 7 (rimangono 6 attivi = 3 standard + 3 simpliciali)
"""

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)


class ProgressivePruningCallback(TrainerCallback):
    """
    Callback per spegnere progressivamente i layer durante il training.
    
    Usage:
        callback = ProgressivePruningCallback(
            pruning_schedule={
                2: [1, 9],   # Epoca 2: spegni layer 2 e 10 (0-indexed: 1, 9)
                3: [2, 8],   # Epoca 3: spegni layer 3 e 9
                4: [4, 6]    # Epoca 4: spegni layer 5 e 7
            }
        )
        trainer = Trainer(..., callbacks=[callback])
    """
    
    def __init__(self, pruning_schedule: dict = None):
        """
        Args:
            pruning_schedule: Dizionario {epoca: [layer_indices]}
                              Se None, usa lo schedule di default
        """
        super().__init__()
        
        # Default schedule: quello del master plan
        if pruning_schedule is None:
            pruning_schedule = {
                2: [1, 9],   # Epoca 2: layer 2, 10 (zero-indexed)
                3: [2, 8],   # Epoca 3: layer 3, 9
                4: [4, 6]    # Epoca 4: layer 5, 7
            }
        
        self.pruning_schedule = pruning_schedule
        self.current_epoch = 0
        self.pruned_layers = set()
        
        logger.info("=" * 70)
        logger.info("📅 PROGRESSIVE PRUNING SCHEDULE")
        logger.info("=" * 70)
        for epoch, layers in sorted(pruning_schedule.items()):
            layer_str = ", ".join([f"Layer {i+1}" for i in layers])
            logger.info(f"  Epoca {epoch}: Bypass {layer_str}")
        logger.info("=" * 70)
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, model=None, **kwargs):
        """
        Chiamato all'inizio di ogni epoca.
        Controlla se ci sono layer da spegnere.
        """
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Esci se non c'è schedule per questa epoca
        if epoch not in self.pruning_schedule:
            return
        
        # Ottieni i layer da spegnere
        layers_to_prune = self.pruning_schedule[epoch]
        
        logger.info("\n" + "🔥" * 35)
        logger.info(f"⚡ EPOCH {epoch}: ATTIVAZIONE PROGRESSIVE PRUNING")
        logger.info("🔥" * 35)
        
        # Spegni i layer specificati
        for layer_idx in layers_to_prune:
            if hasattr(model, 'module'):  # Se wrapped in DataParallel
                model.module.layers[layer_idx].is_bypassed = True
            else:
                model.layers[layer_idx].is_bypassed = True
            
            self.pruned_layers.add(layer_idx)
            logger.info(f"  🚫 Layer {layer_idx+1:2d} → BYPASSED")
        
        # Statistiche
        total_layers = 12  # AlphaGeometry standard
        active_layers = total_layers - len(self.pruned_layers)
        pruning_ratio = (len(self.pruned_layers) / total_layers) * 100
        
        logger.info(f"\n📊 Stato Modello:")
        logger.info(f"  - Layer attivi: {active_layers}/{total_layers}")
        logger.info(f"  - Layer bypassed: {len(self.pruned_layers)}/{total_layers}")
        logger.info(f"  - Pruning ratio: {pruning_ratio:.1f}%")
        
        # Mostra layer ancora attivi
        active_list = []
        for i in range(total_layers):
            is_bypassed = (hasattr(model, 'module') and model.module.layers[i].is_bypassed) or \
                          model.layers[i].is_bypassed
            if not is_bypassed:
                active_list.append(i+1)
        
        logger.info(f"  - Layer attivi: {active_list}")
        logger.info("🔥" * 35 + "\n")
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, model=None, **kwargs):
        """
        Chiamato alla fine di ogni epoca.
        Log dello stato finale.
        """
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Se abbiamo fatto pruning in questa epoca, logga i risultati
        if epoch in self.pruning_schedule:
            logger.info(f"\n✅ Epoca {epoch} completata con pruning attivo")
            
            # Calcola metriche se disponibili
            if state.log_history:
                last_log = state.log_history[-1]
                if 'loss' in last_log:
                    logger.info(f"  - Training Loss: {last_log['loss']:.4f}")
                if 'eval_loss' in last_log:
                    logger.info(f"  - Eval Loss: {last_log['eval_loss']:.4f}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, model=None, **kwargs):
        """
        Chiamato alla fine del training.
        Report finale sullo stato del pruning.
        """
        logger.info("\n" + "="*70)
        logger.info("🎉 PROGRESSIVE PRUNING COMPLETATO")
        logger.info("="*70)
        
        total_layers = 12
        final_active = total_layers - len(self.pruned_layers)
        
        logger.info(f"📊 Risultati Finali:")
        logger.info(f"  - Layer originali: {total_layers}")
        logger.info(f"  - Layer finali attivi: {final_active}")
        logger.info(f"  - Layer eliminati: {len(self.pruned_layers)}")
        logger.info(f"  - Riduzione: {(len(self.pruned_layers)/total_layers)*100:.1f}%")
        
        # Lista layer simplicial rimasti
        simplicial_layers = [3, 7, 11]  # 0-indexed
        active_simplicial = [i+1 for i in simplicial_layers if i not in self.pruned_layers]
        logger.info(f"  - Layer Simpliciali attivi: {active_simplicial}")
        
        logger.info("="*70)
        logger.info("💡 Pronto per FASE 4: Estrazione Fisica del Modello")
        logger.info("="*70 + "\n")


class ProgressivePruningCallbackCustom(TrainerCallback):
    """
    Versione avanzata con controllo più fine sul pruning.
    
    Features:
    - Warmup prima di iniziare pruning
    - Gradualità configurabile
    - Adaptive pruning basato su loss
    """
    
    def __init__(self, 
                 start_epoch: int = 2,
                 warmup_steps: int = 0,
                 adaptive: bool = False,
                 loss_threshold: float = None):
        """
        Args:
            start_epoch: Epoca da cui iniziare il pruning
            warmup_steps: Step di warmup prima di ogni pruning
            adaptive: Se True, pruning basato su loss invece che su schedule fisso
            loss_threshold: Threshold di loss per adaptive pruning
        """
        super().__init__()
        self.start_epoch = start_epoch
        self.warmup_steps = warmup_steps
        self.adaptive = adaptive
        self.loss_threshold = loss_threshold
        self.pruned_layers = set()
        
        # Layers da eliminare in ordine (quelli NON simpliciali)
        # Simmetrico rispetto al centro: 2,10 → 3,9 → 5,7
        self.prune_order = [
            [1, 9],   # Layer 2, 10
            [2, 8],   # Layer 3, 9
            [4, 6]    # Layer 5, 7
        ]
        self.current_prune_stage = 0
    
    def should_prune(self, state: TrainerState) -> bool:
        """Decide se è il momento di fare pruning."""
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Non iniziare prima dell'epoca designata
        if epoch < self.start_epoch:
            return False
        
        # Controlla se abbiamo già fatto tutte le fasi
        if self.current_prune_stage >= len(self.prune_order):
            return False
        
        # Se adaptive, controlla loss
        if self.adaptive and self.loss_threshold:
            if state.log_history:
                recent_loss = state.log_history[-1].get('loss', float('inf'))
                if recent_loss > self.loss_threshold:
                    logger.info(f"⏸️  Pruning posticipato: loss={recent_loss:.4f} > threshold={self.loss_threshold:.4f}")
                    return False
        
        # Controlla se è l'epoca giusta per questa fase
        epochs_since_start = epoch - self.start_epoch
        if epochs_since_start == self.current_prune_stage:
            return True
        
        return False
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, model=None, **kwargs):
        """Adaptive pruning all'inizio dell'epoca."""
        if not self.should_prune(state):
            return
        
        # Esegui pruning
        layers_to_prune = self.prune_order[self.current_prune_stage]
        epoch = int(state.epoch)
        
        logger.info(f"\n⚡ ADAPTIVE PRUNING - Epoch {epoch} - Stage {self.current_prune_stage+1}/{len(self.prune_order)}")
        
        for layer_idx in layers_to_prune:
            if hasattr(model, 'module'):
                model.module.layers[layer_idx].is_bypassed = True
            else:
                model.layers[layer_idx].is_bypassed = True
            
            self.pruned_layers.add(layer_idx)
            logger.info(f"  🚫 Layer {layer_idx+1} → BYPASSED")
        
        self.current_prune_stage += 1
