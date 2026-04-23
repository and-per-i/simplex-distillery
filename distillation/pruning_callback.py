from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import torch

class ProgressivePruningCallback(TrainerCallback):
    """
    Fase 3: Il Progressive Pruning.
    Spegne gradualmente i layer dello studente durante l'addestramento.
    """
    
    def __init__(self, schedule=None):
        """
        schedule: dict {epoca: [lista indici layer da spegnere]}
        Esempio: {2: [1, 9], 3: [2, 8], 4: [4, 6]}
        """
        if schedule is None:
            # Timeline di default dal Master Plan
            self.schedule = {
                2: [1, 9],  # Epoca 2: Layer 2 e 10 (0-indexed: 1 e 9)
                3: [2, 8],  # Epoca 3: Layer 3 e 9 (0-indexed: 2 e 8)
                4: [4, 6],  # Epoca 4: Layer 5 e 7 (0-indexed: 4 e 6)
            }
        else:
            self.schedule = schedule
            
        self.bypassed_layers = set()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = int(state.epoch) + 1 # state.epoch parte da 0.0
        
        if epoch in self.schedule:
            model = kwargs.get("model")
            if model is None:
                return

            # Se il modello è wrapped in DistributedDataParallel o simili
            if hasattr(model, "module"):
                actual_model = model.module
            else:
                actual_model = model

            # Accediamo alla lista dei layer
            # In StudentForCausalLM è in model.model.layers
            if hasattr(actual_model, "model") and hasattr(actual_model.model, "layers"):
                layers = actual_model.model.layers
                
                for layer_idx in self.schedule[epoch]:
                    if layer_idx < len(layers):
                        layers[layer_idx].is_bypassed = True
                        self.bypassed_layers.add(layer_idx)
                        print(f"\n[Pruning] Epoca {epoch}: Layer {layer_idx + 1} BYPASSATO.")
                    else:
                        print(f"\n[Warning] Tentativo di bypassare layer {layer_idx + 1} inesistente.")
            else:
                print("\n[Error] Struttura del modello non compatibile con il Pruning Callback.")
