import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

class VocabProjectionHead(nn.Module):
    """
    Projection head per mappare i logit del Teacher sul vocabolario dello Student.
    Necessario quando i vocabolari non sono identici (es. AG=1024, Student=757).
    """
    def __init__(self, teacher_vocab_size: int, student_vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(teacher_vocab_size, student_vocab_size, bias=False)
        # Inizializzazione identità approssimativa (se possibile) o random
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class TeacherWrapper(nn.Module):
    """
    Wrapper per il modello AlphaGeometry (Teacher) per renderlo compatibile con il KDTrainer.
    Gestisce la proiezione del vocabolario e uniforma l'output.
    """
    def __init__(
        self, 
        wrapped_teacher: nn.Module, 
        student_vocab_size: int,
        logits_key: str = "logits",
        logits_tuple_idx: int = 0
    ):
        super().__init__()
        self.wrapped_teacher = wrapped_teacher
        self.student_vocab_size = student_vocab_size
        self.logits_key = logits_key
        self.logits_tuple_idx = logits_tuple_idx
        
        # Determina la dimensione del vocabolario del teacher
        # Rileva la dimensione REALE dell'output del teacher
        # Anche se il vocab è 757, AlphaGeometry spesso sputa 1024
        self.teacher_vocab_size = self._detect_vocab_size()
        self.student_vocab_size = 757 # Fissato come da tokenizer
        
        print(f"DEBUG: Teacher raw output: {self.teacher_vocab_size} | Student vocab: {self.student_vocab_size}")

        # Se il teacher sputa più token dello student (es. 1024 vs 757), 
        # facciamo uno slicing diretto dei primi 757 token.
        # È più preciso di una proiezione lineare perché i primi 757 ID sono identici.
        self.should_truncate = self.teacher_vocab_size > self.student_vocab_size
        self.vocab_proj = nn.Identity() # Non usiamo pesi da addestrare

    def _detect_vocab_size(self) -> int:
        """Prova a inferire la dimensione del vocabolario del teacher."""
        # Cerchiamo nello stato o nella config del modello originale
        for attr in ["config", "cfg", "params"]:
            obj = getattr(self.wrapped_teacher, attr, None)
            if obj:
                for key in ["vocab_size", "output_dim", "n_vocab"]:
                    val = getattr(obj, key, None) or (obj.get(key) if isinstance(obj, dict) else None)
                    if val: return val
        
        # Fallback: guarda l'ultima layer lineare
        for module in reversed(list(self.wrapped_teacher.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        
        return 1024 # Default per AlphaGeometry

    def forward(self, *args, **inputs) -> CausalLMOutputWithPast:
        """
        Forward pass che uniforma l'output in formato HuggingFace.
        """
        # --- Gestione robusta input (anche per modelli compilati) ---
        tokens = inputs.pop("input_ids", None)
        if tokens is None and len(args) > 0:
            tokens = args[0]
            args = args[1:]
        
        # Rimuoviamo attention_mask e labels (non supportate dal Decoder originale)
        inputs.pop("attention_mask", None)
        inputs.pop("labels", None)

        # Chiamata al modello originale (xs è il primo argomento posizionale)
        raw_output = self.wrapped_teacher(tokens, *args, **inputs)

        # --- Estrazione Logits ---
        if hasattr(raw_output, "logits"):
            logits = raw_output.logits
        elif isinstance(raw_output, dict):
            logits = raw_output[self.logits_key]
        elif isinstance(raw_output, (tuple, list)):
            logits = raw_output[self.logits_tuple_idx]
        elif isinstance(raw_output, torch.Tensor):
            logits = raw_output
        else:
            raise ValueError(f"Impossibile estrarre i logits dall'output del teacher: {type(raw_output)}")

        # --- Proiezione / Troncamento Vocabolario ---
        if self.should_truncate:
            # Slicing dei primi N token (es. prendi i primi 757 da 1024)
            logits = logits[..., :self.student_vocab_size]
        
        if self.vocab_proj:
            logits = self.vocab_proj(logits)

        return CausalLMOutputWithPast(logits=logits)

    @property
    def device(self):
        return next(self.wrapped_teacher.parameters()).device
