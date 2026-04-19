"""
TeacherWrapper — Wrapper che garantisce la compatibilità tra teacher e KDTrainer.

Problema: il teacher potrebbe essere:
  (a) Un modello HuggingFace standard → già compatibile
  (b) Un modello custom (es. JAX/Flax AlphaGeometry) → richiede adattamento
  (c) Un modello PyTorch non-HF → richiede wrapper dell'output

Questo modulo fornisce:
  1. TeacherWrapper: wrap generico per allineare output a CausalLMOutputWithPast
  2. VocabProjectionHead: per allineare vocab_size se teacher e student divergono
  3. load_teacher_from_hf(): helper per caricare teacher da HuggingFace Hub
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class VocabProjectionHead(nn.Module):
    """
    Proiezione lineare per allineare vocab_size del teacher a quello dello student.

    Usata SOLO quando teacher_vocab_size != student_vocab_size.
    Nota: questa è una soluzione di ripiego — è preferibile usare lo stesso tokenizer.

    Args:
        teacher_vocab_size: dimensione vocab del teacher
        student_vocab_size: dimensione vocab dello student (= 757)
    """

    def __init__(self, teacher_vocab_size: int, student_vocab_size: int):
        super().__init__()
        if teacher_vocab_size == student_vocab_size:
            raise ValueError(
                "VocabProjectionHead non necessaria: vocab_size già uguale."
            )
        warnings.warn(
            f"⚠️  Teacher vocab_size ({teacher_vocab_size}) ≠ Student vocab_size "
            f"({student_vocab_size}). Verrà usata una projection head lineare. "
            "Questo riduce la qualità della distillazione — preferibile allineare i vocab.",
            UserWarning,
            stacklevel=2,
        )
        self.proj = nn.Linear(teacher_vocab_size, student_vocab_size, bias=False)
        # Inizializzazione: identità approssimata sui primi min(V_t, V_s) token
        nn.init.zeros_(self.proj.weight)
        min_v = min(teacher_vocab_size, student_vocab_size)
        with torch.no_grad():
            self.proj.weight[:min_v, :min_v] = torch.eye(min_v)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, S, teacher_vocab_size) → (B, S, student_vocab_size)
        return self.proj(logits)


class TeacherWrapper(nn.Module):
    """
    Wrapper che uniforma l'output di qualsiasi teacher al formato CausalLMOutputWithPast.

    Il KDTrainer si aspetta teacher_outputs.logits con shape (B, S, vocab_size).
    Questo wrapper garantisce che sia sempre così, anche se il teacher:
    - ritorna una plain tuple
    - ha vocab_size diverso dallo student
    - è un modello non-HuggingFace

    Args:
        teacher_model: il modello teacher (nn.Module qualsiasi)
        student_vocab_size: vocab_size dello student per verifica/proiezione
        logits_key: se il teacher ritorna un dict, la chiave per i logits
        logits_tuple_idx: se il teacher ritorna una tuple, l'indice dei logits
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_vocab_size: int = 757,
        logits_key: str = "logits",
        logits_tuple_idx: int = 1,  # spesso [loss, logits, ...]
    ):
        super().__init__()
        print(f"DEBUG: Initializing TeacherWrapper with model: {type(teacher_model)}")
        self.wrapped_teacher = teacher_model
        print(f"DEBUG: self.wrapped_teacher set to: {type(self.wrapped_teacher)}")
        self.student_vocab_size = student_vocab_size
        self.logits_key = logits_key
        self.logits_tuple_idx = logits_tuple_idx

        # Determina il vocab_size del teacher (per la proiezione se necessario)
        self._teacher_vocab_size = self._detect_teacher_vocab_size()
        self._needs_projection = (
            self._teacher_vocab_size is not None
            and self._teacher_vocab_size != student_vocab_size
        )
        if self._needs_projection:
            self.vocab_proj = VocabProjectionHead(
                self._teacher_vocab_size, student_vocab_size
            )
        else:
            self.vocab_proj = None

        # Il teacher è sempre frozen
        self.wrapped_teacher.eval()
        for p in self.wrapped_teacher.parameters():
            p.requires_grad = False

    def _detect_teacher_vocab_size(self) -> Optional[int]:
        """Rileva il vocab_size del teacher dal config o dagli attributi."""
        vsize = None
        # Caso HF: ha config.vocab_size (oggetto)
        if hasattr(self.wrapped_teacher, "config"):
            config = self.wrapped_teacher.config
            if hasattr(config, "vocab_size"):
                vsize = config.vocab_size
            elif isinstance(config, dict) and "vocab_size" in config:
                vsize = config["vocab_size"]
        
        # Caso generico: prova a trovare il lm_head o embedding
        if vsize is None:
            for name in ["lm_head", "output_projection", "decoder", "embedding"]:
                mod = getattr(self.wrapped_teacher, name, None)
                if isinstance(mod, nn.Linear):
                    vsize = mod.out_features
                    break
                elif isinstance(mod, nn.Embedding):
                    vsize = mod.num_embeddings
                    break
        
        print(f"DEBUG: Detected teacher vocab_size: {vsize}")
        return vsize

    @torch.no_grad()
    def forward(self, **inputs) -> CausalLMOutputWithPast:
        """
        Forward del teacher con normalizzazione dell'output.

        Returns:
            CausalLMOutputWithPast con .logits di shape (B, S, student_vocab_size)
        """
        # --- Gestione nomi argomenti (es. input_ids -> xs per AG Decoder) ---
        if "input_ids" in inputs:
            import inspect
            sig = inspect.signature(self.wrapped_teacher.forward)
            if "xs" in sig.parameters and "input_ids" not in sig.parameters:
                inputs["xs"] = inputs.pop("input_ids")
            elif "input_tokens" in sig.parameters and "input_ids" not in sig.parameters:
                inputs["input_tokens"] = inputs.pop("input_ids")

        # Rimuovi attention_mask se il teacher non lo supporta (AG Decoder non lo supporta)
        if "attention_mask" in inputs:
            sig = inspect.signature(self.wrapped_teacher.forward)
            if "attention_mask" not in sig.parameters:
                inputs.pop("attention_mask")

        raw_output = self.wrapped_teacher(**inputs)

        # --- Estrai i logits dal formato di output ---
        if hasattr(raw_output, "logits"):
            # Caso HF standard: ModelOutput con .logits
            logits = raw_output.logits
        elif isinstance(raw_output, dict):
            logits = raw_output[self.logits_key]
        elif isinstance(raw_output, (tuple, list)):
            logits = raw_output[self.logits_tuple_idx]
        elif isinstance(raw_output, torch.Tensor):
            logits = raw_output
        else:
            raise TypeError(
                f"Formato output del teacher non riconosciuto: {type(raw_output)}. "
                "Imposta logits_key o logits_tuple_idx nel TeacherWrapper."
            )

        # --- Proiezione vocab se necessario ---
        if self._needs_projection and self.vocab_proj is not None:
            logits = self.vocab_proj(logits)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=getattr(raw_output, "hidden_states", None),
            attentions=None,
        )

    def train(self, mode: bool = True):
        """Il teacher rimane SEMPRE in eval mode, indipendentemente dal training loop."""
        super().train(mode)
        if hasattr(self, 'wrapped_teacher') and self.wrapped_teacher is not None:
            self.wrapped_teacher.eval()
        return self


# ---------------------------------------------------------------------------
# Helper per caricare il teacher
# ---------------------------------------------------------------------------


def load_teacher_from_hf(
    model_name_or_path: str,
    student_vocab_size: int = 757,
    device: str = "cpu",
    torch_dtype=torch.float32,
) -> TeacherWrapper:
    """
    Carica un teacher da HuggingFace Hub o da un path locale.

    Args:
        model_name_or_path: es. "gpt2", "meta-llama/Llama-2-7b", o path locale.
        student_vocab_size: vocab_size dello student per verifica compatibilità.
        device: dispositivo su cui caricare il teacher.
        torch_dtype: dtype per il teacher (usa torch.float16 per modelli grandi).

    Returns:
        TeacherWrapper pronto per il KDTrainer.

    Esempio:
        teacher = load_teacher_from_hf("gpt2", student_vocab_size=757)
    """
    print(f"📥 Caricamento teacher da: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
    ).to(device)

    teacher_vocab = model.config.vocab_size
    if teacher_vocab != student_vocab_size:
        warnings.warn(
            f"⚠️  Teacher vocab_size={teacher_vocab} ≠ student_vocab_size={student_vocab_size}. "
            "Verrà aggiunta una VocabProjectionHead. "
            "Per risultati ottimali, usa un teacher con vocab_size=757.",
            UserWarning,
        )

    return TeacherWrapper(model, student_vocab_size=student_vocab_size)
