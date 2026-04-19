"""
KDTrainer — Subclass di HuggingFace Trainer per Knowledge Distillation.

Questo è il cuore del sistema di distillazione.

Il Trainer standard chiama compute_loss(model, inputs) che di default usa
solo model(**inputs).loss. Noi lo sovrascriviamo per:
1. Fare il forward dello student (normale)
2. Fare il forward del teacher (frozen, no_grad)
3. Combinare CE loss + KD loss con la formula di Hinton

Contratto rispettato:
- Il teacher viene mantenuto SEMPRE in eval mode
- Il teacher NON è su self.model (che il Trainer gestisce), ma su self.teacher
- Le metriche KD (ce_loss, kd_loss) vengono loggate separatamente
"""

from typing import Optional, Tuple, Union

import torch
from transformers import Trainer
from transformers.utils import logging

from .kd_loss import KnowledgeDistillationLoss
from models.teacher_wrapper import TeacherWrapper

logger = logging.get_logger(__name__)


class KDTrainer(Trainer):
    """
    HuggingFace Trainer esteso con Knowledge Distillation loss.

    Args:
        teacher_model: TeacherWrapper (o nn.Module) del modello teacher.
                       DEVE essere già wrappato con TeacherWrapper per
                       garantire l'output in formato CausalLMOutputWithPast.
        temperature (float): Temperatura per soft targets. Default: 4.0.
        alpha (float): Peso della CE loss. Default: 0.5.
                       Loss totale = α*CE + (1-α)*KD.
        **kwargs: Tutti gli altri argomenti passati a Trainer
                  (model, args, train_dataset, eval_dataset, tokenizer, ...)
    """

    def __init__(
        self,
        teacher_model: Optional[TeacherWrapper] = None,
        temperature: float = 4.0,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.teacher = teacher_model
        self.kd_loss_fn = KnowledgeDistillationLoss(
            temperature=temperature,
            alpha=alpha,
        )

        # Sposta il teacher sullo stesso device del modello student (se presente)
        if self.teacher is not None:
            self._move_teacher_to_device()

    def _move_teacher_to_device(self):
        """Sposta il teacher sul device corretto dopo l'init del Trainer."""
        device = self.args.device
        if hasattr(self.teacher, "wrapped_teacher"):
            # TeacherWrapper: sposta il modello interno (operazione in-place)
            self.teacher.wrapped_teacher.to(device)
            if self.teacher.vocab_proj is not None:
                self.teacher.vocab_proj.to(device)
        else:
            self.teacher = self.teacher.to(device)

        # Garantisce che il teacher sia sempre in eval
        if hasattr(self.teacher, "wrapped_teacher"):
            if self.teacher.wrapped_teacher is not None:
                self.teacher.wrapped_teacher.eval()
        elif self.teacher is not None:
            self.teacher.eval()

        logger.info(
            f"✅ Teacher spostato su: {device} | "
            f"Temperature: {self.kd_loss_fn.temperature} | "
            f"Alpha: {self.kd_loss_fn.alpha}"
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, any]]:
        """
        Calcola la KD loss combinata CE + KL.

        Sovrascrive Trainer.compute_loss().

        Args:
            model: lo student model (gestito dal Trainer)
            inputs: batch dal DataLoader — contiene input_ids, attention_mask, labels
            return_outputs: se True, ritorna anche gli output dello student

        Returns:
            total_loss (Tensor scalar), oppure (total_loss, student_outputs)
        """
        labels = inputs.get("labels")

        # ------------------------------------------------------------------ #
        # 1. Forward dello Student
        # ------------------------------------------------------------------ #
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits   # (B, S, V)
        ce_loss = student_outputs.loss            # già calcolata dal modello

        # ------------------------------------------------------------------ #
        # 2. Forward del Teacher o caricamento Logit Offline
        # ------------------------------------------------------------------ #
        teacher_logits = None
        teacher_indices = inputs.get("teacher_indices")
        teacher_values = inputs.get("teacher_values")

        if teacher_indices is not None and teacher_values is not None:
            # MODALITÀ OFFLINE: Usiamo i logit già estratti dallo script JAX
            pass
        elif self.teacher is not None:
            # MODALITÀ ONLINE: Eseguiamo il teacher ora
            with torch.no_grad():
                teacher_inputs = {
                    k: v for k, v in inputs.items()
                    if k in ("input_ids", "attention_mask")
                }
                teacher_outputs = self.teacher(**teacher_inputs)
                teacher_logits = teacher_outputs.logits

            # Verifica shape
            if student_logits.shape != teacher_logits.shape:
                raise ValueError(
                    f"Shape mismatch: student={student_logits.shape} vs teacher={teacher_logits.shape}"
                )
        else:
            # Se non c'è né teacher né logit offline, non possiamo fare KD
            # In questo caso il trainer si comporterà come un Trainer standard (solo CE loss)
            pass

        # ------------------------------------------------------------------ #
        # 3. KD loss combinata
        # ------------------------------------------------------------------ #
        loss_dict = self.kd_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_indices=teacher_indices,
            teacher_values=teacher_values,
            labels=labels,
            ce_loss=ce_loss,
        )

        total_loss = loss_dict["total_loss"]

        # ------------------------------------------------------------------ #
        # 4. Logging delle metriche di distillazione
        # ------------------------------------------------------------------ #
        # Il Trainer le raccoglierà nel suo log history
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "kd/ce_loss": loss_dict["ce_loss"].item(),
                "kd/kd_loss": loss_dict["kd_loss"].item(),
                "kd/total_loss": total_loss.item(),
                "kd/temperature": loss_dict["temperature"],
                "kd/alpha": loss_dict["alpha"],
            })

        return (total_loss, student_outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """
        Sovrascrive training_step per assicurarsi che il teacher
        rimanga sempre in eval mode.
        """
        if self.teacher is not None:
            if hasattr(self.teacher, "wrapped_teacher") and self.teacher.wrapped_teacher is not None:
                self.teacher.wrapped_teacher.eval()
            else:
                self.teacher.eval()

        return super().training_step(model, inputs, *args, **kwargs)
