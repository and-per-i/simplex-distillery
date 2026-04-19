"""
KnowledgeDistillationLoss — Loss function per Knowledge Distillation logit-based.

Implementa la formula di Hinton et al. (2015):
    L_KD = α * CE(student_logits, labels) + (1 - α) * T² * KL(soft_teacher || soft_student)

dove:
    - T (temperatura) smootha le distribuzioni del teacher esponendo il "dark knowledge"
    - α bilancia la loss CE (hard targets) con la loss KD (soft targets)
    - T² corregge la scala dei gradienti rispetto alla temperatura
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class KnowledgeDistillationLoss(nn.Module):
    """
    Loss combinata CE + KL-Divergence per knowledge distillation.

    Args:
        temperature (float): Temperatura per smoothing dei logits. Default: 4.0.
            Valori più alti → distribuzioni più morbide → più informazione sui rank.
            Range tipico: 2.0 – 8.0. Scegliere con validation set.
        alpha (float): Peso della CE loss (hard targets). Default: 0.5.
            (1 - alpha) è il peso della KD loss (soft targets).
            Range: 0.0 – 1.0.
            α→0: distillazione pura (solo soft targets)
            α→1: training standard (solo hard targets)
        reduction (str): "batchmean" (raccomandato per KLDiv) o "mean".
        ignore_index (int): Token ID da ignorare nella CE loss (default: -100, standard HF).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "batchmean",
        ignore_index: int = -100,
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, f"alpha deve essere in [0, 1], ricevuto: {alpha}"
        assert temperature > 0.0, f"temperature deve essere > 0, ricevuto: {temperature}"

        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_indices: Optional[torch.Tensor] = None,
        teacher_values: Optional[torch.Tensor] = None,
        ce_loss: torch.Tensor = None,
    ) -> dict:
        """
        Calcola la loss totale di distillazione.

        Args:
            student_logits: (B, S, V) — output dello student
            teacher_indices: (B, S, K) — indici dei Top-K logit del teacher
            teacher_values: (B, S, K) — valori dei Top-K logit del teacher
            ce_loss: se già calcolata dal modello, la riusa

        Returns:
            dict con chiavi:
                "total_loss": loss combinata (usata per backprop)
                "ce_loss": cross-entropy loss (hard targets)
                "kd_loss": KL-divergence loss (soft targets)
                "temperature": temperatura usata
                "alpha": alpha usato
        """
        B, S, V = student_logits.shape
        T = self.temperature

        # ------------------------------------------------------------------ #
        # 1. CE loss (hard targets) — standard next-token prediction
        # ------------------------------------------------------------------ #
        if ce_loss is None:
            # Shift per causal LM: predici posizione t+1 da posizione t
            shift_student = student_logits[..., :-1, :].contiguous()   # (B, S-1, V)
            shift_labels = labels[..., 1:].contiguous()                 # (B, S-1)
            ce_loss = F.cross_entropy(
                shift_student.view(-1, V),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
            )

        # ------------------------------------------------------------------ #
        # 2. KD loss (soft targets)
        # ------------------------------------------------------------------ #
        kd_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Filtriamo le posizioni valide (no padding)
        valid_mask = (labels[..., 1:] != self.ignore_index)
        s_logits = student_logits[..., :-1, :][valid_mask] # (N_valid, V)

        if teacher_logits is not None:
            # Caso 1: Logit completi (meno probabile con 2.6M righe)
            t_logits = teacher_logits[..., :-1, :][valid_mask]
            
            student_soft = F.log_softmax(s_logits / T, dim=-1)
            teacher_soft = F.softmax(t_logits / T, dim=-1)
            kd_loss = F.kl_div(student_soft, teacher_soft, reduction=self.reduction) * (T**2)
            
        elif teacher_indices is not None and teacher_values is not None:
            # Caso 2: Top-K Distillation (molto più efficiente)
            t_idx = teacher_indices[..., :-1, :][valid_mask]  # (N_valid, K)
            t_val = teacher_values[..., :-1, :][valid_mask]   # (N_valid, K)
            
            # Distribuzione soft del Teacher (solo sui K token salvati)
            teacher_soft = F.softmax(t_val / T, dim=-1) # (N_valid, K)
            
            # Per lo Student, dobbiamo estrarre i logit corrispondenti agli indici del Teacher
            # s_logits ha shape (N_valid, V). Usiamo gather per prendere i K indici.
            s_gathered = torch.gather(s_logits, dim=-1, index=t_idx) # (N_valid, K)
            
            # Importante: calcoliamo il log_softmax dello student RISTRETTO ai top-K del teacher?
            # No, Hint: per una distillazione corretta, lo student dovrebbe essere softmax-ato 
            # su tutto il vocabolario, ma qui confrontiamo solo i rami del teacher.
            # Una versione robusta è usare log_softmax globale e poi gather.
            student_log_soft_all = F.log_softmax(s_logits / T, dim=-1)
            student_log_soft_gathered = torch.gather(student_log_soft_all, dim=-1, index=t_idx)
            
            # KL Div approssimata sui Top-K: sum( P_t * (log P_t - log P_s) )
            # Nota: P_t * log P_t è l'entropia del teacher (costante rispetto allo student)
            # Durante la distillazione ci interessa minimizzare: -sum( P_t * log P_s )
            kd_loss = -(teacher_soft * student_log_soft_gathered).sum(dim=-1).mean() * (T**2)

        # ------------------------------------------------------------------ #
        # 3. Loss combinata
        # ------------------------------------------------------------------ #
        total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "temperature": T,
            "alpha": self.alpha,
        }

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, alpha={self.alpha}, "
            f"reduction={self.reduction}"
        )
