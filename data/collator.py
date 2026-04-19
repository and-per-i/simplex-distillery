"""
GeometryDataCollator — DataCollator HuggingFace-compatible per sequenze geometriche.

Gestisce:
- Padding dinamico (al batch, non al max_length fisso) → efficiente
- Padding di labels con -100 (standard HF per ignorare le posizioni paddate)
- attention_mask aggiornata correttamente
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


@dataclass
class GeometryDataCollator:
    """
    DataCollator con dynamic padding per sequenze geometriche.

    Il Trainer HF chiama questo collator per ogni batch.
    Garantisce che:
        - input_ids siano paddati al token pad_token_id
        - attention_mask sia 0 sulle posizioni paddate
        - labels siano paddati con -100 (ignore_index per CE loss)

    Args:
        tokenizer: tokenizer usato per ottenere pad_token_id
        pad_to_multiple_of: se impostato, padda a multipli di questo valore
                            (utile per ottimizzare operazioni CUDA su tensori allineati)
        label_pad_token_id: ID usato per paddare i labels (default: -100)
    """

    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Trova la lunghezza massima nel batch (dynamic padding)
        max_len = max(f["input_ids"].shape[0] for f in features)

        # Opzionale: padda a multipli di 8/16/64 per efficienza GPU
        if self.pad_to_multiple_of is not None:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        pad_id = self.tokenizer.pad_token_id or 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # input_ids: padda con pad_token_id
            input_ids = torch.cat([
                f["input_ids"],
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])

            # attention_mask: 1 per token validi, 0 per padding
            attention_mask = torch.cat([
                f["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ])

            # labels: padda con -100 (ignorato dalla CE loss)
            labels = torch.cat([
                f["labels"],
                torch.full((pad_len,), self.label_pad_token_id, dtype=torch.long),
            ])

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids),           # (B, max_len)
            "attention_mask": torch.stack(batch_attention_mask),  # (B, max_len)
            "labels": torch.stack(batch_labels),                  # (B, max_len)
        }
