"""
GeometryDataset — Dataset HuggingFace-compatible per sequenze geometriche AlphaGeometry.

Il Trainer HF si aspetta che il dataset ritorni dict con:
    {
        "input_ids":      torch.LongTensor (S,)
        "attention_mask": torch.LongTensor (S,)
        "labels":         torch.LongTensor (S,)  # -100 per le posizioni da ignorare
    }

Formato dei dati di input (testo):
    Ogni riga è una sequenza geometrica tokenizzata, es:
    "a b c coll a b c ; d e f para d e f ;"

Supporto per:
    - File di testo (.txt) con una sequenza per riga
    - Liste Python di stringhe in memoria
    - Generazione sintetica per test
"""

import os
import random
from typing import List, Optional, Union

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from tokenizer.hf_tokenizer import AlphaGeometryHFTokenizer


class GeometryDataset(Dataset):
    """
    Dataset per sequenze geometriche compatibile con HuggingFace Trainer.

    Args:
        sequences: Lista di stringhe, o path a file .txt (una sequenza per riga).
        tokenizer: AlphaGeometryHFTokenizer (o qualsiasi PreTrainedTokenizer).
        max_length: Lunghezza massima della sequenza (truncation/padding).
        stride: Per sliding window su sequenze lunghe. 0 = nessuno stride.
        add_bos: Se True, aggiunge BOS token all'inizio.
        add_eos: Se True, aggiunge EOS token alla fine.
    """

    def __init__(
        self,
        sequences: Union[str, List[str]],
        tokenizer: AlphaGeometryHFTokenizer,
        max_length: int = 512,
        stride: int = 0,
        add_bos: bool = True,
        add_eos: bool = True,
        is_distillation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.is_distillation = is_distillation

        self.samples = []
        self.teacher_indices = None
        self.teacher_values = None

        # Carica le sequenze
        if isinstance(sequences, str):
            self._load_from_file(sequences)
        elif isinstance(sequences, list):
            self.samples = sequences
        else:
            raise TypeError(
                f"sequences deve essere str (path) o List[str], ricevuto: {type(sequences)}"
            )

        print(f"📊 Dataset caricato: {len(self.samples)} sequenze")
        if self.is_distillation:
            print(f"🧠 Distillation mode attiva: {self.teacher_indices is not None}")

    def _load_from_file(self, path: str) -> List[str]:
        """Carica sequenze da file .txt o .parquet."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file non trovato: {path}")
        
        # Se è un file Parquet (es. dataset1)
        if path.endswith(".parquet") or "dataset" in path:
            try:
                print(f"📦 Caricamento Parquet: {path}...")
                df = pd.read_parquet(path)
                
                # Se siamo in modalità distillazione, cerchiamo le colonne dei logit
                if self.is_distillation and "top_k_indices" in df.columns:
                    print("✨ Trovati logit del Teacher nel Parquet!")
                    self.teacher_indices = df["top_k_indices"].tolist()
                    self.teacher_values = df["top_k_values"].tolist()

                # Combina question e solution con uno spazio
                self.samples = (df["question"] + " " + df["solution"]).tolist()
                print(f"✅ Parquet caricato: {len(self.samples)} campioni")
            except Exception as e:
                print(f"⚠️ Errore lettura Parquet, provo come testo: {e}")
                with open(path, "r", encoding="utf-8") as f:
                    self.samples = [line.strip() for line in f if line.strip()]

        # Fallback al file di testo (.txt)
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"📂 Caricato da file di testo: {path} → {len(lines)} righe")
        return lines

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text = self.samples[idx]

        # Tokenizza senza padding (lo gestisce il DataCollator)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,  # ritorna liste Python, non tensori
            add_special_tokens=False,  # gestiamo noi BOS/EOS
        )
        input_ids = encoding["input_ids"]

        # Aggiungi BOS / EOS
        if self.add_bos and self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        # Trunca di nuovo se BOS/EOS hanno sforato
        input_ids = input_ids[: self.max_length]

        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Se abbiamo i logit del teacher, aggiungiamoli
        if self.is_distillation and self.teacher_indices is not None:
            # teacher_indices shape: (Seq, TopK)
            # Dobbiamo assicurarci che la lunghezza corrisponda a input_ids
            # Durante l'estrazione abbiamo usato seq_len fisso (es. 128)
            t_idx = np.array(self.teacher_indices[idx])
            t_val = np.array(self.teacher_values[idx])
            
            # Troncamento/Padding per matchare input_ids se necessario
            curr_len = len(input_ids)
            if t_idx.shape[0] > curr_len:
                t_idx = t_idx[:curr_len]
                t_val = t_val[:curr_len]
            elif t_idx.shape[0] < curr_len:
                # Padding con zeri/valori neutri (non dovrebbe succedere se seq_len è uguale)
                pad_w = curr_len - t_idx.shape[0]
                t_idx = np.pad(t_idx, ((0, pad_w), (0, 0)), constant_values=0)
                t_val = np.pad(t_val, ((0, pad_w), (0, 0)), constant_values=-100.0)

            output["teacher_indices"] = torch.tensor(t_idx, dtype=torch.long)
            output["teacher_values"] = torch.tensor(t_val, dtype=torch.float)

        return output


# ---------------------------------------------------------------------------
# Generatore di sequenze sintetiche per test/debug
# ---------------------------------------------------------------------------

_GEOMETRY_PREDICATES = [
    "coll", "para", "perp", "midp", "eqangle", "eqratio",
    "cyclic", "foot", "circle", "tangent",
]

_POINT_LABELS = list("abcdefghijklmnopqrstuvwxyz")


def generate_synthetic_sequences(
    n: int = 1000,
    min_predicates: int = 1,
    max_predicates: int = 4,
    seed: int = 42,
) -> List[str]:
    """
    Genera sequenze geometriche sintetiche per test e pre-training.

    Formato: "<pred> <p1> <p2> ... ; <pred> <p1> <p2> ... ;"

    Args:
        n: numero di sequenze da generare
        min_predicates: numero minimo di predicati per sequenza
        max_predicates: numero massimo di predicati per sequenza
        seed: seed per riproducibilità

    Returns:
        Lista di n stringhe
    """
    random.seed(seed)
    sequences = []

    for _ in range(n):
        n_pred = random.randint(min_predicates, max_predicates)
        parts = []
        for _ in range(n_pred):
            pred = random.choice(_GEOMETRY_PREDICATES)
            n_args = random.randint(2, 4)
            args = random.sample(_POINT_LABELS[:10], min(n_args, 10))
            parts.append(f"{pred} {' '.join(args)}")
        sequences.append(" ; ".join(parts) + " ;")

    return sequences


def make_test_dataset(
    tokenizer: AlphaGeometryHFTokenizer,
    n: int = 100,
    max_length: int = 128,
) -> "GeometryDataset":
    """
    Crea un piccolo dataset sintetico per smoke test.

    Uso:
        ds = make_test_dataset(tokenizer)
        sample = ds[0]
        print(sample["input_ids"].shape)
    """
    seqs = generate_synthetic_sequences(n=n)
    return GeometryDataset(seqs, tokenizer, max_length=max_length)
