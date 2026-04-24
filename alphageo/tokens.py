"""
Token IDs per il vocabolario AlphaGeometry.
Basato sul file pt_ckpt/vocab.model (SentencePiece).

Questi ID sono stati verificati tramite ispezione del vocab file e test empirici.
"""

# Token speciali standard
PAD_ID = 0          # <pad>
EOS_ID = 1          # </s>
BOS_ID = 2          # <s>
UNK_ID = 3          # <unk>

# Token semantici per geometria
SEMICOLON_ID = 263  # '▁;' - EOS marker primario per sequenze geometriche

# Alias per compatibilità con il codice esistente
GEOMETRY_EOS_ID = SEMICOLON_ID  # Usa sempre semicolon come EOS in geometria
