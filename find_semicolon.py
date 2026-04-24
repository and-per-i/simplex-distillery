import sentencepiece as spm
import os

model_path = "tokenizer/weights/geometry.757.model"
if not os.path.exists(model_path):
    model_path = "pt_ckpt/vocab.model"

sp = spm.SentencePieceProcessor()
sp.Load(model_path)

print(f"Token at 263: '{sp.IdToPiece(263)}'")
for i in range(sp.GetPieceSize()):
    piece = sp.IdToPiece(i)
    if piece == ';':
        print(f"Found ';' at {i}")
