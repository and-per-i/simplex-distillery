import sentencepiece as spm
import os

model_path = "tokenizer/weights/geometry.757.model"
if not os.path.exists(model_path):
    model_path = "pt_ckpt/vocab.model"

sp = spm.SentencePieceProcessor()
sp.Load(model_path)

print(f"Vocab size: {sp.GetPieceSize()}")

print("\nPieces 100-200:")
for i in range(100, 200):
    print(f"{i:3}: {sp.IdToPiece(i)}")
