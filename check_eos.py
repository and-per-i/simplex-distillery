import sentencepiece as spm
import os

model_path = "tokenizer/weights/geometry.757.model"
if not os.path.exists(model_path):
    model_path = "pt_ckpt/vocab.model"

sp = spm.SentencePieceProcessor()
sp.Load(model_path)

print(f"';' ID: {sp.PieceToId(';')}")
print(f"EOS ID: {sp.eos_id()}")
