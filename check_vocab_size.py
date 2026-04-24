import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("pt_ckpt/vocab.model")
print(f"Vocab size of pt_ckpt/vocab.model: {sp.GetPieceSize()}")
