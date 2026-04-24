import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("tokenizer/weights/geometry.757.model")
for i in range(260, 300):
    try:
        print(f"{i}: {sp.IdToPiece(i)}")
    except:
        pass
