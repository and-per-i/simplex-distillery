import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("../Newclid_Transformer/alphageometry_model/geometry.757.model")
for i in range(260, 320):
    try:
        print(f"{i}: {sp.IdToPiece(i)}")
    except:
        pass
