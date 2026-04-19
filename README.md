# AlphaGeometry Tokenizer

Questo pacchetto contiene il tokenizer SentencePiece estratto dal progetto AlphaGeometry.

## Contenuto
- `weights/geometry.757.model`: Il modello del tokenizer.
- `weights/geometry.757.vocab`: Il vocabolario in formato testo.
- `sp_server.py`: Lo script che gestisce l'interazione con la libreria SentencePiece.
- `tokenizer_client.py`: Una classe Python pronta all'uso per codificare e decodificare testo.

## Requisiti
È necessario avere la libreria `sentencepiece` installata nel proprio ambiente Python:
```bash
pip install sentencepiece
```

## Esempio di Utilizzo
Puoi usare il tokenizer direttamente importando `AlphaGeometryTokenizer` da `tokenizer_client.py`:

```python
from tokenizer_client import AlphaGeometryTokenizer

tokenizer = AlphaGeometryTokenizer('weights/geometry.757.model')

# Encode
ids = tokenizer.encode("a b c coll a b c ;")
print(ids)

# Decode
text = tokenizer.decode(ids)
print(text)
```
# simplex-distillery
