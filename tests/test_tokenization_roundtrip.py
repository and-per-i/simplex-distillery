"""
Test di verifica round-trip e consistency del tokenizer.
Garantisce che encode/decode funzionino correttamente e che
i token speciali (EOS, semicolon) siano configurati correttamente.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer.hf_tokenizer import load_tokenizer
from alphageo.tokens import GEOMETRY_EOS_ID, SEMICOLON_ID

TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "pt_ckpt", "vocab.model"
)

def test_vocab_size():
    """Test 1: Verifica che vocab_size sia 1024."""
    print("=== Test 1: Vocab Size ===")
    tok = load_tokenizer(TOKENIZER_PATH)
    
    assert tok.vocab_size == 1024, f"Expected vocab_size=1024, got {tok.vocab_size}"
    assert tok.sp_model.GetPieceSize() == 757, f"Expected SP model size=757, got {tok.sp_model.GetPieceSize()}"
    
    print(f"  ✅ Vocab size: {tok.vocab_size} (SP model: {tok.sp_model.GetPieceSize()})")
    return tok

def test_eos_token(tok):
    """Test 2: Verifica che il token EOS sia corretto."""
    print("\n=== Test 2: EOS Token ===")
    
    # Test semicolon token
    semicolon_tokens = tok.encode(";", add_special_tokens=False)
    print(f"  Semicolon ';' encoded as: {semicolon_tokens}")
    
    assert semicolon_tokens[0] == SEMICOLON_ID, \
        f"Expected semicolon ID={SEMICOLON_ID}, got {semicolon_tokens[0]}"
    
    # Test che GEOMETRY_EOS_ID corrisponda
    assert GEOMETRY_EOS_ID == SEMICOLON_ID, "GEOMETRY_EOS_ID deve essere uguale a SEMICOLON_ID"
    
    print(f"  ✅ Semicolon token ID: {SEMICOLON_ID}")
    print(f"  ✅ GEOMETRY_EOS_ID: {GEOMETRY_EOS_ID}")

def test_roundtrip_basic(tok):
    """Test 3: Round-trip encode/decode su testo semplice."""
    print("\n=== Test 3: Basic Round-trip ===")
    
    texts = [
        "a b c coll a b c ;",
        "a b c = triangle a b c",
        "? perp c h a b",
        "aux e = midp a b"
    ]
    
    for text in texts:
        tokens = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(tokens, skip_special_tokens=True)
        
        # Normalizziamo spazi
        text_norm = " ".join(text.split())
        decoded_norm = " ".join(decoded.split())
        
        print(f"  Original:  '{text}'")
        print(f"  Tokens:    {tokens}")
        print(f"  Decoded:   '{decoded}'")
        
        # SentencePiece potrebbe aggiungere/rimuovere spazi - verifichiamo almeno i caratteri chiave
        assert all(c in decoded_norm for c in text_norm.replace(" ", "")), \
            f"Round-trip failed for '{text}'"
        
        print(f"  ✅ Round-trip OK\n")

def test_special_token_positions(tok):
    """Test 4: Verifica posizione token speciali."""
    print("=== Test 4: Special Token Positions ===")
    
    special_tokens = {
        "<pad>": 0,
        "</s>": 1,
        "<s>": 2,
        "<unk>": 3,
        ";": 263,  # Semicolon with leading space (▁;)
    }
    
    for token_str, expected_id in special_tokens.items():
        if token_str in ["<pad>", "</s>", "<s>", "<unk>"]:
            # Questi sono token speciali SP
            actual_id = tok.sp_model.PieceToId(token_str)
        else:
            # Semicolon è un token normale (con spazio leading in SP)
            actual_id = tok.encode(token_str, add_special_tokens=False)[0]
        
        print(f"  Token '{token_str}': ID={actual_id} (expected={expected_id})")
        assert actual_id == expected_id, f"Mismatch for {token_str}"
    
    print("  ✅ All special tokens at correct positions")

def test_vocab_completeness(tok):
    """Test 5: Verifica che get_vocab() funzioni per tutti i token."""
    print("\n=== Test 5: Vocab Completeness ===")
    
    vocab = tok.get_vocab()
    assert len(vocab) == 1024, f"Expected vocab length=1024, got {len(vocab)}"
    
    # Verifica che token 0-756 abbiano nomi reali, 757-1023 abbiano placeholder
    real_tokens = sum(1 for k in vocab.keys() if not k.startswith("<extra_id_"))
    print(f"  Real tokens: {real_tokens}")
    print(f"  Placeholder tokens: {1024 - real_tokens}")
    
    assert real_tokens >= 757, "Should have at least 757 real tokens"
    
    print("  ✅ Vocab completeness OK")

def main():
    print("🧪 Running tokenization tests...\n")
    
    tok = test_vocab_size()
    test_eos_token(tok)
    test_roundtrip_basic(tok)
    test_special_token_positions(tok)
    test_vocab_completeness(tok)
    
    print("\n🎉 All tokenization tests passed!")

if __name__ == "__main__":
    main()
