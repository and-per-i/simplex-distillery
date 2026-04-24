"""
Test per verificare che il modello AlphaGeometry convertito
accetti input in formato Newclid (aux + predicati strutturati).
"""

import os
import sys
import torch
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer.hf_tokenizer import load_tokenizer
from alphageo.alphageometry import get_lm
from newclid.llm_input import problem_to_llm_input
from newclid.jgex.problem_builder import JGEXProblemBuilder
from newclid.jgex.formulation import alphabetize

logging.basicConfig(level=logging.INFO)

def test_newclid_prompt_format():
    """
    Test 1: Verifica che il modello accetti prompt Newclid.
    """
    print("=== TEST FORMATO PROMPT NEWCLID ===\n")
    
    # 1. Carica modello e tokenizer
    ckpt_path = Path("./pt_ckpt")
    vocab_path = "./pt_ckpt/vocab.model"
    
    # Determina device disponibile
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Caricamento modello da {ckpt_path} su device={device}...")
    model = get_lm(ckpt_path, device)
    tokenizer = load_tokenizer(vocab_path, vocab_size=1024)
    print(f"✅ Modello caricato. Vocab size: {tokenizer.vocab_size}\n")
    
    # 2. Genera prompt in formato Newclid
    problem_txt = "a b c = triangle a b c; h : orthocenter h a b c; ? perp c h a b"
    
    jb = JGEXProblemBuilder(rng=42)
    jb.with_problem_from_txt(problem_txt, problem_name="test_ortho")
    jb.jgex_problem, _ = alphabetize(jb.jgex_problem)
    
    newclid_prompt = problem_to_llm_input(jb.jgex_problem, aux_tag="aux")
    print(f"Prompt Newclid generato:\n{newclid_prompt}\n")
    
    # 3. Tokenizza
    tokens = tokenizer.encode(newclid_prompt, add_special_tokens=False)
    print(f"Token IDs ({len(tokens)} tokens): {tokens[:20]}...\n")
    
    # 4. Forward pass
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model(input_ids)
    
    logits = output if isinstance(output, torch.Tensor) else output.logits
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: (1, {len(tokens)}, 1024)")
    
    assert logits.shape == (1, len(tokens), 1024), "Shape mismatch!"
    print("✅ Forward pass successful!\n")
    
    # 5. Genera next token
    next_token_logits = logits[0, -1, :]
    top5_tokens = torch.topk(next_token_logits, k=5)
    
    print("Top 5 token predetti:")
    for i, (score, idx) in enumerate(zip(top5_tokens.values, top5_tokens.indices)):
        token_str = tokenizer.convert_ids_to_tokens(int(idx))
        print(f"  {i+1}. ID={idx} '{token_str}' (score={score:.2f})")
    
    print("\n✅ Test completato: il modello accetta formato Newclid!")
    
    return True

def test_alphagemotry_vs_newclid_prompts():
    """
    Test 2: Confronta la distribuzione di probabilità per prompt
    in formato AG originale vs Newclid (se possibile ricostruire formato AG).
    """
    # TODO: implementare se necessario dopo Test 1
    pass

if __name__ == "__main__":
    try:
        test_newclid_prompt_format()
    except Exception as e:
        print(f"\n❌ Test fallito: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
