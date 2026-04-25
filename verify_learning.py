
import os
import torch
import torch.nn as nn
from pathlib import Path
from models.student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer
import glob

def verify_latest_checkpoint():
    print("🔍 Verifica apprendimento - Davide 2-Simplex (RTX 5090)")
    print("======================================================")

    # 1. Trova l'ultimo checkpoint
    checkpoint_dirs = sorted(glob.glob("runs/distill_5090/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))
    if not checkpoint_dirs:
        print("❌ Nessun checkpoint trovato in runs/distill_5090/")
        return
    
    latest_ckpt = checkpoint_dirs[-1]
    print(f"📂 Caricamento ultimo checkpoint: {latest_ckpt}")

    # 2. Configurazione
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1024
    dim_hidden = 384
    num_layers = 12
    
    # Inizializza modello
    model = StudentModelProgressive(
        vocab_size=vocab_size,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        simplicial_layers=[3, 7, 11]
    )
    
    # Carica pesi (HuggingFace Trainer salva in pytorch_model.bin o model.safetensors)
    weights_path = Path(latest_ckpt) / "pytorch_model.bin"
    if not weights_path.exists():
        # Prova safe tensors
        weights_path = Path(latest_ckpt) / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            print(f"❌ Pesi non trovati in {latest_ckpt}")
            return
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
    
    model.load_state_dict(state_dict)
    
    # Attiva bypass (siamo a epoca 2.3+, quindi layer 2 e 10 sono bypassed)
    # Layer indices 0-indexed: 1 e 9
    model.layers[1].is_bypassed = True
    model.layers[9].is_bypassed = True
    
    model.to(device)
    model.eval()
    print(f"✅ Modello caricato e configurato (Layer 2 e 10 bypassed)")

    # 3. Tokenizer
    tokenizer = load_tokenizer("pt_ckpt/vocab.model", vocab_size=1024)
    
    # 4. Test di generazione
    # Prompt: Triangolo con ortocentro
    prompt = "a b c = triangle a b c; h = orthocenter h a b c; aux h a b c ;"
    print(f"\n📝 Prompt di test: {prompt}")
    
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    
    print("\n🔮 Generazione (Top-5 per ogni step):")
    generated = []
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[0, -1, :]
            
            # Top 5
            top5 = torch.topk(next_token_logits, k=5)
            next_id = top5.indices[0].item()
            token_str = tokenizer.convert_ids_to_tokens(next_id)
            
            # Stampa i primi 3 per vedere la "certezza"
            top_strs = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top5.indices[:3]]
            print(f"Step {len(generated)+1:2d}: {top_strs[0]:15s} (Alternativi: {top_strs[1]}, {top_strs[2]})")
            
            generated.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            
            if token_str == "</s>":
                break

    full_gen = tokenizer.decode(generated)
    print(f"\n✨ Sequenza generata: {full_gen}")
    print("\n======================================================")
    print("💡 ANALISI: Se vedi predicati geometrici coerenti (perp, para, cong) ")
    print("   o nomi di punti esistenti nel prompt (a, b, c, h), il modello")
    print("   sta imparando la struttura logica, non solo memorizzando.")

if __name__ == "__main__":
    verify_latest_checkpoint()
