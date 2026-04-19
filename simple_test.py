import torch
import torch.nn.functional as F
from models.student_model import StudentForCausalLM
from models.student_config import StudentConfig
from tokenizer.hf_tokenizer import load_tokenizer
import os

def test_logic(model, tokenizer, prompt, device):
    print(f"\nPROMPT: {prompt}")
    inputs = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :] # Prendi l'ultimo token
        
        # Vediamo le 5 opzioni più probabili per il prossimo passo
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, 5)
        
        print("Cosa sta pensando il modello (Top 5 probabilità):")
        for i in range(5):
            token = tokenizer.decode([top_ids[0][i].item()])
            # Se il token è vuoto o strano, mostriamo l'ID
            token_display = token if token.strip() else f"ID:{top_ids[0][i].item()}"
            print(f"  {i+1}. {token_display} ({top_probs[0][i].item():.2%})")

        # Generazione breve
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        print(f"Generazione: {tokenizer.decode(output_tokens[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    MODEL_DIR = "./student_v1"
    TOKENIZER_FILE = "./tokenizer/weights/geometry.757.model"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = load_tokenizer(TOKENIZER_FILE)
    config = StudentConfig.from_pretrained(MODEL_DIR)
    config.use_triton = False
    model = StudentForCausalLM.from_pretrained(MODEL_DIR, config=config).to(device)
    model.eval()

    # Problemi "Baby":
    simple_prompts = [
        "a b = line a b;",         # Dovrebbe continuare con un'altra definizione o un punto
        "a b c = triangle a b c;", # Dovrebbe definire qualcosa sul triangolo
        "p = point;"               # Estremamente semplice
    ]

    for p in simple_prompts:
        test_logic(model, tokenizer, p, device)
