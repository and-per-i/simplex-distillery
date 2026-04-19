import torch
from models.student_model import StudentForCausalLM
from models.student_config import StudentConfig
from tokenizer.hf_tokenizer import load_tokenizer
import os

def run_pro_inference(prompt, model_path, tokenizer_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = load_tokenizer(tokenizer_path)
    
    print(f"\n--- Loading Model on {device} ---")
    config = StudentConfig.from_pretrained(model_path)
    config.use_triton = False 
    
    model = StudentForCausalLM.from_pretrained(
        model_path, 
        config=config,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()

    print(f"\nPROMPT: {prompt}")
    inputs = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt").items()}

    print("Generating proof (Beam Search)...")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=128,
            num_beams=5,             # Esplora 5 strade diverse
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    solution = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("\n================ SOLUTION ================")
    print(solution)
    print("==========================================\n")

if __name__ == "__main__":
    MODEL_DIR = "./student_v1"
    TOKENIZER_FILE = "./tokenizer/weights/geometry.757.model"

    # Tre test di "livello superiore":
    theorems = [
        # 1. Teorema di Talete / Proporzionalità
        "a b c = triangle a b c; d = midpoint a b; e = midpoint a c; ? parallel d e b c",
        
        # 2. Angoli alla circonferenza (Classico AlphaGeometry)
        "a b c d = circle a b c d; ? cong angle a c b angle a d b",
        
        # 3. Ortocentro
        "a b c = triangle a b c; h1 = altitude a b c; h2 = altitude b a c; h = intersection h1 h2; ? perpendicular c h a b"
    ]

    print("Scegli un test:")
    for i, t in enumerate(theorems): print(f"{i+1}. {t}")
    
    choice = input("\nInserisci il numero o scrivi un nuovo prompt: ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(theorems):
        prompt = theorems[int(choice)-1]
    else:
        prompt = choice if choice else theorems[0]

    run_pro_inference(prompt, MODEL_DIR, TOKENIZER_FILE)
