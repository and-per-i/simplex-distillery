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

    print("Generating proof (Greedy Search)...")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,   # Forza il modello a non ripetersi
            no_repeat_ngram_size=3,    # Impedisce loop di 3 o più token
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 5. Traduzione "Master"
    RULE_MAP = {
        "r00": "Perpendiculars give parallel", "r01": "Definition of cyclic", "r02": "Parallel from inclination",
        "r03": "Arc determines internal angles", "r04": "Congruent angles are in a cyclic", "r05": "Same arc same chord",
        "r06": "Base of half triangle", "r07": "Thales Theorem I", "r08": "Right triangles common angle I",
        "r09": "Sum of angles of a triangle", "r11": "Bisector theorem I", "r13": "Isosceles triangle equal angles",
        "r14": "Equal base angles imply isosceles", "r15": "Arc determines inscribed angles (tangent)",
        "r16": "Same arc giving tangent", "r19": "Hypotenuse is diameter", "r20": "Diameter is hypotenuse", 
        "r25": "Diagonals of parallelogram I", "r27": "Thales theorem II", "r28": "Overlapping parallels", 
        "r29": "Midpoint Theorem", "r34": "AA Similarity (Direct)", "r42": "Thales theorem IV", 
        "r43": "Orthocenter theorem", "r51": "Midpoint splits in two", "r54": "Definition of midpoint", 
        "r57": "Pythagoras theorem", "r72": "Disassembling a circle", "r73": "Definition of circle"
    }
    
    import re
    # Estraggo i punti dal prompt (es. a, b, c...)
    initial_points = re.findall(r'\b([a-z])\b', prompt.split('?')[0])
    POINT_MAP = {}
    # Mappiamo i primi N indici ai punti del prompt
    for i, name in enumerate(initial_points):
        POINT_MAP[str(i)] = name.upper()
        POINT_MAP[f"{i:02d}"] = name.upper()
    
    # Punti ausiliari extra (fino a 50)
    all_alphabet = "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
    for i in range(len(initial_points), 50):
        name = all_alphabet[i - len(initial_points)] if (i - len(initial_points)) < len(all_alphabet) else f"P{i}"
        POINT_MAP[str(i)] = name.upper()
        POINT_MAP[f"{i:02d}"] = name.upper()

    solution_raw = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    raw_tokens = re.findall(r'[a-z0-9]+|;|:', solution_raw.lower())
    
    translated_parts = []
    step_count = 1
    
    for t in raw_tokens:
        t_clean = re.sub(r'[^a-z0-9;:]', '', t)
        if t_clean in RULE_MAP:
            translated_parts.append(f"\nSTEP {step_count}: Using {RULE_MAP[t_clean]} on")
            step_count += 1
        elif t_clean in POINT_MAP:
            translated_parts.append(POINT_MAP[t_clean])
        elif t_clean == ";":
            translated_parts.append(";")
        elif t_clean == ":":
            translated_parts.append(":")
        elif t_clean.startswith("r") and len(t_clean) > 1:
            translated_parts.append(f"\nSTEP {step_count}: Using Rule {t_clean.upper()} on")
            step_count += 1
        else:
            translated_parts.append(t_clean.upper())
            
    final_proof = " ".join(translated_parts).replace(" ;", ";").replace(" :", ":")

    print("\n================ TRANSLATED PROOF ================")
    print(final_proof)
    print("==================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Test for Simplex Distillery")
    parser.add_argument("--model_path", type=str, default="./student_v1", help="Path to the trained model directory")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/weights/geometry.757.model", help="Path to the tokenizer model file")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation. If None, enters interactive mode.")
    args = parser.parse_args()

    MODEL_DIR = args.model_path
    TOKENIZER_FILE = args.tokenizer_path

    if args.prompt:
        prompt = args.prompt
    else:
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
        
        choice = input("\nInserisci il numero o scrivi un nuovo prompt (invio per il primo): ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(theorems):
            prompt = theorems[int(choice)-1]
        else:
            prompt = choice if choice else theorems[0]

    run_pro_inference(prompt, MODEL_DIR, TOKENIZER_FILE)
