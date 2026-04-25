"""
Verifica apprendimento del modello Davide 2-Simplex durante il training.

Usa il formato REALE del dataset di training:
  <problem> a : ; b : ; ... ? goal </problem>
  <numerical_check> ... ; </numerical_check>
  <proof> ... ; (il modello deve completare questo)
"""

import os
import sys
import glob
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer


def load_latest_checkpoint(device):
    checkpoint_dirs = sorted(
        glob.glob("runs/distill_5090/checkpoint-*"),
        key=lambda x: int(x.split("-")[-1])
    )
    if not checkpoint_dirs:
        raise FileNotFoundError("❌ Nessun checkpoint trovato in runs/distill_5090/")

    latest_ckpt = checkpoint_dirs[-1]
    print(f"📂 Caricamento: {latest_ckpt}")

    model = StudentModelProgressive(
        vocab_size=1024,
        dim_hidden=384,
        num_layers=12,
        simplicial_layers=[3, 7, 11]
    )

    weights_path = Path(latest_ckpt) / "pytorch_model.bin"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        weights_path = Path(latest_ckpt) / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            raise FileNotFoundError(f"Pesi non trovati in {latest_ckpt}")

    model.load_state_dict(state_dict)

    # Attiva il pruning coerente con l'epoca corrente (siamo a ~2.3)
    model.layers[1].is_bypassed = True  # Layer 2
    model.layers[9].is_bypassed = True  # Layer 10

    model.to(device)
    model.eval()
    print("✅ Modello caricato (Layer 2 e 10 bypassed)")
    return model


def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, device="cuda"):
    """Genera tokens con temperature sampling."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_tensor)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            token_str = tokenizer.convert_ids_to_tokens(next_id)
            generated.append(next_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_id]], device=device)], dim=1)

            if token_str in ("</s>", "<pad>"):
                break

    return tokenizer.decode(generated)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔍 Verifica Apprendimento - Davide 2-Simplex")
    print(f"   Device: {device}")
    print("=" * 60)

    tokenizer = load_tokenizer("pt_ckpt/vocab.model", vocab_size=1024)
    model = load_latest_checkpoint(device)

    # ===== PROMPT NEL FORMATO REALE DEL DATASET =====
    test_cases = [
        {
            "name": "Triangolo isoscele (training set)",
            # Questo è il formato esatto usato nel training
            "prompt": "<problem> a : ; b : ; c : ; d : ; e : cong b d b e [000] cong a d a e [001] ? simtrir a b d a b e </problem>\n<numerical_check> sameclock a b d a e b [002] ; </numerical_check>\n<proof>",
            "expected_contains": ["eqratio", "simtrir", ";"],  # Token attesi nella prova
        },
        {
            "name": "Problema perpendicolare",
            "prompt": "<problem> a : ; b : ; c : ; d : perp a b c d [000] ? cong a c b d </problem>\n<numerical_check> sameside a c b d [001] ; </numerical_check>\n<proof>",
            "expected_contains": ["cong", ";"],
        },
        {
            "name": "Problema angoli",
            "prompt": "<problem> a : ; b : ; c : ; eqangle a b b c b c c a [000] ? simtri a b c b c a </problem>\n<numerical_check> sameclock a b c b c a [001] ; </numerical_check>\n<proof>",
            "expected_contains": ["simtri", ";"],
        },
    ]

    all_ok = True
    for tc in test_cases:
        print(f"\n📝 Test: {tc['name']}")
        print(f"   Prompt: ...{tc['prompt'][-60:]}")

        # Prova con 3 temperature diverse
        for temp in [0.6, 0.8, 1.0]:
            output = generate(model, tokenizer, tc["prompt"],
                              max_new_tokens=60, temperature=temp, device=device)
            found = [kw for kw in tc["expected_contains"] if kw in output]

            status = "✅" if found else "⚠️"
            print(f"   T={temp}: {status} '{output[:100].strip()}'")
            if found:
                print(f"            → Trovati predicati attesi: {found}")
                break
        else:
            all_ok = False

    # ===== ANALISI ENTROPÍA =====
    print("\n📊 Analisi entropia delle predizioni:")
    prompt = "<problem> a : ; b : ; c : ; d : ; e : cong b d b e [000] ? simtrir a b d a b e </problem>\n<proof>"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long, device=device
    )
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
        top5_ids = torch.topk(last_logits, 5).indices.tolist()
        top5_tokens = [tokenizer.convert_ids_to_tokens(i) for i in top5_ids]

    print(f"   Entropia: {entropy:.2f} (alta=incerto, bassa=fiducioso)")
    print(f"   Top-5 token: {top5_tokens}")

    if entropy < 2.0:
        print("   💎 Modello MOLTO fiducioso (probabilmente sta imparando bene)")
    elif entropy < 4.0:
        print("   ✅ Modello ragionevolmente fiducioso")
    else:
        print("   ⚠️  Modello incerto — potrebbe aver bisogno di più training")

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ RISULTATO: Il modello sta imparando la struttura delle prove!")
    else:
        print("⚠️  RISULTATO: Alcune prove non contengono predicati attesi.")
        print("   → Normale a metà training. Riprova all'epoca 4.")


if __name__ == "__main__":
    main()
