import os
import sys
import numpy as np
import subprocess
import gin
import jax

# Set AlphaGeometry Path
AG_PATH = "/Users/andrea/Documents/alphageometry"
MELIAD_PATH = os.path.join(AG_PATH, "meliad_lib/meliad")
AG_VENV_BIN = os.path.join(AG_PATH, "venv/bin")

# PRIORITIZE AG_PATH and MELIAD_PATH
sys.path.insert(0, AG_PATH)
sys.path.insert(0, MELIAD_PATH)

# Add AG Venv Bin to PATH
os.environ["PATH"] = AG_VENV_BIN + os.pathsep + os.environ["PATH"]

# Force JAX to use CPU for this local test
os.environ["JAX_PLATFORM_NAME"] = "cpu"

try:
    import lm_inference
    import flax
    print("✅ lm_inference e flax importati correttamente")
except ImportError as e:
    print(f"❌ Errore importazione: {e}")
    sys.exit(1)

def test_logits():
    print("🧠 Configurazione GIN...")
    try:
        gin.add_config_file_search_path(os.path.join(MELIAD_PATH, "transformer/configs"))
        gin.add_config_file_search_path(AG_PATH)
        
        gin_files = [
            "base_htrans.gin",
            "size/medium_150M.gin",
            "options/positions_t5.gin",
            "options/lr_cosine_decay.gin",
            "options/seq_1024_nocache.gin",
            "geometry_150M_generate.gin",
        ]
        gin_bindings = [
            "DecoderOnlyLanguageModelGenerate.output_token_losses=True",
            "TransformerTaskConfig.batch_size=1",
            "TransformerTaskConfig.sequence_length=128",
            "Trainer.restore_state_variables=False",
        ]
        gin.parse_config_files_and_bindings(gin_files, gin_bindings)
        print("✅ GIN configurato.")
    except Exception as e:
        print(f"❌ Errore GIN: {e}")
        return

    print("🧠 Inizializzazione AlphaGeometry...")
    try:
        vocab_p = os.path.join(AG_PATH, "weights/geometry.757.model")
        checkpoint_p = os.path.join(AG_PATH, "weights/checkpoint_10999999")
        
        cwd = os.getcwd()
        os.chdir(AG_PATH)
        
        teacher = lm_inference.LanguageModelInference(
            vocab_path=vocab_p,
            load_dir=checkpoint_p
        )
        print("✅ Teacher inizializzato.")
        
        # Test string
        test_str = "<problem> a : ; b : ; c : ; ? cong a b b a </problem>"
        tokens = teacher.vocab.encode(test_str)
        tokens_np = np.array([tokens])
        length = tokens_np.shape[1]
        
        print(f"🚀 Estrazione logit chiamando direttamente bound_model.decoder...")
        
        # Input preparato
        batch_size = 1
        targets = jax.numpy.array(tokens_np)
        start_of_sequence = jax.numpy.array([True] * batch_size)
        
        # Bind delle variabili
        variables = {"params": teacher.tstate.optimizer.target, **teacher.tstate.state}
        bound_model = teacher.imodel.bind(variables)
        
        # Chiamata diretta al decoder (metodo di DecoderOnlyLanguageModelGenerate)
        # Nota: DecoderOnlyLanguageModelGenerate ha un attributo 'decoder' che è lo stack
        # In models.py: (logits, dstate, _) = self.decoder(...)
        logits, dstate, _ = bound_model.decoder(
            input_tokens=targets,
            target_tokens=None,
            start_of_sequence=start_of_sequence,
        )
        
        print("\n--- RISULTATO LOGIT ---")
        print(f"Logits shape: {logits.shape}") # (B, S, Vocab)
        
        if logits is not None:
            logits_np = np.array(logits)
            # Predizione per l'ultima posizione
            last_pos_logits = logits_np[0, length-1, :]
            top_5_idx = np.argsort(last_pos_logits)[-5:][::-1]
            
            print(f"\nTop 5 predizioni per il token dopo l'ultimo:")
            for idx in top_5_idx:
                token_str = teacher.vocab.decode([int(idx)])
                print(f"  ID {idx:3} ('{token_str}'): {last_pos_logits[idx]:.4f}")
            
            print("\n✅ Estrazione logit DIRETTA riuscita!")
        
        os.chdir(cwd)

    except Exception as e:
        print(f"❌ Errore durante l'estrazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logits()
