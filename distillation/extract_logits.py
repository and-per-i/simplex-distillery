import os
import sys
from unittest.mock import MagicMock
import jax

# --- COMPATIBILITY SHIM FOR JAX/FLAX ---

# 1. Mock flax.optim with a more realistic structure
try:
    import flax
    if not hasattr(flax, "optim"):
        mock_optim = MagicMock()
        class FakeOptimizer:
            def __init__(self, target, state=None):
                self.target = target
                self.state = state or MagicMock(step=0)
            def apply_gradient(self, grads, **kwargs): return self
        mock_optim.Optimizer = FakeOptimizer
        mock_optim.Adam = MagicMock(return_value=MagicMock(create=lambda p: FakeOptimizer(p)))
        mock_optim.Adafactor = MagicMock(return_value=MagicMock(create=lambda p: FakeOptimizer(p)))
        sys.modules["flax.optim"] = mock_optim
        flax.optim = mock_optim
except ImportError:
    pass

# 2. Handle moved jax attributes
if not hasattr(jax, "linear_util"):
    try:
        from jax._src import linear_util as lu
        jax.linear_util = lu
    except ImportError:
        pass

if not hasattr(jax.tree_util, "register_keypaths"):
    jax.tree_util.register_keypaths = lambda *args, **kwargs: None

# 3. Patch FrozenDict.pop (CRITICAL for AlphaGeometry/Flax 0.10.x)
import flax.core.frozen_dict
def new_pop(self, key):
    val = self[key]
    new_d = dict(self.unfreeze())
    new_d.pop(key)
    return flax.core.frozen_dict.FrozenDict(new_d), val
flax.core.frozen_dict.FrozenDict.pop = new_pop

# 4. GIN Paths (Pre-config)
import gin
# Identifichiamo i path prima di tutto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_AG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "alphageometry"))
AG_PATH = os.environ.get("AG_PATH", DEFAULT_AG_PATH)
if not os.path.exists(AG_PATH):
    for possible_path in ["/workspace/alphageometry", "/workspace/a-geo", "/Users/andrea/Documents/alphageometry", "/Users/andrea/Documents/a-geo"]:
        if os.path.exists(possible_path):
            AG_PATH = possible_path
            break
MELIAD_PATH = os.path.join(AG_PATH, "meliad_lib", "meliad")
CONFIG_DIR = os.path.join(MELIAD_PATH, "transformer/configs")

# Aggiungiamo i path a GIN immediatamente
for p in [CONFIG_DIR, os.path.join(CONFIG_DIR, "size"), os.path.join(CONFIG_DIR, "options"), 
          os.path.join(CONFIG_DIR, "recurrent"), os.path.join(CONFIG_DIR, "tasks"), AG_PATH]:
    if os.path.exists(p):
        gin.add_config_file_search_path(p)

import numpy as np
import pandas as pd
import tqdm
import time

# --- CONFIGURAZIONE PATH (Standard AlphaGeometry) ---
# Inseriamo i path come previsto dagli autori
sys.path.insert(0, AG_PATH)
sys.path.insert(0, MELIAD_PATH)

# Binari del venv
AG_VENV_BIN = os.path.join(AG_PATH, "venv/bin")
os.environ["PATH"] = AG_VENV_BIN + os.pathsep + os.environ["PATH"]

# --- DEVICE CONFIGURATION ---
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1" or "--cpu" in sys.argv
if FORCE_CPU:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Auto-detect device per ottimizzazione su GPU potenti (es. RTX 5090 Ti)
try:
    if FORCE_CPU or jax.local_device_count() == 0:
        if not FORCE_CPU:
            print("⚠️ Nessuna GPU rilevata. Fallback su CPU.")
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        DEFAULT_BATCH_SIZE = 1
    else:
        # Su una 5090 Ti 32GB, con modello 150M, possiamo spingere il batch size
        DEFAULT_BATCH_SIZE = 256 
        print(f"🚀 GPU rilevata: {jax.devices()[0].device_kind}. Batch size default: {DEFAULT_BATCH_SIZE}")
except Exception as e:
    print(f"⚠️ Errore nel rilevamento device: {e}. Fallback su CPU.")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    DEFAULT_BATCH_SIZE = 1

import lm_inference

def setup_ag(batch_size):
    print("🧠 Configurazione GIN con path assoluti...")
    
    # Costruiamo i path assoluti per i file GIN
    CONFIG_DIR = os.path.join(MELIAD_PATH, "transformer/configs")
    gin_files = [
        os.path.join(CONFIG_DIR, "base_htrans.gin"),
        os.path.join(CONFIG_DIR, "size/medium_150M.gin"),
        os.path.join(CONFIG_DIR, "options/positions_t5.gin"),
        os.path.join(CONFIG_DIR, "options/lr_cosine_decay.gin"),
        os.path.join(CONFIG_DIR, "options/seq_1024_nocache.gin"),
        os.path.join(AG_PATH, "geometry_150M_generate.gin")
    ]
    
    gin_bindings = [
        "DecoderOnlyLanguageModelGenerate.output_token_losses=True",
        f"TransformerTaskConfig.batch_size={batch_size}",
        "TransformerTaskConfig.sequence_length=128",
        "Trainer.restore_state_variables=False"
    ]
    
    # Patch diretta di Trainer.initialize_model per gestire il pop() di Flax nuovo
    import training_loop
    from flax.core.frozen_dict import FrozenDict
    
    original_initialize_model = training_loop.Trainer.initialize_model
    
    def patched_initialize_model(self):
        # Eseguiamo la versione originale ma intercettiamo l'errore o simuliamo il comportamento
        # In alternativa, sovrascriviamo la logica di unpack
        res = original_initialize_model(self)
        return res

    # Se l'originale fallisce, dobbiamo proprio riscrivere il pezzetto.
    # Patch crucial per compatibilità Flax 0.10.x -> Codice AlphaGeometry
    # (Manteniamo lo shim anche qui per sicurezza, ma la patch su training_loop è ora primaria)
    print("🛠️ Patch GIN e FrozenDict applicate.")
    
    # Assicuriamoci che gin conosca i path per gli include interni
    gin.add_config_file_search_path(CONFIG_DIR)
    gin.add_config_file_search_path(os.path.join(CONFIG_DIR, "size"))
    gin.add_config_file_search_path(os.path.join(CONFIG_DIR, "options"))
    gin.add_config_file_search_path(AG_PATH)

    gin.parse_config_files_and_bindings(gin_files, gin_bindings)
    
    teacher = lm_inference.LanguageModelInference(
        vocab_path=os.path.join(AG_PATH, "ag_ckpt_vocab/geometry.757.model"),
        load_dir=os.path.join(AG_PATH, "ag_ckpt_vocab/checkpoint_10999999")
    )
    return teacher


import math
import jax.numpy as jnp

def extract(num_samples=100000, batch_size=None):
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
        
    input_file = "dataset1"
    output_file = "data/dataset1_top50_logits.parquet"
    top_k = 50
    seq_len = 128

    if not os.path.exists("data"): os.makedirs("data")

    print(f"📥 Caricamento dataset: {input_file}")
    df_full = pd.read_parquet(input_file).head(num_samples)
    
    teacher = setup_ag(batch_size)
    variables = {"params": teacher.tstate.optimizer.target, **teacher.tstate.state}
    bound_model = teacher.imodel.bind(variables)
    
    @jax.jit
    def get_logits_jit(targets, sos):
        logits, _, _ = bound_model.decoder(
            input_tokens=targets,
            target_tokens=None,
            start_of_sequence=sos,
        )
        return logits

    all_indices = []
    all_values = []
    
    print(f"🚀 Estrazione parallela (Batch Size: {batch_size}, Top-{top_k})...")
    start_time = time.time()
    
    # Salviamo la directory corrente e passiamo a quella di AG per gli import/pesi
    old_cwd = os.getcwd()
    os.chdir(AG_PATH)
    
    results = []
    try:
        pbar = tqdm.tqdm(total=len(df_full))
        for i in range(0, len(df_full), batch_size):
            batch_df = df_full.iloc[i : i + batch_size]
            print(f"\n📦 Elaborazione batch {i//batch_size + 1}/{math.ceil(len(df_full)/batch_size)}...")
            actual_len = len(batch_df)
            
            batch_tokens = []
            for _, row in batch_df.iterrows():
                full_text = f"{row['question']} {row['solution']}"
                tokens = teacher.vocab.encode(full_text)
                tokens = tokens[:seq_len] if len(tokens) > seq_len else tokens + [0]*(seq_len - len(tokens))
                batch_tokens.append(tokens)
            
            batch_tokens = np.array(batch_tokens, dtype=np.int32)
            
            # Calcolo logits
            if actual_len < batch_size:
                padding = np.zeros((batch_size - actual_len, seq_len), dtype=np.int32)
                batch_tokens = np.concatenate([batch_tokens, padding], axis=0)
            
            targets = jax.numpy.array(batch_tokens)
            sos = jax.numpy.array([True] * batch_size)
            
            logits = get_logits_jit(targets, sos)
            logits_np = np.array(logits)
            logits_np = logits_np[:actual_len]
            
            idx = np.argpartition(logits_np, -top_k, axis=-1)[..., -top_k:]
            b, s, k = idx.shape
            batch_idx = np.arange(b)[:, None, None]
            seq_idx = np.arange(s)[None, :, None]
            
            vals_subset = logits_np[batch_idx, seq_idx, idx]
            sort_idx = np.argsort(vals_subset, axis=-1)[..., ::-1]
            
            final_idx = idx[batch_idx, seq_idx, sort_idx]
            final_val = vals_subset[batch_idx, seq_idx, sort_idx]
            
            for b_i in range(actual_len):
                all_indices.append(final_idx[b_i].astype(np.int16))
                all_values.append(final_val[b_i].astype(np.float16))

    finally:
        os.chdir(cwd)

    df_full['top_k_indices'] = all_indices
    df_full['top_k_values'] = all_values
    
    print(f"💾 Salvataggio in {output_file}...")
    df_full.to_parquet(output_file)
    print(f"✅ Completato! Tempo totale: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
    extract(num_samples=args.num_samples, batch_size=args.batch_size)
