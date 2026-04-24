"""
FASE 4: ESTRAZIONE FISICA DEL MODELLO FINALE
=============================================

Script offline per estrarre il modello 6-layer finale dai pesi del training.

Processo:
1. Carica checkpoint finale del training (12 layer, alcuni bypassed)
2. Rimuove fisicamente i layer bypassed dal state_dict
3. Rinumera i layer rimasti in ordine sequenziale
4. Salva modello compatto finale: Davide_2Simplex_40M_Finale.pt

Layer rimossi (indices 0-based):
- [1, 2, 4, 6, 8, 9] → Layer 2, 3, 5, 7, 9, 10

Layer mantenuti (3 standard + 3 simpliciali):
- Layer 1 (idx 0) - Standard
- Layer 4 (idx 3) - Simpliciale
- Layer 6 (idx 5) - Standard
- Layer 8 (idx 7) - Simpliciale
- Layer 11 (idx 10) - Standard
- Layer 12 (idx 11) - Simpliciale

Usage:
    python scripts/fase4_estrazione_fisica.py --checkpoint runs/distill_final/pytorch_model.bin --output Davide_2Simplex_40M.pt
"""

import torch
import argparse
from pathlib import Path
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def estrai_modello_finale(checkpoint_path: Path, output_path: Path,
                          layers_to_remove: list = [1, 2, 4, 6, 8, 9]):
    """
    FUNZIONE PRINCIPALE: Estrae modello 6-layer finale dal checkpoint 12-layer.
    
    Args:
        checkpoint_path: Path al checkpoint del training completo
        output_path: Path dove salvare il modello finale compatto
        layers_to_remove: Indici (0-based) dei layer da rimuovere fisicamente
    """
    logger.info("=" * 70)
    logger.info("FASE 4: ESTRAZIONE FISICA - Compattazione Finale")
    logger.info("=" * 70)
    
    # 1. Carica checkpoint
    logger.info(f"\n📦 Caricamento checkpoint da: {checkpoint_path}")
    
    # Se è una directory, punta al file pytorch_model.bin
    actual_checkpoint_path = checkpoint_path / "pytorch_model.bin" if checkpoint_path.is_dir() else checkpoint_path
    
    if not actual_checkpoint_path.exists():
        # Prova a cercare altri nomi comuni se pytorch_model.bin non esiste
        if checkpoint_path.is_dir():
            for alt_name in ["model.safetensors", "model.pt", "checkpoint.pt"]:
                if (checkpoint_path / alt_name).exists():
                    actual_checkpoint_path = checkpoint_path / alt_name
                    break
    
    if not actual_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {actual_checkpoint_path}")
        
    state_dict = torch.load(actual_checkpoint_path, map_location='cpu')
    
    # Se è un checkpoint HF completo, estrai solo il model state
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    logger.info(f"✅ Checkpoint caricato: {len(state_dict)} chiavi")
    
    # 2. Analizza struttura layer
    total_layers = 12
    layers_to_keep = [i for i in range(total_layers) if i not in layers_to_remove]
    
    logger.info(f"\n📊 Piano di Estrazione:")
    logger.info(f"  - Layer originali: {total_layers}")
    logger.info(f"  - Layer da rimuovere: {len(layers_to_remove)}")
    logger.info(f"  - Layer finali: {len(layers_to_keep)}")
    logger.info(f"\n🗑️  Layer rimossi: {[i+1 for i in layers_to_remove]}")
    logger.info(f"✅ Layer mantenuti: {[i+1 for i in layers_to_keep]}")
    
    # Identifica layer simpliciali
    simplicial_layers_original = [3, 7, 11]  # 0-indexed
    simplicial_kept = [i for i in simplicial_layers_original if i in layers_to_keep]
    logger.info(f"🔷 Layer Simpliciali finali: {[i+1 for i in simplicial_kept]}")
    
    # 3. Costruisci nuovo state dict
    logger.info(f"\n🔨 Ricostruzione State Dict...")
    new_state_dict = OrderedDict()
    
    # Mappa vecchi indici → nuovi indici
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(layers_to_keep)}
    
    logger.info(f"\n📋 Mappatura Layer:")
    for old_idx, new_idx in old_to_new.items():
        layer_type = "🔷 SIMPLICIALE" if old_idx in simplicial_layers_original else "📘 Standard"
        logger.info(f"  Layer {old_idx+1:2d} (old) → Layer {new_idx+1} (new) {layer_type}")
    
    # Itera su tutte le chiavi del checkpoint
    keys_copied = 0
    keys_skipped = 0
    
    for key, value in state_dict.items():
        # Individua se la chiave appartiene a un layer
        layer_idx = None
        
        # Pattern comuni: "layers.{i}.*", "transformer{i}.*", "blocks.{i}.*"
        if 'layers.' in key:
            try:
                parts = key.split('.')
                idx_pos = parts.index('layers') + 1
                layer_idx = int(parts[idx_pos])
            except (ValueError, IndexError):
                pass
        
        # Se la chiave non è di un layer specifico (embedding, output head, etc.)
        if layer_idx is None:
            new_state_dict[key] = value
            keys_copied += 1
            continue
        
        # Se il layer deve essere rimosso, skippa
        if layer_idx in layers_to_remove:
            keys_skipped += 1
            continue
        
        # Rinomina con il nuovo indice
        new_idx = old_to_new[layer_idx]
        new_key = key.replace(f'layers.{layer_idx}', f'layers.{new_idx}')
        new_state_dict[new_key] = value
        keys_copied += 1
    
    logger.info(f"\n📊 Statistiche Compattazione:")
    logger.info(f"  - Chiavi copiate: {keys_copied}")
    logger.info(f"  - Chiavi rimosse: {keys_skipped}")
    logger.info(f"  - Chiavi finali: {len(new_state_dict)}")
    
    # 4. Aggiorna metadata
    if 'meta' in new_state_dict:
        meta = new_state_dict['meta']
        if isinstance(meta, dict):
            meta['num_layers'] = len(layers_to_keep)
            meta['original_layers'] = total_layers
            meta['pruned_layers'] = layers_to_remove
            # Aggiorna indici simpliciali
            new_simplicial = [old_to_new[i] for i in simplicial_kept]
            meta['simplicial_layers'] = new_simplicial
            meta['phase'] = 'FASE_4_EXTRACTED'
    
    # 5. Calcola statistiche finali
    total_params = sum(p.numel() for p in new_state_dict.values() if isinstance(p, torch.Tensor))
    
    # 6. Salva modello finale
    logger.info(f"\n💾 Salvataggio modello finale...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output_path)
    
    size_mb = output_path.stat().st_size / (1024**2)
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 ESTRAZIONE COMPLETATA!")
    logger.info("=" * 70)
    logger.info(f"📦 File finale: {output_path}")
    logger.info(f"💾 Dimensione: {size_mb:.2f} MB")
    logger.info(f"🔢 Parametri totali: ~{total_params/1e6:.1f}M")
    logger.info(f"📐 Architettura finale: {len(layers_to_keep)} layers")
    logger.info(f"   - Standard: {len(layers_to_keep) - len(simplicial_kept)}")
    logger.info(f"   - Simpliciali: {len(simplicial_kept)}")
    logger.info("=" * 70)
    logger.info("\n✅ Modello pronto per deployment!")
    logger.info("   Usa questo checkpoint per inferenza ottimizzata.")
    logger.info("=" * 70 + "\n")


def verifica_integrità(model_path: Path):
    """
    Verifica l'integrità del modello estratto.
    
    Args:
        model_path: Path al modello da verificare
    """
    logger.info("\n🔍 VERIFICA INTEGRITÀ MODELLO")
    logger.info("-" * 50)
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Conta layer
    layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
    unique_layers = set()
    for key in layer_keys:
        try:
            parts = key.split('.')
            idx = int(parts[parts.index('layers') + 1])
            unique_layers.add(idx)
        except:
            pass
    
    logger.info(f"✅ Layer trovati: {sorted(unique_layers)}")
    logger.info(f"✅ Numero layer: {len(unique_layers)}")
    
    # Verifica presenza componenti essenziali
    has_embedding = any('embedding' in k for k in state_dict.keys())
    has_lm_head = any('lm_head' in k or 'output' in k for k in state_dict.keys())
    has_attention = any('attention' in k for k in state_dict.keys())
    has_mlp = any('mlp' in k for k in state_dict.keys())
    
    logger.info(f"✅ Embedding layer: {'Present' if has_embedding else 'MISSING'}")
    logger.info(f"✅ LM Head: {'Present' if has_lm_head else 'MISSING'}")
    logger.info(f"✅ Attention modules: {'Present' if has_attention else 'MISSING'}")
    logger.info(f"✅ MLP modules: {'Present' if has_mlp else 'MISSING'}")
    
    # Verifica K' per layer simpliciali
    k_prime_keys = [k for k in state_dict.keys() if 'k_prime' in k.lower() or 'keys_prime' in k.lower()]
    logger.info(f"✅ Matrici K' (Simplicial): {len(k_prime_keys)} trovate")
    
    # Metadata
    if 'meta' in state_dict:
        logger.info(f"\n📋 Metadata:")
        for k, v in state_dict['meta'].items():
            logger.info(f"  - {k}: {v}")
    
    logger.info("-" * 50)
    logger.info("✅ Verifica completata!\n")


def confronta_checkpoints(original_path: Path, extracted_path: Path):
    """
    Confronta checkpoint originale vs estratto per debug.
    
    Args:
        original_path: Path al checkpoint 12-layer originale
        extracted_path: Path al modello 6-layer estratto
    """
    logger.info("\n📊 CONFRONTO CHECKPOINTS")
    logger.info("-" * 50)
    
    orig = torch.load(original_path, map_location='cpu')
    extr = torch.load(extracted_path, map_location='cpu')
    
    if 'model' in orig:
        orig = orig['model']
    if 'model' in extr:
        extr = extr['model']
    
    orig_size = sum(p.numel() for p in orig.values() if isinstance(p, torch.Tensor))
    extr_size = sum(p.numel() for p in extr.values() if isinstance(p, torch.Tensor))
    
    reduction = (1 - extr_size / orig_size) * 100
    
    logger.info(f"Originale:")
    logger.info(f"  - Keys: {len(orig)}")
    logger.info(f"  - Parametri: {orig_size/1e6:.2f}M")
    logger.info(f"\nEstratto:")
    logger.info(f"  - Keys: {len(extr)}")
    logger.info(f"  - Parametri: {extr_size/1e6:.2f}M")
    logger.info(f"\nRiduzione: {reduction:.1f}%")
    logger.info("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fase 4: Estrazione Fisica del Modello Finale")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path al checkpoint del training completo')
    parser.add_argument('--output', type=str, default='checkpoints/Davide_2Simplex_40M_Finale.pt',
                        help='Path di output per il modello finale')
    parser.add_argument('--layers_to_remove', type=int, nargs='+', default=[1, 2, 4, 6, 8, 9],
                        help='Indici (0-based) dei layer da rimuovere')
    parser.add_argument('--verify', action='store_true',
                        help='Verifica integrità del modello estratto')
    parser.add_argument('--compare', type=str, default=None,
                        help='Confronta con checkpoint originale (fornire path)')
    
    args = parser.parse_args()
    
    # Estrazione
    estrai_modello_finale(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        layers_to_remove=args.layers_to_remove
    )
    
    # Verifica (opzionale)
    if args.verify:
        verifica_integrità(Path(args.output))
    
    # Confronto (opzionale)
    if args.compare:
        confronta_checkpoints(Path(args.compare), Path(args.output))


if __name__ == "__main__":
    main()
