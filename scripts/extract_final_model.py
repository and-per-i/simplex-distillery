import torch
import argparse
import collections

def extract_surviving_layers(input_path, output_path):
    print(f"Loading trained weights from {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu")
    
    # Layer da rimuovere (0-indexed): 2, 3, 5, 7, 9, 10
    # Che corrispondono a quelli bypassati nel piano: [1, 2, 4, 6, 8, 9]
    to_remove = [1, 2, 4, 6, 8, 9]
    
    nuovo_state_dict = collections.OrderedDict()
    
    # Identifichiamo i layer superstiti
    # Maestro ha 12 layer (0-11). Superstiti: 0, 3, 5, 7, 10, 11 (Esempio)
    # In realtà il piano dice: rimpiazza i layer 2, 3, 5, 7, 9, 10.
    # Quindi i superstiti sono gli indici che NON sono in to_remove.
    
    surviving_indices = sorted([i for i in range(12) if i not in to_remove])
    mapping = {old: new for new, old in enumerate(surviving_indices)}
    
    print(f"Mapping layers: {mapping}")

    for key, weight in state_dict.items():
        if "layers." in key:
            parts = key.split(".")
            old_idx = int(parts[2]) # model.model.layers.N... -> parts[3] se StudentForCausalLM
            # Adattamento per StudentForCausalLM: model.model.layers.0.attention...
            # In state_dict di HF: model.layers.0.attention...
            
            # Cerchiamo l'indice del layer nel nome della chiave
            idx_pos = -1
            for i, p in enumerate(parts):
                if p.isdigit():
                    idx_pos = i
                    break
            
            if idx_pos != -1:
                layer_idx = int(parts[idx_pos])
                if layer_idx in to_remove:
                    continue # Scarta questo peso
                
                # Rinomina l'indice
                new_idx = mapping[layer_idx]
                parts[idx_pos] = str(new_idx)
                new_key = ".".join(parts)
                nuovo_state_dict[new_key] = weight
            else:
                nuovo_state_dict[key] = weight
        else:
            nuovo_state_dict[key] = weight

    print(f"Extracted {len(surviving_indices)} layers.")
    print(f"Saving final model to {output_path}...")
    torch.save(nuovo_state_dict, output_path)
    print("Physical extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 4: Estrazione Fisica del modello 2-Simplex")
    parser.add_argument("--input_path", type=str, required=True, help="Checkpoint finale dello studente")
    parser.add_argument("--output_path", type=str, default="Davide_2Simplex_40M_Finale.pt")
    
    args = parser.parse_args()
    extract_surviving_layers(args.input_path, args.output_path)
