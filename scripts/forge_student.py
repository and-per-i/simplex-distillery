import torch
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
from models.student_model import StudentForCausalLM
from models.student_config import StudentConfig

def rimpicciolisci_svd(matrice, nuova_dim):
    """Esegue SVD e restituisce la matrice compressa alla nuova dimensione."""
    # Gestione per matrici 1D (bias) o 2D (weights)
    if len(matrice.shape) == 1:
        # Per i bias, facciamo un semplice troncamento o padding
        if matrice.shape[0] > nuova_dim:
            return matrice[:nuova_dim].clone()
        else:
            res = torch.zeros(nuova_dim, device=matrice.device, dtype=matrice.dtype)
            res[:matrice.shape[0]] = matrice
            return res

    # Per pesi 2D
    U, S, Vh = torch.linalg.svd(matrice.to(torch.float32), full_matrices=False)
    
    # Determiniamo le dimensioni di output desiderate
    out_dim, in_dim = matrice.shape
    
    # Se la matrice è quadrata (es. hidden_size x hidden_size)
    if out_dim == in_dim:
        k = nuova_dim
    else:
        # Se è rettangolare (es. intermediate_size x hidden_size)
        # Cerchiamo di mantenere il rapporto se possibile, o usiamo un'euristica
        # Qui seguiamo l'indicazione del piano: dimezzare.
        k = min(nuova_dim, S.shape[0])

    comp = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
    
    # Ridimensioniamo fisicamente la matrice al target
    # Nota: nuova_dim è riferita a hidden_size (768 -> 384)
    # MLP intermediate è tipicamente 4x hidden_size
    
    final_out = out_dim
    final_in = in_dim
    
    if out_dim > 512: # Probabilmente intermediate o vocab
        if out_dim != 30522 and out_dim != 757: # Non ridimensioniamo il vocabolario
             final_out = nuova_dim * 4 if out_dim > in_dim else nuova_dim
    
    if in_dim > 512:
         final_in = nuova_dim * 4 if in_dim > out_dim else nuova_dim

    return comp[:final_out, :final_in].to(matrice.dtype)

def forge_student(teacher_path, output_path, hidden_size=384, simplex_layers=[3, 7, 11]):
    print(f"Loading teacher from {teacher_path}...")
    teacher_state = torch.load(teacher_path, map_location="cpu")
    
    # Assumiamo che il teacher sia un AlphaGeometry standard (768 hidden)
    # Lo studente avrà hidden_size (384)
    
    print(f"Forging student with hidden_size={hidden_size}...")
    nuovo_state_dict = {}
    
    # Mappatura chiavi (Esempio basato su StudentModel)
    # Questa parte potrebbe richiedere aggiustamenti in base alle chiavi esatte del Maestro
    
    for key, weight in tqdm(teacher_state.items(), desc="Compressing layers"):
        # Mapping per embeddings
        if "embeddings.token_embeddings" in key:
            nuovo_state_dict[key] = rimpicciolisci_svd(weight, hidden_size)
        elif "embeddings.position_embeddings" in key:
            nuovo_state_dict[key] = rimpicciolisci_svd(weight, hidden_size)
        
        # Mapping per i layer
        elif "layers." in key:
            parts = key.split(".")
            layer_idx = int(parts[1])
            
            # Compressione SVD standard
            new_weight = rimpicciolisci_svd(weight, hidden_size)
            nuovo_state_dict[key] = new_weight
            
            # Trucco del Clone per Identità Perturbata sui layer Simpliciali
            if layer_idx in simplex_layers and "attention.k_proj.weight" in key:
                # Inizializziamo K' e V' (kp_proj e vp_proj)
                for proj_name in ["kp_proj", "vp_proj"]:
                    proj_key = key.replace("k_proj.weight", f"simplex_attn.{proj_name}.weight")
                    bias_key = proj_key.replace("weight", "bias")
                    
                    # Copia pesi e aggiungi rumore
                    nuovo_state_dict[proj_key] = new_weight.clone() + (torch.randn_like(new_weight) * 1e-4)
                    
                    # Cerca il bias corrispondente nel teacher
                    teacher_bias_key = key.replace("weight", "bias")
                    if teacher_bias_key in teacher_state:
                        new_bias = rimpicciolisci_svd(teacher_state[teacher_bias_key], hidden_size)
                        nuovo_state_dict[bias_key] = new_bias.clone() + (torch.randn_like(new_bias) * 1e-4)
                
                print(f" [Simplex] Added K' and V' (kp_proj, vp_proj) for layer {layer_idx}")
        
        # Mapping per final layer norm e head
        elif "final_ln" in key or "lm_head" in key:
            nuovo_state_dict[key] = rimpicciolisci_svd(weight, hidden_size)

    print(f"Saving forged student to {output_path}...")
    torch.save(nuovo_state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 1: La Forgia - SVD + Perturbed Identity")
    parser.add_argument("--teacher_path", type=str, required=True, help="Path al pytorch_model.bin del Maestro")
    parser.add_argument("--output_path", type=str, default="studente_inizializzato.pt")
    parser.add_argument("--hidden_size", type=int, default=384)
    
    args = parser.parse_args()
    forge_student(args.teacher_path, args.output_path, args.hidden_size)
