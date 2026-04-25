"""
FASE 1: LA FORGIA (SVD + Identità Perturbata)
=============================================

Script offline per generare studente_inizializzato.pt dal modello AlphaGeometry.

Processo:
1. Carica pytorch_model.bin del Teacher (151M, 12 layers)
2. Per ogni layer: applica SVD per dimezzare dimensioni (768 → 384)
3. Per layer simpliciali (4, 8, 12): clona K e aggiungi rumore (Identità Perturbata)
4. Salva studente_inizializzato.pt (40M, 12 layers compressi)

Usage:
    python scripts/fase1_forgia_svd.py --teacher_path pt_ckpt/params.sav --output studente_inizializzato.pt
"""

import torch
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def rimpicciolisci_svd(matrice: torch.Tensor, nuova_dim: int) -> torch.Tensor:
    """
    Applica SVD per comprimere una matrice mantenendo le informazioni principali.
    
    Args:
        matrice: Tensor di forma (dim_old, dim_old) o (dim_old, *)
        nuova_dim: Nuova dimensione ridotta (es. 384 invece di 768)
    
    Returns:
        Matrice compressa di forma (nuova_dim, nuova_dim)
    """
    original_dtype = matrice.dtype
    original_device = matrice.device
    
    # SVD richiede float32 per stabilità numerica
    matrice_fp32 = matrice.to(torch.float32).cpu()
    
    U, S, Vh = torch.linalg.svd(matrice_fp32, full_matrices=False)
    
    # Tronca ai primi nuova_dim componenti principali
    U_comp = U[:, :nuova_dim]
    S_comp = S[:nuova_dim]
    Vh_comp = Vh[:nuova_dim, :]
    
    # Ricostruisce la matrice compressa
    matrice_compressa = U_comp @ torch.diag(S_comp) @ Vh_comp
    
    # Ritaglia alle dimensioni target
    matrice_compressa = matrice_compressa[:nuova_dim, :nuova_dim]
    
    return matrice_compressa.to(original_dtype).to(original_device)


def applica_identita_perturbata(matrice: torch.Tensor, noise_scale: float = 1e-4) -> torch.Tensor:
    """
    TRUCCO DEL CLONE: Crea una copia con rumore microscopico per layer simpliciali.
    
    Args:
        matrice: Matrice originale K
        noise_scale: Scala del rumore gaussiano (default: 1e-4)
    
    Returns:
        K' = K + ε, dove ε ~ N(0, noise_scale)
    """
    rumore = torch.randn_like(matrice) * noise_scale
    K_prime = matrice.clone() + rumore
    
    logger.info(f"  → Identità Perturbata applicata (noise_scale={noise_scale:.2e})")
    return K_prime


def comprimi_layer_attention(layer_state: dict, layer_idx: int, nuova_dim: int, 
                              is_simplicial: bool = False) -> dict:
    """
    Comprime le matrici di attenzione (Q, K, V, O) di un layer.
    """
    compressed = {}
    
    for key, W in layer_state.items():
        # Saltiamo scale factors o altri parametri non-weight/bias
        if 'weight' not in key.lower() and 'bias' not in key.lower():
            continue

        if any(x in key.lower() for x in ['query', 'queries', 'key', 'keys', 'value', 'values', 'output', 'out']):
            if W.dim() == 2:
                # Matrice 2D: SVD
                compressed[key] = rimpicciolisci_svd(W, nuova_dim)
                logger.info(f"  {key}: {W.shape} → {compressed[key].shape}")
                
                # TRUCCO DEL CLONE per layer simpliciali
                if is_simplicial and any(x in key.lower() for x in ['key', 'keys']):
                    key_prime = key.replace('keys', 'keys_prime').replace('key', 'key_prime')
                    compressed[key_prime] = applica_identita_perturbata(compressed[key])
            elif W.dim() == 1:
                # Vettore 1D (bias): Troncamento
                compressed[key] = W[:nuova_dim]
                logger.info(f"  {key} (bias): {W.shape} → {compressed[key].shape}")
            else:
                # Altro: copia speculare
                compressed[key] = W
    
    return compressed


def comprimi_layer_mlp(layer_state: dict, layer_idx: int, nuova_dim: int) -> dict:
    """
    Comprime le matrici MLP (feed-forward) di un layer.
    """
    compressed = {}
    
    for key, W in layer_state.items():
        if 'mlp' in key.lower() or 'ffn' in key.lower():
            # Evitiamo di processare i layernorm qui, se possibile (verranno gestiti dopo)
            if 'layernorm' in key.lower() or 'ln' in key.lower():
                continue

            if W.dim() == 2:
                # MLP di solito ha dim nascosta 4x (es. 768 → 3072)
                # Dobbiamo capire se è il primo layer (espansione) o il secondo (contrazione)
                if W.shape[0] > W.shape[1]:  # Espansione: (Hidden*4, Hidden)
                    target_dim_out = nuova_dim * 4
                    target_dim_in = nuova_dim
                    # rimpicciolisci_svd ridimensiona matrice quadrata o taglia
                    # Per matrici rettangolari dobbiamo stare attenti
                    compressed[key] = rimpicciolisci_svd_rect(W, target_dim_out, target_dim_in)
                else:  # Contrazione: (Hidden, Hidden*4)
                    target_dim_out = nuova_dim
                    target_dim_in = nuova_dim * 4
                    compressed[key] = rimpicciolisci_svd_rect(W, target_dim_out, target_dim_in)
                
                logger.info(f"  MLP: {W.shape} → {compressed[key].shape}")
            
            elif W.dim() == 1:
                # Bias MLP
                if W.shape[0] > nuova_dim * 2:  # Probabilmente bias del layer espanso
                    compressed[key] = W[:nuova_dim * 4]
                else:
                    compressed[key] = W[:nuova_dim]
                logger.info(f"  MLP bias: {W.shape} → {compressed[key].shape}")
    
    return compressed


def rimpicciolisci_svd_rect(matrice: torch.Tensor, target_out: int, target_in: int) -> torch.Tensor:
    """SVD per matrici rettangolari."""
    original_dtype = matrice.dtype
    original_device = matrice.device
    
    matrice_fp32 = matrice.to(torch.float32).cpu()
    U, S, Vh = torch.linalg.svd(matrice_fp32, full_matrices=False)
    
    # K è il numero di componenti da tenere (il minimo tra i due target)
    k = min(target_out, target_in, S.shape[0])
    
    U_comp = U[:, :k]
    S_comp = S[:k]
    Vh_comp = Vh[:k, :]
    
    res = U_comp @ torch.diag(S_comp) @ Vh_comp
    # Ritaglia/Pad alle dimensioni esatte richieste
    final = torch.zeros((target_out, target_in), dtype=torch.float32)
    h, w = min(res.shape[0], target_out), min(res.shape[1], target_in)
    final[:h, :w] = res[:h, :w]
    
    return final.to(original_dtype).to(original_device)


def forgia_studente(teacher_path: Path, output_path: Path, 
                    dim_originale: int = 1024, nuova_dim: int = 384,
                    simplicial_layers: list = [3, 7, 11]):
    """
    FUNZIONE PRINCIPALE: Genera lo studente inizializzato da Teacher.
    
    Args:
        teacher_path: Path al checkpoint del teacher (params.sav)
        output_path: Path dove salvare studente_inizializzato.pt
        dim_originale: Dimensione hidden del teacher (default: 1024 per AG)
        nuova_dim: Dimensione hidden dello student (default: 384)
        simplicial_layers: Indici (0-based) dei layer simpliciali (default: [3,7,11] = layer 4,8,12)
    """
    logger.info("=" * 60)
    logger.info("FASE 1: LA FORGIA - Compressione SVD + Identità Perturbata")
    logger.info("=" * 60)
    
    # Carica Teacher checkpoint
    logger.info(f"\n📦 Caricamento Teacher da: {teacher_path}")
    
    # Se è una directory, punta al file params.sav
    actual_teacher_path = teacher_path / "params.sav" if teacher_path.is_dir() else teacher_path
    
    if not actual_teacher_path.exists():
        raise FileNotFoundError(f"Checkpoint Teacher non trovato: {actual_teacher_path}")
        
    teacher_state = torch.load(actual_teacher_path, map_location='cpu')
    
    # Determina numero di layer
    num_layers = 12  # AlphaGeometry standard
    logger.info(f"📊 Configurazione:")
    logger.info(f"  - Layers: {num_layers}")
    logger.info(f"  - Dimensione originale: {dim_originale}")
    logger.info(f"  - Nuova dimensione: {nuova_dim}")
    logger.info(f"  - Layer Simpliciali (0-indexed): {simplicial_layers}")
    logger.info(f"  - Riduzione: {(1 - (nuova_dim/dim_originale)**2)*100:.1f}% parametri")
    
    # Inizializza state dict dello studente
    student_state = {}
    
    # META-INFORMAZIONI per il modello
    student_state['meta'] = {
        'dim_hidden': nuova_dim,
        'num_layers': num_layers,
        'simplicial_layers': simplicial_layers,
        'vocab_size': 1024,
        'source': 'AlphaGeometry_SVD_Compressed'
    }
    
    # Loop sui 12 layer
    for i in range(num_layers):
        is_simplicial = i in simplicial_layers
        layer_type = "🔷 SIMPLICIALE" if is_simplicial else "📘 Standard"
        
        logger.info(f"\n🔨 Layer {i+1:2d}/12 {layer_type}")
        
        # Estrai lo state del layer i dal teacher
        # Nota: la struttura dipende da come è salvato AlphaGeometry
        # Pattern comune: "layers.{i}.*" o "transformer{i}.*"
        layer_keys = [k for k in teacher_state.keys() if f'transformer{i}' in k or f'layers.{i}' in k]
        
        if not layer_keys:
            logger.warning(f"  ⚠️ Nessuna chiave trovata per layer {i} - skippo")
            continue
        
        layer_state = {k: teacher_state[k] for k in layer_keys}
        
        # 1. Comprimi Attention
        compressed_attn = comprimi_layer_attention(layer_state, i, nuova_dim, is_simplicial)
        student_state.update(compressed_attn)
        
        # 2. Comprimi MLP
        compressed_mlp = comprimi_layer_mlp(layer_state, i, nuova_dim)
        student_state.update(compressed_mlp)
        
        # 3. Copia LayerNorm (non si comprimono, solo si riscalano)
        for key in layer_keys:
            if 'layernorm' in key.lower() or 'ln' in key.lower():
                student_state[key] = teacher_state[key][:nuova_dim]  # Tronca se necessario
    
    # Copia embedding e output layer (modificati per nuova_dim)
    logger.info("\n📝 Processamento Embedding e Output Layer...")
    for key in teacher_state.keys():
        if 'embedding' in key.lower() or 'embed' in key.lower():
            emb = teacher_state[key]
            if emb.dim() == 2 and emb.shape[1] == dim_originale:
                # Comprimi embedding: (vocab_size, dim_originale) → (vocab_size, nuova_dim)
                student_state[key] = rimpicciolisci_svd(emb.T, nuova_dim).T
                logger.info(f"  Embedding: {emb.shape} → {student_state[key].shape}")
            else:
                student_state[key] = emb
    
    # Salva checkpoint studente
    logger.info(f"\n💾 Salvataggio studente in: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student_state, output_path)
    
    # Statistiche finali
    size_mb = output_path.stat().st_size / (1024**2)
    num_params_student = sum(p.numel() for p in student_state.values() if isinstance(p, torch.Tensor))
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ FORGIA COMPLETATA!")
    logger.info(f"📦 File generato: {output_path}")
    logger.info(f"📊 Dimensione: {size_mb:.2f} MB")
    logger.info(f"🔢 Parametri stimati: ~{num_params_student/1e6:.1f}M")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fase 1: Forgia SVD + Identità Perturbata")
    parser.add_argument('--teacher_path', type=str, default='pt_ckpt/params.sav',
                        help='Path al checkpoint del teacher')
    parser.add_argument('--output', type=str, default='checkpoints/studente_inizializzato.pt',
                        help='Path di output per lo studente')
    parser.add_argument('--dim_originale', type=int, default=1024,
                        help='Dimensione hidden del teacher')
    parser.add_argument('--nuova_dim', type=int, default=384,
                        help='Dimensione hidden dello student')
    parser.add_argument('--simplicial_layers', type=int, nargs='+', default=[3, 7, 11],
                        help='Indici (0-based) dei layer simpliciali')
    
    args = parser.parse_args()
    
    forgia_studente(
        teacher_path=Path(args.teacher_path),
        output_path=Path(args.output),
        dim_originale=args.dim_originale,
        nuova_dim=args.nuova_dim,
        simplicial_layers=args.simplicial_layers
    )


if __name__ == "__main__":
    main()
