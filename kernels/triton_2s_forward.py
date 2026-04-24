import torch
import triton
import triton.language as tl

@triton.jit
def _triton_2s_attn_fwd_kernel(
    Q, K, V, Kp, Vp,
    Out, L, # L for softmax normalization (logsumexp)
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_on, stride_oh, stride_od,
    n_tokens, n_heads, head_dim,
    w1, w2,
    BLOCK_SIZE: tl.constexpr,
):
    # Id del programma (parallelo su token e head)
    idx_token = tl.program_id(0)
    idx_head = tl.program_id(1)

    # Offset iniziali
    offs_d = tl.arange(0, BLOCK_SIZE)
    
    # Caricamento Q_i (1, H, D)
    q_ptrs = Q + idx_token * stride_qn + idx_head * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)

    # Range delle finestre (sliding window)
    j_start = tl.maximum(0, idx_token - w1 + 1)
    k_start = tl.maximum(0, idx_token - w2 + 1)
    
    # Inizializzatori per softmax (online softmax style)
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Ciclo sulla finestra j e k
    # Per semplicità in questo kernel MVP, iteriamo esplicitamente sulle finestre
    for j in range(j_start, idx_token + 1):
        # Carica K_j
        k_ptrs = K + j * stride_kn + idx_head * stride_kh + offs_d * stride_kd
        k = tl.load(k_ptrs)
        
        # Carica V_j
        v_ptrs = V + j * stride_vn + idx_head * stride_vh + offs_d * stride_vd
        v = tl.load(v_ptrs)

        for k_idx in range(k_start, idx_token + 1):
            # Carica Kp_k
            kp_ptrs = Kp + k_idx * stride_kn + idx_head * stride_kh + offs_d * stride_kd
            kp = tl.load(kp_ptrs)
            
            # Carica Vp_k
            vp_ptrs = Vp + k_idx * stride_vn + idx_head * stride_vh + offs_d * stride_vd
            vp = tl.load(vp_ptrs)

            # Calcolo Score: (q * k * kp) / sqrt(d)
            # Nota: tl.sum fa la riduzione sul prodotto elemento per elemento
            score = tl.sum(q * k * kp) / tl.sqrt(head_dim.to(tl.float32))
            
            # Online Softmax update
            m_next = tl.maximum(m_i, score)
            l_i = l_i * tl.exp(m_i - m_next) + tl.exp(score - m_next)
            
            # Update accumulatore: acc = acc * exp(m_i - m_next) + p * (v * vp)
            p = tl.exp(score - m_next)
            acc = acc * tl.exp(m_i - m_next) + p * (v * vp)
            
            m_i = m_next

    # Normalizzazione finale
    acc = acc / l_i
    
    # Salvataggio Output
    out_ptrs = Out + idx_token * stride_on + idx_head * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc)
    
    # Salvataggio L (necessario per il backward)
    l_ptr = L + idx_token * n_heads + idx_head
    tl.store(l_ptr, m_i + tl.log(l_i))

def forward(tri_feats, edge_index, Q, K, V, Kp, Vp, out_dim, num_heads, head_dim, w1=8, w2=8):
    # Nota: tri_feats e edge_index sono placeholders per compatibilità con l'interfaccia 2-simplex completa
    # In questo distillatore usiamo sequenze tokenizzate, quindi la topologia è la sliding window [w1, w2]
    
    N, H, D = Q.shape
    Out = torch.empty_like(V)
    L = torch.empty((N, H), device=Q.device, dtype=torch.float32)
    
    # Configurazione GPU
    BLOCK_SIZE = D # Assumiamo head_dim <= 128/256 (standard)
    grid = (N, H)
    
    _triton_2s_attn_fwd_kernel[grid](
        Q, K, V, Kp, Vp,
        Out, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        N, H, D,
        w1, w2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return Out, L
