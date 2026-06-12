"""
ModelSVD: SVD Knowledge Injection — Algorithm Sketch
=====================================================
ACADEMIC REFERENCE ONLY. This is pseudocode for illustrating the method.
It is NOT executable and deliberately omits critical implementation details.
For commercial licensing, contact 1417981857@qq.com.
"""

# ============================================================
# Algorithm 1: SVD Knowledge Encoding
# ============================================================

def encode_knowledge_fingerprint(delta_W, tau=0.85, epsilon=0.01):
    """
    Extract purified knowledge directions from a weight delta matrix.

    Parameters
    ----------
    delta_W : ndarray (m, n)
        Weight delta from LoRA training: delta_W = B @ A
    tau : float
        Energy retention threshold for purification
    epsilon : float
        Relative energy floor for noise filtering

    Returns
    -------
    fingerprint : dict
        Purified singular components {U, S, V, k_star}
    """
    # Step 1: SVD decomposition
    U, S, Vt = svd(delta_W)                        # U(m,m), S(min_dim,), Vt(n,n)

    # Step 2: Energy-based purification
    total_energy = sum(S ** 2)
    k_star = min(
        k for k in range(len(S))
        if sum(S[:k] ** 2) / total_energy >= tau
    )
    S[k_star:] = 0                                   # Truncate low-energy noise
    S[S < epsilon * S[0]] = 0                        # Relative threshold

    return {"U": U, "S": S, "Vt": Vt, "k_star": k_star}


# ============================================================
# Algorithm 2: Bottom-Swap Injection
# ============================================================

def bottom_swap_inject(W_orig, fingerprint, rank, offset):
    """
    Inject knowledge directions into bottom singular positions.

    Principle: Replace the LEAST important singular vectors of W_orig
    with the MOST important singular vectors of delta_W.
    Top vectors (base capability) are never touched.

    Parameters
    ----------
    W_orig : ndarray (m, n)
        Original weight matrix
    fingerprint : dict
        Purified knowledge fingerprint from encode_knowledge_fingerprint()
    rank : int
        Number of knowledge directions to inject
    offset : int
        Number of bottom positions to skip (protects noise-level directions)

    Returns
    -------
    W_new : ndarray (m, n)
        Injected weight matrix
    """
    U_w, S_w, Vt_w = svd(W_orig)
    U_d, S_d, Vt_d = fingerprint["U"], fingerprint["S"], fingerprint["Vt"]
    min_dim = min(W_orig.shape)

    W_new = W_orig.copy()
    for j in range(rank):
        src = j                                     # j-th most important in delta_W
        dst = min_dim - 1 - offset - j              # j-th least important in W_orig

        # Remove old weak direction
        W_new -= S_w[dst] * outer(U_w[:, dst], Vt_w[dst, :])

        # Add new knowledge direction
        direction = S_d[src] * outer(U_d[:, src], Vt_d[src, :])
        W_new += direction

    return W_new


# ============================================================
# Algorithm 3: Deep-Layer + FFN-Only Injection
# ============================================================

def inject_model_knowledge(model, delta_weights, config):
    """
    Apply knowledge injection to target model with safety constraints.

    Constraints:
    - Deep layers only (latter 40-60% of transformer layers)
    - FFN matrices only (gate_proj, up_proj, down_proj)
    - Skip attention projections (q_proj, k_proj, v_proj, o_proj)
    """
    allowed_modules = {"gate_proj", "up_proj", "down_proj"}   # FFN only
    deep_start = config["deep_start"]                          # e.g., layer 14 of 24

    for layer_idx in range(deep_start, config["num_layers"]):
        for module_name in allowed_modules:
            W_orig = model[layer_idx][module_name]
            delta = delta_weights[layer_idx][module_name]

            # Encode knowledge fingerprint
            fp = encode_knowledge_fingerprint(delta)

            # Inject via Bottom-Swap
            W_new = bottom_swap_inject(
                W_orig, fp,
                rank=config["rank"],
                offset=config["offset"]
            )
            model[layer_idx][module_name] = W_new

    return model


# ============================================================
# Key Hyperparameters (empirically determined sweet spot)
# ============================================================
"""
SWEET_SPOT = {
    "rank": 8,           # Number of knowledge directions
    "offset": 16,        # Bottom offset for injection position
    "deep_start": 14,    # First layer to inject (0-indexed, 24-layer model)
    "tau": 0.85,         # Energy retention for purification
    "epsilon": 0.01,     # Relative noise floor
}
"""
