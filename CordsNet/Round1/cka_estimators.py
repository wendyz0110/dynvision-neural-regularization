"""
Three CKA estimators from Chun, Canatar, Chung & Lee (2025), "Estimating neural
representation alignment from sparsely sampled inputs and features"
(arXiv 2502.15104; official code github.com/badooki/CKA, CCN_code/moment_utils.py).

  CKA_0  naive estimator = the usual (biased) linear CKA, as used in rounds 3/4
  CKA_S  stimulus-corrected (Song et al. 2012, "debiased" / "unbiased" HSIC)
  CKA_C  stimulus- and neuron-corrected (the paper's estimator)

This is a PyTorch port of `getest_all`. The official code builds every term with a
4-index einsum; here the same terms are written with P x P Gram matrices (for the
cross terms) and per-feature sums (for the neuron correction), which gives the same
numbers but scales to model layers with ~200k features. Features are centered over
stimuli first (the estimators are invariant to that; it only improves precision).
`test_cka_estimators.py` checks the port against the official code.

Conventions (as in the paper's code):
  A, B are (P, Q) arrays: rows = stimuli (images), columns = features
  (model units or electrodes). Each is scaled by 1 / sqrt(P * Q) first.
  Model vs brain are different populations, so the numerator uses independent
  columns (its "ours" estimate equals the Song estimate); each denominator compares
  a population with itself, so it gets the neuron correction.
  CKA_0 = num_naive / sqrt(den_naive_X * den_naive_Y)
  CKA_S = num_song  / sqrt(den_song_X  * den_song_Y)
  CKA_C = num_song  / sqrt(den_ours_X  * den_ours_Y)
Needs P > 3 and Q > 1. Computed in float64.
"""
import torch


def moments(A):
    """Everything the estimators need from one (P, Q) population.

    Returns a dict with the scaled Gram matrix K (P x P), the P, Q sizes, and the
    nine per-feature sums used by the neuron correction (only for A paired with itself).
    """
    A = A.to(torch.float64)
    P, Q = A.shape
    # All three estimators are unchanged by subtracting each feature's mean over stimuli
    # (checked against the official code), but the raw form loses precision when the
    # stimuli differ only slightly (e.g. a model layer shortly after stimulus onset).
    # So center first. rel_var = variance across stimuli / total power, ~0 when all
    # stimuli give the same pattern -> CKA undefined.
    total = float((A * A).sum().detach())
    A = A - A.mean(0)
    rel_var = float((A * A).sum().detach()) / total if total > 0 else 0.0
    A = A / (P * Q) ** 0.5
    K = A @ A.T
    s1, s2 = A.sum(0), (A * A).sum(0)
    s3, s4 = (A ** 3).sum(0), (A ** 4).sum(0)
    q = torch.stack([
        (s2 * s2).sum(),            # ijji
        s4.sum(),                   # iiii
        (s1 * s3).sum(),            # ijjj
        (s3 * s1).sum(),            # iiij
        (s1 * s2 * s1).sum(),       # ijjl
        (s2 * s2).sum(),            # iijj
        (s2 * s1 * s1).sum(),       # iijl
        (s1 * s1 * s2).sum(),       # ijll
        (s1 ** 4).sum(),            # ijlm
    ])
    return {"K": K, "P": P, "Q": Q, "q": q, "rel_var": rel_var}


def _p_terms(KA, KB):
    """The nine terms sum A_ia A_ja B_lb B_mb over index patterns, via Gram matrices."""
    dA, dB = KA.diagonal(), KB.diagonal()
    return torch.stack([
        (KA * KB.T).sum(),                    # ijji
        (dA * dB).sum(),                      # iiii
        (KA * dB[None, :]).sum(),             # ijjj
        (dA[:, None] * KB).sum(),             # iiij
        (KA.sum(0) * KB.sum(1)).sum(),        # ijjl
        dA.sum() * dB.sum(),                  # iijj
        dA.sum() * KB.sum(),                  # iijl
        KA.sum() * dB.sum(),                  # ijll
        KA.sum() * KB.sum(),                  # ijlm
    ])


def _combine(t, P):
    f1, f2, f3 = P / (P - 2), 2 / (P - 2), 1 / ((P - 1) * (P - 2))
    naive = t[0] - 2 / P * t[4] + (1 / P) ** 2 * t[8]
    song = P / (P - 3) * (t[0] - f1 * t[1] + f2 * (t[2] + t[3] - t[4]) + f3 * (t[5] - t[6] - t[7] + t[8]))
    return naive, song


def h_cross(ma, mb):
    """H estimates between two different populations: (naive, song). 'Ours' = song here."""
    assert ma["P"] == mb["P"], "both populations need the same stimuli"
    return _combine(_p_terms(ma["K"], mb["K"]), ma["P"])


def h_self(m):
    """H estimates of a population with itself: (naive, song, ours)."""
    P, Q = m["P"], m["Q"]
    t = _p_terms(m["K"], m["K"])
    naive, song = _combine(t, P)
    _, s = _combine(t - m["q"], P)          # same formula on the neuron-corrected terms
    ours = (Q / (Q - 1)) * s                # = (P/(P-3)) (Q/(Q-1)) (...) as in the paper's code
    return naive, song, ours


def cka_all(mx, my, hx=None, hy=None, min_rel_var=1e-8):
    """CKA_0, CKA_S, CKA_C between populations X and Y (pass h_self results to reuse them).

    Returns NaNs when either population has (almost) no variance across stimuli.
    A corrected estimator can also be NaN when its denominator estimate is negative
    (possible with few stimuli).
    """
    if min(mx["rel_var"], my["rel_var"]) < min_rel_var:
        nan = torch.tensor(float("nan"), dtype=torch.float64)
        return nan, nan, nan
    num_n, num_s = h_cross(mx, my)
    hx = hx if hx is not None else h_self(mx)
    hy = hy if hy is not None else h_self(my)
    return (num_n / torch.sqrt(hx[0] * hy[0]),
            num_s / torch.sqrt(hx[1] * hy[1]),
            num_s / torch.sqrt(hx[2] * hy[2]))


def linear_cka(X, Y):
    """Plain linear CKA with centered Gram matrices (reference for CKA_0)."""
    X, Y = X.to(torch.float64), Y.to(torch.float64)
    X, Y = X - X.mean(0), Y - Y.mean(0)
    hxy = (X.T @ Y).pow(2).sum()
    return hxy / torch.sqrt((X.T @ X).pow(2).sum() * (Y.T @ Y).pow(2).sum())
