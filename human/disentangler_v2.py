"""Difference voting + adaptive per-series tolerance.

1. Pairwise differences cancel the series offset.
2. Each difference D votes D / m, m = 1..M; peaks give the steps.
3. Series are peeled as chains x -> x + d (graph connected components).
4. Refinement: each value joins the series with the nearest predicted
   neighbour (P_k +/- d_k) within that series' tolerance; per-series sigma is
   re-estimated from its own gaps and the tolerance only shrinks.
   Unassigned values are voted again and may form new series.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.signal import find_peaks


def _votes(a, max_divisor):
    D = np.subtract.outer(a, a)
    D = D[D > 0]
    m = np.arange(1, max_divisor + 1)
    V = (D[:, None] / m).ravel()
    W = np.broadcast_to(1.0 / m, (len(D), len(m))).ravel()
    keep = V >= max(np.diff(a).min(), 0.5)
    return V[keep], W[keep]


def _steps(V, W, top, bw=0.25):
    """Top peaks of the 1/m-weighted vote histogram, refined by a shrinking window median."""
    h, e = np.histogram(V, bins=np.arange(V.min(), V.max() + 2 * bw, bw), weights=W)
    h = np.convolve(h, np.ones(5), "same")
    c = (e[:-1] + e[1:]) / 2
    peaks, _ = find_peaks(np.r_[0, h, 0])
    out = []
    for d in c[peaks[np.argsort(-h[peaks - 1])][:top] - 1]:
        for w in (1.5, 0.8, 0.4, 0.2, 0.1):
            near = V[np.abs(V - d) < w]
            if len(near):
                d = np.median(near)
        out.append(d)
    return out


def _chain(a, d, tol):
    """Largest connected component of edges a[i] -> a[j] with a[j] - a[i] ~ d."""
    n = len(a)
    t = a + d
    j = np.clip(np.searchsorted(a, t), 1, n - 1)
    j = np.where(np.abs(a[j] - t) < np.abs(a[j - 1] - t), j, j - 1)
    ok = np.abs(a[j] - t) < tol
    i, j = np.arange(n)[ok], j[ok]
    _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    comp = lab == np.bincount(lab).argmax()
    fwd, bwd = np.zeros(n, bool), np.zeros(n, bool)
    fwd[i], bwd[j] = True, True
    inner = (a > a[comp].min() + tol) & (a < a[comp].max() - tol)
    return comp & (~inner | (fwd & bwd))


def _fit_chain(a, d, min_len):
    """Smallest tolerance after which the chain stops growing."""
    tols = [t for t in (0.3, 0.6, 1.2, 2.4, 4.8, 9.6) if t <= 0.2 * d] or [0.3]
    masks = [_chain(a, d, t) for t in tols]
    sizes = np.array([m.sum() for m in masks])
    for m, size, nxt in zip(masks, sizes, np.r_[sizes[1:], sizes[-1]]):
        if size >= nxt and size >= min_len:
            return m
    return None


def _peel(x, max_divisor, min_len, top=8):
    labels = -np.ones(len(x), int)
    k = 0
    while (labels < 0).sum() >= min_len:
        rem = np.where(labels < 0)[0]
        a = x[rem]
        V, W = _votes(a, max_divisor)
        if len(V) == 0:
            break
        masks = [m for m in (_fit_chain(a, d, min_len) for d in _steps(V, W, top)) if m is not None]
        mask = max(masks, key=lambda m: (m.sum(), np.ptp(a[m]))) if masks else None
        if mask is None or mask.sum() < min_len:
            break
        labels[rem[mask]] = k
        k += 1
    return labels


def _gap_step(P, d):
    g = np.diff(np.sort(P))
    g = g[(g > 0.5 * d) & (g < 1.5 * d)]
    return np.median(g) if len(g) else d


def _pooled_sigma(x, labels, steps):
    """Noise scale from in-series gap residuals pooled over all series."""
    r = []
    for k, d in enumerate(steps):
        g = np.diff(np.sort(x[labels == k]))
        g = g[(g > 0.5 * d) & (g < 1.5 * d)]
        r.append(g - np.median(g) if len(g) else g)
    r = np.concatenate(r) if r else np.array([])
    return 1.4826 * np.median(np.abs(r)) if len(r) >= 3 else None


def _neighbor_cost(x, labels, steps):
    C = np.full((len(x), len(steps)), np.inf)
    for k, d in enumerate(steps):
        P = x[labels == k]
        if len(P) == 0:
            continue
        Q = np.sort(np.r_[P - d, P + d])
        j = np.clip(np.searchsorted(Q, x), 1, len(Q) - 1)
        C[:, k] = np.minimum(np.abs(x - Q[j]), np.abs(x - Q[j - 1]))
    return C


def _can_merge(P, Q, d):
    """Two series with close steps put their values in near pairs; one series does not."""
    gaps = np.diff(np.sort(np.r_[P, Q]))
    return (gaps < d / 4).sum() <= min(len(P), len(Q)) // 2


def _compact(x, labels, steps, tol, min_len, merge=0.3):
    """Merge series with equal steps, drop short ones (their values become -1)."""
    order = np.argsort(steps)
    target = -np.ones(len(steps), int)
    kept = []
    for k in order:
        for j, s in enumerate(kept):
            if (abs(steps[k] - steps[s]) < max(0.015 * steps[s], merge)
                    and _can_merge(x[labels == k], x[labels == s], steps[s])):
                target[k] = j
                break
        else:
            target[k] = len(kept)
            kept.append(k)
    new = np.where(labels >= 0, target[np.maximum(labels, 0)], -1)
    counts = np.bincount(new[new >= 0], minlength=len(kept))
    keep = counts >= min_len
    remap = -np.ones(len(kept), int)
    remap[keep] = np.arange(keep.sum())
    new = np.where(new >= 0, remap[np.maximum(new, 0)], -1)
    idx = np.array(kept, int)[keep]
    return new, steps[idx], tol[idx]


def _relabel(labels):
    """Cluster ids in order of first appearance."""
    _, first = np.unique(labels, return_index=True)
    order = labels[np.sort(first)]
    lut = np.empty(labels.max() + 1, int)
    lut[order] = np.arange(len(order))
    return lut[labels]


def disentangle(x, max_divisor=10, min_len=3, iters=25, tol_floor=0.2):
    x = np.asarray(x, float)
    if len(x) < 2 * min_len:
        return np.zeros(len(x), int)

    labels = _peel(x, max_divisor, min_len)
    if labels.max() < 0:
        return np.zeros(len(x), int)
    steps = np.array([_gap_step(x[labels == k], np.median(np.diff(np.sort(x[labels == k]))))
                      for k in range(labels.max() + 1)])
    tol = 0.25 * steps
    labels, steps, tol = _compact(x, labels, steps, tol, min_len)

    for _ in range(iters):
        if len(steps) == 0:
            break
        C = _neighbor_cost(x, labels, steps)
        best = C.argmin(1)
        new = np.where(C[np.arange(len(x)), best] <= tol[best], best, -1)

        for k in range(len(steps)):
            P = x[new == k]
            if len(P) > 1:
                steps[k] = _gap_step(P, steps[k])
        sigma = _pooled_sigma(x, new, steps)
        if sigma is not None:
            tol = np.minimum(tol, max(4 * sigma, tol_floor))

        free = np.where(new < 0)[0]
        if len(free) >= 2 * min_len:
            sub = _peel(x[free], max_divisor, min_len)
            for j in range(sub.max() + 1):
                P = x[free][sub == j]
                d = _gap_step(P, np.median(np.diff(np.sort(P))))
                if not np.any(np.abs(steps - d) < max(0.015 * d, 0.3)):
                    new[free[sub == j]] = len(steps)
                    steps = np.r_[steps, d]
                    tol = np.r_[tol, 0.25 * d]

        new, steps, tol = _compact(x, new, steps, tol, min_len, merge=max(0.3, 2 * (sigma or 0)))
        if np.array_equal(new, labels):
            break
        labels = new

    if len(steps) == 0:
        return np.zeros(len(x), int)
    lost = labels < 0
    if lost.any():
        labels[lost] = _neighbor_cost(x, labels, steps)[lost].argmin(1)
    return _relabel(labels)


def disentangle_algorithm(sequence_entangled):
    a = np.asarray(sequence_entangled, float)
    labels = disentangle(a)
    order = np.lexsort((a, labels))
    return a[order], labels[order], labels
