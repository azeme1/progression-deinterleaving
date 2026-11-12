"""Disentangle interleaved progressions with step features and clustering.

Pipeline:
    1. Features: each value x votes for the steps it could belong to, x / n for
       n = 1..L. Values of one series all vote near that series' step.
    2. Vote clustering: 1D single-linkage clustering of the votes gives
       candidate steps, ranked by how many distinct values support them.
    3. Comb scoring: a candidate step d predicts the slots d, 2d, ..., Ld.
       It scores by how many slots hold a value. Half or double the true step
       only fills about half the slots, so the true step wins.
    4. Value clustering: a k-means style loop where each centroid is a whole
       progression. Values are matched one-to-one to slots, then each series'
       step and offset are refit by least squares, until nothing changes.
    5. Model selection: the number of series K must divide the number of
       values. Each K is tried from smallest to largest, and the first one whose
       residuals all stay within tolerance wins.
"""

from itertools import combinations

import numpy as np


# ---------- 1. Features ----------

def step_votes(a, length):
    """Each difference between two values votes D / m for every term gap m it could span."""
    i, j = np.triu_indices(len(a), 1)
    m = np.arange(1, length)
    votes = ((a[j] - a[i])[:, None] / m[None, :]).ravel()
    owners = np.repeat(j, len(m))
    return votes, owners


# ---------- 2. Vote clustering ----------

def cluster_votes(votes, owners, gap):
    """Single-linkage clustering in 1D: sorted votes split wherever the gap exceeds `gap`.

    Returns (center, support) per cluster, where support counts distinct values.
    """
    order = np.argsort(votes)
    v, o = votes[order], owners[order]
    breaks = np.flatnonzero(np.diff(v) > gap) + 1
    clusters = []
    for idx in np.split(np.arange(len(v)), breaks):
        clusters.append((float(np.median(v[idx])), len(set(o[idx].tolist()))))
    return clusters


# ---------- 3. Comb scoring ----------

def comb_fit(a, step, length, tolerance, rounds=3):
    """Refine a step by fitting the values that land on its comb.

    Returns (step, offset, filled slots).
    """
    n = np.arange(1, length + 1)
    # Start the comb at the first value of the most common phase a mod step.
    phase = a % step
    diff = np.abs(phase[:, None] - phase[None, :])
    close = np.minimum(diff, step - diff) <= tolerance
    members = close[close.sum(axis=1).argmax()]
    d, c = step, a[members].min() - step
    filled = 0
    for _ in range(rounds):
        slots = n * d + c
        dist = np.abs(a[None, :] - slots[:, None])
        nearest = dist.argmin(axis=1)
        hit = dist[np.arange(length), nearest] <= tolerance
        filled = int(hit.sum())
        if filled >= 2:
            d, c = np.polyfit(n[hit], a[nearest[hit]], 1)
        elif filled == 1:
            d = a[nearest[hit]][0] / n[hit][0]
    return float(d), float(c), filled


def candidate_steps(a, length, tolerance, gaps, top, dedupe=0.1):
    """Cluster the step votes at several scales and refine each cluster on its comb.

    Fine gaps keep series with nearly equal steps apart. Coarse gaps still catch
    series whose votes are spread out by noise.
    """
    votes, owners = step_votes(a, length)
    clusters = []
    for gap in gaps:
        clusters += cluster_votes(votes, owners, gap)
    # Keep clusters supported by enough distinct values to be a real series.
    clusters = [c for c in clusters if c[1] >= max(2, length // 2)]
    clusters.sort(key=lambda c: -c[1])

    candidates = []
    for center, _ in clusters:
        d, c, filled = comb_fit(a, center, length, tolerance)
        if d <= 0:
            continue
        if any(abs(d - other[0]) < dedupe for other in candidates):
            continue
        candidates.append((d, c, filled))
    candidates.sort(key=lambda c: -c[2])
    return candidates[:top]


# ---------- 4. Value clustering ----------

def match(a, steps, offsets, length):
    """Assign each value to one slot (series, n), greedily by smallest distance.

    Then swap pairs of values between series while it lowers the total cost.
    """
    k = len(steps)
    n = np.arange(1, length + 1)
    slots = (n[None, :] * steps[:, None] + offsets[:, None]).ravel()
    cost = (a[:, None] - slots[None, :]) ** 2

    labels = np.full(len(a), -1)
    positions = np.zeros(len(a), dtype=int)
    used_point = np.zeros(len(a), bool)
    used_slot = np.zeros(len(slots), bool)
    for flat in np.argsort(cost, axis=None):
        i, s = divmod(int(flat), len(slots))
        if used_point[i] or used_slot[s]:
            continue
        used_point[i] = used_slot[s] = True
        labels[i], positions[i] = s // length, s % length + 1

    slot_of = labels * length + positions - 1
    improved = True
    while improved:
        improved = False
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                si, sj = slot_of[i], slot_of[j]
                if cost[i, sj] + cost[j, si] + 1e-12 < cost[i, si] + cost[j, sj]:
                    slot_of[i], slot_of[j] = sj, si
                    improved = True
    labels, positions = slot_of // length, slot_of % length + 1
    return labels, positions


def refit(a, labels, positions, k):
    steps, offsets = np.zeros(k), np.zeros(k)
    for s in range(k):
        mask = labels == s
        steps[s], offsets[s] = np.polyfit(positions[mask], a[mask], 1)
    return steps, offsets


def comb_kmeans(a, steps, offsets, length, max_iter=20):
    k = len(steps)
    labels = None
    for _ in range(max_iter):
        new_labels, positions = match(a, steps, offsets, length)
        steps, offsets = refit(a, new_labels, positions, k)
        if labels is not None and (new_labels == labels).all():
            break
        labels = new_labels
    residual = a - (positions * steps[labels] + offsets[labels])
    return labels, positions, residual


# ---------- 5. Model selection ----------

def fit_k(a, k, length, tolerance, gaps, top):
    candidates = candidate_steps(a, length, tolerance, gaps, top)
    if len(candidates) < k:
        return None

    best = None
    for combo in combinations(candidates, k):
        steps = np.array([c[0] for c in combo])
        offsets = np.array([c[1] for c in combo])
        labels, positions, residual = comb_kmeans(a, steps, offsets, length)
        cost = float((residual ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, labels, positions, np.abs(residual).max())
    return best


def disentangle_algorithm(sequence_entangled, tolerance=1.5, vote_gaps=(0.15, 0.3, 0.6), top=8, min_length=2):
    """Split a sorted mix of arithmetic progressions back into its series.

    Returns, matching the original function:
        sequence_disentangled: values grouped by series, each in term order.
        class_disentangled: the series label of each value in sequence_disentangled.
        class_entangled_predicted: the series label of each value in the input order.
    """
    a = np.asarray(sequence_entangled, dtype=float)
    total = len(a)
    if total == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    chosen = None  # (worst residual, labels, positions)
    for k in range(1, total + 1):
        if total % k or total // k < min_length:
            continue
        result = fit_k(a, k, total // k, tolerance, vote_gaps, top)
        if result is None:
            continue
        cost, labels, positions, worst = result
        if worst <= tolerance:
            chosen = (worst, labels, positions)
            break
        # No K fits yet: remember the closest one.
        if chosen is None or worst < chosen[0]:
            chosen = (worst, labels, positions)

    if chosen is None:
        labels, positions = np.arange(total), np.ones(total, dtype=int)
    else:
        _, labels, positions = chosen

    order = np.lexsort((positions, labels))
    return a[order], labels[order], labels
