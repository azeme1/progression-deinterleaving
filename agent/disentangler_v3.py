"""Disentangle interleaved progressions with step features and clustering.

Pipeline:
    1. Features: each difference D between two values votes D / m for m = 1..M.
       Values of one series all vote near that series' step.
    2. Vote clustering: 1D single-linkage clustering of the votes gives
       candidate steps, ranked by how many distinct values support them.
    3. Comb scoring: a candidate step d predicts slots c + d, c + 2d, ... over the
       whole data range. It scores by filled slots minus empty slots between the
       first and last filled one, so half the true step scores low.
    4. Value clustering: a k-means style loop where each centroid is a whole
       progression. Values are matched one-to-one to slots, then each series'
       step and offset are refit by least squares, until nothing changes.
    5. Model selection: K is tried from 1 upwards, and the first K where every
       value is within tolerance and no series has an empty slot inside wins.
"""

from itertools import combinations

import numpy as np


# ---------- 1. Features ----------

def step_votes(a, max_divisor, min_step):
    """Each difference between two values votes D / m for every term gap m it could span."""
    i, j = np.triu_indices(len(a), 1)
    m = np.arange(1, max_divisor + 1)
    votes = ((a[j] - a[i])[:, None] / m[None, :]).ravel()
    owners = np.repeat(j, len(m))
    keep = votes >= min_step
    return votes[keep], owners[keep]


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

def comb_slots(a, step, offset):
    """Slot numbers 1..n that cover the data range for a comb x = n * step + offset."""
    return np.arange(1, int((a.max() - offset) / step) + 2)


def comb_fit(a, step, tolerance, rounds=3):
    """Refine a step by fitting the values that land on its comb.

    Returns (step, offset, score).
    """
    # Start the comb at the first value of the most common phase a mod step.
    phase = a % step
    diff = np.abs(phase[:, None] - phase[None, :])
    close = np.minimum(diff, step - diff) <= tolerance
    members = close[close.sum(axis=1).argmax()]
    d, c = step, a[members].min() - step
    score = 0
    for _ in range(rounds):
        n = comb_slots(a, d, c)
        dist = np.abs(a[None, :] - (n * d + c)[:, None])
        nearest = dist.argmin(axis=1)
        hit = np.flatnonzero(dist[np.arange(len(n)), nearest] <= tolerance)
        if len(hit) == 0:
            return float(d), float(c), 0
        score = 2 * len(hit) - (hit[-1] - hit[0] + 1)
        if len(hit) < 2:
            break
        d, c = np.polyfit(n[hit], a[nearest[hit]], 1)
        if d <= tolerance:
            return float(d), float(c), 0
    return float(d), float(c), int(score)


def candidate_steps(a, tolerance, gaps, top, max_divisor, min_length, dedupe=0.1):
    """Cluster the step votes at several scales and refine each cluster on its comb.

    Fine gaps keep series with nearly equal steps apart. Coarse gaps still catch
    series whose votes are spread out by noise.
    """
    votes, owners = step_votes(a, max_divisor, 2 * tolerance)
    clusters = []
    for gap in gaps:
        clusters += cluster_votes(votes, owners, gap)
    # Keep clusters supported by enough distinct values to be a real series.
    clusters = [c for c in clusters if c[1] >= min_length]
    clusters.sort(key=lambda c: -c[1])

    candidates = []
    for center, _ in clusters:
        d, c, score = comb_fit(a, center, tolerance)
        if score < min_length:
            continue
        if any(abs(d - other[0]) < dedupe for other in candidates):
            continue
        candidates.append((d, c, score))
    candidates.sort(key=lambda c: -c[2])
    return candidates[:top]


# ---------- 4. Value clustering ----------

def match(a, steps, offsets):
    """Assign each value to one slot (series, n), greedily by smallest distance.

    Then swap pairs of values between series while it lowers the total cost.
    """
    slot_series, slot_n = [], []
    for s, (d, c) in enumerate(zip(steps, offsets)):
        n = comb_slots(a, d, c)
        slot_series.append(np.full(len(n), s))
        slot_n.append(n)
    slot_series, slot_n = np.concatenate(slot_series), np.concatenate(slot_n)
    slots = slot_n * steps[slot_series] + offsets[slot_series]
    cost = (a[:, None] - slots[None, :]) ** 2

    slot_of = np.full(len(a), -1)
    used_slot = np.zeros(len(slots), bool)
    for flat in np.argsort(cost, axis=None):
        i, s = divmod(int(flat), len(slots))
        if slot_of[i] >= 0 or used_slot[s]:
            continue
        slot_of[i] = s
        used_slot[s] = True

    improved = True
    while improved:
        improved = False
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                si, sj = slot_of[i], slot_of[j]
                if cost[i, sj] + cost[j, si] + 1e-12 < cost[i, si] + cost[j, sj]:
                    slot_of[i], slot_of[j] = sj, si
                    improved = True
    return slot_series[slot_of], slot_n[slot_of]


def refit(a, labels, positions, steps, offsets):
    steps, offsets = steps.copy(), offsets.copy()
    for s in range(len(steps)):
        mask = labels == s
        if mask.sum() >= 2:
            steps[s], offsets[s] = np.polyfit(positions[mask], a[mask], 1)
    return steps, offsets


def comb_kmeans(a, steps, offsets, tolerance, max_iter=20):
    labels = None
    for _ in range(max_iter):
        new_labels, positions = match(a, steps, offsets)
        steps, offsets = refit(a, new_labels, positions, steps, offsets)
        if (steps <= tolerance).any():
            break
        if labels is not None and (new_labels == labels).all():
            break
        labels = new_labels
    residual = a - (positions * steps[new_labels] + offsets[new_labels])
    return new_labels, positions, residual


def no_holes(labels, positions, k, min_length):
    """Every series is a run of consecutive slots with at least min_length values."""
    for s in range(k):
        p = positions[labels == s]
        if len(p) < min_length or p.max() - p.min() + 1 != len(p):
            return False
    return True


# ---------- 5. Model selection ----------

def fit_k(a, k, tolerance, gaps, top, max_divisor, min_length):
    candidates = candidate_steps(a, tolerance, gaps, top, max_divisor, min_length)
    if len(candidates) < k:
        return None

    best = None
    for combo in combinations(candidates, k):
        steps = np.array([c[0] for c in combo])
        offsets = np.array([c[1] for c in combo])
        labels, positions, residual = comb_kmeans(a, steps, offsets, tolerance)
        cost = float((residual ** 2).sum())
        fits = np.abs(residual).max() <= tolerance and no_holes(labels, positions, k, min_length)
        if best is None or (fits, -cost) > (best[0], -best[1]):
            best = (fits, cost, labels, positions)
    return best


def disentangle_algorithm(sequence_entangled, tolerance=1.5, vote_gaps=(0.15, 0.3, 0.6), top=8,
                          max_divisor=10, min_length=3, max_k=8):
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

    chosen = None  # (cost, labels, positions)
    for k in range(1, min(max_k, total // min_length) + 1):
        result = fit_k(a, k, tolerance, vote_gaps, top, max_divisor, min_length)
        if result is None:
            continue
        fits, cost, labels, positions = result
        if fits:
            chosen = (cost, labels, positions)
            break
        # No K fits yet: remember the closest one.
        if chosen is None or cost < chosen[0]:
            chosen = (cost, labels, positions)

    if chosen is None:
        labels, positions = np.arange(total), np.ones(total, dtype=int)
    else:
        _, labels, positions = chosen

    order = np.lexsort((positions, labels))
    return a[order], labels[order], labels
