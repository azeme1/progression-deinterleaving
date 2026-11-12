import numpy as np


def _fit(ns, xs):
    """Fit x = d * n + c."""
    d, c = np.polyfit(ns, xs, 1)
    return d, c


def _cost(a, labels, positions, k):
    """Sum of squared residuals of every series around its own fitted line."""
    total = 0.0
    for s in range(k):
        mask = labels == s
        d, c = _fit(positions[mask], a[mask])
        total += float(((a[mask] - (positions[mask] * d + c)) ** 2).sum())
    return total


def _search(a, k, tolerance, min_length, max_nodes):
    """Depth-first search for ways to split `a` into `k` progressions of any length >= min_length.

    A new series always starts at the smallest unassigned value. Each next term
    is predicted from the line fitted to the terms found so far, and every
    unassigned value within `tolerance` of the prediction is tried, closest
    first. A series may also end once it has min_length terms.
    Returns the complete split with the lowest residual cost, or None.
    """
    labels = np.full(len(a), -1)
    positions = np.zeros(len(a), dtype=int)
    best = {"cost": np.inf, "labels": None, "positions": None}
    nodes = [0]

    def extend(series, ns, xs):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return

        n = len(ns) + 1
        free = np.flatnonzero(labels == -1)
        if n == 2:
            # The start is unknown, so any later value can be the second term.
            candidates = free[a[free] > xs[0]]
        else:
            d, c = _fit(ns, xs)
            distance = np.abs(a[free] - (n * d + c))
            candidates = free[np.argsort(distance)]
            candidates = candidates[np.sort(distance) <= tolerance]
        for idx in candidates:
            labels[idx], positions[idx] = series, n
            extend(series, ns + [n], xs + [a[idx]])
            labels[idx] = -1

        # End this series here.
        if len(ns) < min_length:
            return
        if len(free) == 0:
            if series + 1 == k:
                cost = _cost(a, labels, positions, k)
                if cost < best["cost"]:
                    best.update(cost=cost, labels=labels.copy(), positions=positions.copy())
            return
        if series + 1 < k:
            first = free[0]
            labels[first], positions[first] = series + 1, 1
            extend(series + 1, [1], [a[first]])
            labels[first] = -1

    labels[0], positions[0] = 0, 1
    extend(0, [1], [a[0]])

    if best["labels"] is None:
        return None
    return best["labels"], best["positions"]


def disentangle_algorithm(sequence_entangled, tolerance=3.0, min_length=3, max_nodes=20000):
    """Split a sorted mix of arithmetic progressions back into its series.

    Each hidden series is x_n = start + n * step + noise for n = 1..L_k.
    The input is every value pooled and sorted.

    The number of series K is tried from 1 upwards, and the first K that splits cleanly wins.

    Returns, matching the original function:
        sequence_disentangled: values grouped by series, each in term order.
        class_disentangled: the series label of each value in sequence_disentangled.
        class_entangled_predicted: the series label of each value in the input order.
    """
    a = np.asarray(sequence_entangled, dtype=float)
    total = len(a)
    if total == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    result = None
    for k in range(1, total // min_length + 1):
        result = _search(a, k, tolerance, min_length, max_nodes)
        if result is not None:
            break

    if result is None:
        # Nothing fits: every value is its own series.
        labels, positions = np.arange(total), np.ones(total, dtype=int)
    else:
        labels, positions = result

    order = np.lexsort((positions, labels))
    return a[order], labels[order], labels
