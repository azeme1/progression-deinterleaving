import numpy as np


def _line(ns, xs):
    """Fit x = step * n + offset. With one point, assume offset = 0."""
    if len(ns) == 1:
        return xs[0] / ns[0], 0.0
    return np.polyfit(ns, xs, 1)


def _look_ahead(a, free, ns, xs, length, depth, threshold):
    """Mean distance of the next `depth` predicted terms to the closest free values.

    A wrong pick bends the series' line, so the following terms stop landing on values.
    """
    free = free.copy()
    ns, xs = list(ns), list(xs)
    total, steps = 0.0, 0
    for _ in range(depth):
        if len(ns) >= length:
            break
        step, offset = _line(ns, xs)
        n = len(ns) + 1
        dist = np.where(free, np.abs(a - (n * step + offset)), np.inf)
        k = np.argmin(dist)
        total += min(dist[k], threshold)
        steps += 1
        if dist[k] >= threshold:
            break
        free[k] = False
        ns.append(n)
        xs.append(a[k])
    return total / steps if steps else 0.0


def _cost(a, free, k, ns, xs, last, step, length, threshold, weights):
    """Score a candidate value a[k] as the next term of the series (lower is better).

    Features:
        pred:  distance to the next term predicted from the last term and the step so far.
        fit:   largest residual of the series' line after adding the candidate.
        ahead: how well the following terms still land on free values.
    """
    i = len(ns) + 1
    pred = abs(a[k] - (last + step))

    ns2, xs2 = ns + [i], xs + [a[k]]
    line_step, line_offset = _line(ns2, xs2)
    fit = np.abs(np.array(xs2) - (np.array(ns2) * line_step + line_offset)).max() if len(ns2) > 2 else 0.0

    without_k = free.copy()
    without_k[k] = False
    ahead = _look_ahead(a, without_k, ns2, xs2, length, weights["ahead_depth"], threshold)

    return weights["pred"] * pred + weights["fit"] * fit + weights["ahead"] * ahead


def _greedy(a, threshold, length, weights):
    s = []
    c = {}
    free = np.ones(len(a), bool)  # values not yet claimed by any series

    for j in range(len(a)):
        if not free[j]:
            continue

        # a[j] is the smallest unclaimed value, so it is the first term of a new series.
        free[j] = False
        s += [float(a[j])]
        c[float(a[j])] = j
        first = last = a[j]
        step = a[j]
        ns, xs = [1], [a[j]]

        for i in range(2, length + 1):
            if i == 2:
                # The start is unknown, so any later value can be the second term.
                # Pick the one whose whole chain lands best on free values.
                near = np.flatnonzero(free & (a > first))
                if len(near) == 0:
                    break
                costs = [_look_ahead(a, free & (np.arange(len(a)) != k), [1, 2], [first, a[k]],
                                     length, length, threshold) for k in near]
            else:
                # Candidates: free values close to the predicted next term.
                near = np.flatnonzero(free & (np.abs(a - (last + step)) < threshold))
                if len(near) == 0:
                    break
                costs = [_cost(a, free, k, ns, xs, last, step, length, threshold, weights) for k in near]
            k = near[int(np.argmin(costs))]

            free[k] = False
            s += [float(a[k])]
            c[float(a[k])] = j
            last = a[k]
            step = (last - first) / (i - 1)
            ns.append(i)
            xs.append(a[k])

    return s, c


DEFAULT_WEIGHTS = {"pred": 1.0, "fit": 4.0, "ahead": 8.0, "ahead_depth": 2}


def disentangle_algorithm(sequence_entangled, threshold=2.5, weights=None):
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    a = np.array(sequence_entangled)
    n = len(a)

    # All series share one length, and it must divide the number of values.
    # Try lengths from longest to shortest and keep the first where every series is complete.
    for length in [L for L in range(n, 1, -1) if n % L == 0] or [1]:
        s, c = _greedy(a, threshold, length, weights)
        sizes = np.unique(list(c.values()), return_counts=True)[1]
        if (sizes == length).all():
            break

    sequence_disentangled = np.array(s)
    class_disentangled = np.array([c[item] for item in sequence_disentangled])
    class_etangled_predicted = np.array([c[item] for item in sequence_entangled])

    return sequence_disentangled, class_disentangled, class_etangled_predicted
