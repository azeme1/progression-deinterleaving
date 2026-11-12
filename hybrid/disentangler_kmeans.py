"""human_v2 + k-lines refinement.

K and the initial split come from human_v2. Each series is then a line
x = c_k + n * d_k; values are matched one-to-one to slots c_k + n * d_k
(Hungarian), and each line is refit by least squares, until stable.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from human.disentangler_v2 import disentangle as disentangle_v2


def _fit_lines(x, labels):
    c, d = [], []
    for k in range(labels.max() + 1):
        P = np.sort(x[labels == k])
        g = np.diff(P)
        step = np.median(g[g > 0.5 * np.median(g)]) if len(g) else 1.0
        n = np.round((P - P[0]) / step)
        if np.ptp(n) > 0:
            step, start = np.polyfit(n, P, 1)
        else:
            start = P[0]
        c.append(start)
        d.append(step)
    return np.array(c), np.array(d)


LIKELIHOOD = True


SHRINK = 10


def _line_sigma(x, labels, c, d, floor):
    """Per series residual std from its own line, shrunk to the pooled std (weight SHRINK)."""
    res = []
    for k in range(len(c)):
        P = x[labels == k]
        res.append(P - (c[k] + np.round((P - c[k]) / d[k]) * d[k]))
    allr = np.concatenate(res)
    pooled = max(1.4826 * np.median(np.abs(allr - np.median(allr))), floor)
    sig = np.empty(len(c))
    for k, r in enumerate(res):
        v = np.mean((r - np.median(r)) ** 2) if len(r) else 0.0
        sig[k] = np.sqrt((len(r) * v + SHRINK * pooled ** 2) / (len(r) + SHRINK))
    return np.maximum(sig, floor)


def refine(x, labels, iters=20, pad=2, floor=0.05):
    x = np.asarray(x, float)
    for _ in range(iters):
        c, d = _fit_lines(x, labels)
        if not np.all(np.isfinite(d)) or np.any(d <= 0.5):
            break
        slot_k, slot_v = [], []
        for k in range(len(c)):
            n = np.round((x[labels == k] - c[k]) / d[k])
            ns = np.arange(n.min() - pad, n.max() + pad + 1)
            slot_k.append(np.full(len(ns), k))
            slot_v.append(c[k] + ns * d[k])
        slot_k, slot_v = np.concatenate(slot_k), np.concatenate(slot_v)
        if len(slot_v) < len(x):
            break
        cost = (x[:, None] - slot_v[None, :]) ** 2
        if LIKELIHOOD:
            sig = _line_sigma(x, labels, c, d, floor)[slot_k]
            cost = cost / (2 * sig[None, :] ** 2) + np.log(sig)[None, :]
        r, s = linear_sum_assignment(cost)
        new = labels.copy()
        new[r] = slot_k[s]
        if np.array_equal(new, labels) or np.bincount(new, minlength=len(c)).min() < 2:
            break
        labels = new
    return labels


def disentangle(x):
    x = np.asarray(x, float)
    labels = disentangle_v2(x)
    if labels.max() > 0:
        labels = refine(x, labels)
    _, first = np.unique(labels, return_index=True)
    lut = np.empty(labels.max() + 1, int)
    lut[labels[np.sort(first)]] = np.arange(len(first))
    return lut[labels]


def disentangle_algorithm(sequence_entangled):
    a = np.asarray(sequence_entangled, float)
    labels = disentangle(a)
    order = np.lexsort((a, labels))
    return a[order], labels[order], labels
