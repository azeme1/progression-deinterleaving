import numpy as np

def disentangle_algorithm(sequence_entangled):

    s = []
    c = {}
    a = np.array(sequence_entangled)

    check_set = set(a)
    threshold = np.diff(a).min()

    for j in range(len(a)):
        if check_set == set(s):
            break

        if len(s) > 0 and np.abs(a[j] - np.array(s)).min() < threshold:
            continue

        for i in range(1, len(a) + 1):
            cost = np.abs(a - i * a[j])
            k = np.argmin(cost)
            m = cost[k]

            if m < threshold:
                s += [float(a[k])]
                c[float(a[k])] = j

    sequence_disentangled = np.array(s)
    class_disentangled = np.array([c[item] for item in sequence_disentangled])
    class_etangled_predicted = np.array([c[item] for item in sequence_entangled])

    return sequence_disentangled, class_disentangled, class_etangled_predicted


