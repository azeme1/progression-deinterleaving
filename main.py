import numpy as np

from human.disentangler import disentangle_algorithm
from generator import generate_sample


def check_disentanglement(class_entangled, class_entangled_predicted):

    mapping = {}
    for e, d in zip(class_entangled, class_entangled_predicted):
        if e in mapping:
            if mapping[e] == d:
                pass
            else:
                return False
        else:
            mapping[e] = d

    return True 

if __name__ == "__main__":
    np.random.seed(777)

    sequence_entangle, class_entangled = generate_sample()

    sequence_disentangle, class_disentangled, class_entangled_predicted = disentangle_algorithm(sequence_entangle)

    result = check_disentanglement(class_entangled, class_entangled_predicted)

    print("Disentanglement check result:", result)
