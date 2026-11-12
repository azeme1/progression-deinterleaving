import numpy as np


def generate_sample(sequence_count_config=(1, 3), sequence_size_config=(4, 10), noise_level_config=(0, 1.), sequence_step_config=(10, 100)):
    """Dataset Generation Constraints
    You must write a utility script that dynamically creates a test dataset under the following parameters:
    • Number of Series (K): At launch, the script must randomly choose an integer between 1 and 3.
    • Step Sizes: Each hidden series must have a unique constant baseline step size (difference) chosen
    randomly from a continuous range of 10.0 to 100.0. These step sizes can be floating-point numbers.
    • Noise Floor: Add a minor Gaussian noise component to the steps to simulate measurement uncertainty.
    • The Interleaving Mechanism: Your generator must pool all generated progression values together and
    strictly sort the final combined array. This destroys the original temporal order and interleaves the
    streams purely by numerical magnitude.
    """

    sequence_count = np.random.randint(sequence_count_config[0], sequence_count_config[1] + 1)
    sequence_step = np.random.uniform(sequence_step_config[0], sequence_step_config[1], size=(sequence_count, ))
    sequence_size = np.random.randint(sequence_size_config[0], sequence_size_config[1] + 1)
    noise = np.random.uniform(noise_level_config[0], noise_level_config[1], size=(sequence_count, sequence_size))

    sequence_list_original = []
    class_list_original = []
    for k in range(sequence_count):
        sequence_list_original.append(np.arange(1, sequence_size + 1) * sequence_step[k] + noise[k])
        class_list_original.append(np.array([k] * sequence_size))

    sequence_list_original = np.concatenate(sequence_list_original)
    class_list_original = np.concatenate(class_list_original)

    order = np.argsort(sequence_list_original)
    sequence_list = sequence_list_original[order]
    class_list = class_list_original[order]

    return sequence_list, class_list
