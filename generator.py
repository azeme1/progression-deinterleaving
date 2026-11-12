import numpy as np


def generate_sample(sequence_count_config=(1, 3), sequence_size_config=(4, 20), noise_level_config=(0, 1.),
                    sequence_step_config=(10, 100), offset_config=None, return_params=False):
    """Sorted mix of K noisy arithmetic progressions.

    offset_config=None: series k is n * step_k + noise (starts at its step).
    offset_config=(lo, hi): series k is offset_k + n * step_k + noise, offset_k ~ U(lo, hi).
    """
    sequence_count = np.random.randint(sequence_count_config[0], sequence_count_config[1] + 1)
    sequence_step = np.random.uniform(sequence_step_config[0], sequence_step_config[1], size=(sequence_count, ))
    sequence_size = np.random.randint(sequence_size_config[0], sequence_size_config[1] + 1)
    noise = np.random.uniform(noise_level_config[0], noise_level_config[1], size=(sequence_count, sequence_size))
    if offset_config is None:
        offset = np.zeros(sequence_count)
    else:
        offset = np.random.uniform(offset_config[0], offset_config[1], size=(sequence_count, ))

    n = np.arange(1, sequence_size + 1)
    values = offset[:, None] + n[None, :] * sequence_step[:, None] + noise
    classes = np.repeat(np.arange(sequence_count), sequence_size)

    values = values.ravel()
    order = np.argsort(values)
    sequence_list, class_list = values[order], classes[order]

    if return_params:
        params = {"count": int(sequence_count), "size": int(sequence_size),
                  "steps": sequence_step.tolist(), "offsets": offset.tolist()}
        return sequence_list, class_list, params
    return sequence_list, class_list
