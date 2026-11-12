import time

import numpy as np
import pytest

from generator import generate_sample
from main import ALGORITHMS, MODES, check_disentanglement, run
from human.disentangler_v2 import disentangle_algorithm

SEEDS = range(30)


def test_example():
    x = [25.5, 40.0, 51.0, 76.5, 80.0, 102.0, 120.0, 160.0]
    _, _, p = disentangle_algorithm(x)
    assert p.tolist() == [0, 1, 0, 0, 1, 0, 1, 1]


@pytest.mark.parametrize("mode", MODES)
def test_generator(mode):
    np.random.seed(0)
    x, y, params = generate_sample(offset_config=MODES[mode], return_params=True)
    assert np.all(np.diff(x) >= 0)
    assert len(x) == len(y) == params["count"] * params["size"]


def test_check_rejects_merge():
    assert not check_disentanglement([0, 0, 1, 1], [0, 0, 0, 0])
    assert check_disentanglement([0, 0, 1, 1], [5, 5, 3, 3])


@pytest.mark.parametrize("mode,min_rate", [("no_offset", 0.7), ("offset", 0.5)])
def test_human_v2_rate(mode, min_rate):
    rate, runs, _, _ = run(disentangle_algorithm, SEEDS, MODES[mode])
    assert runs == len(SEEDS) and rate >= min_rate


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_human_v2_single_series(seed, mode):
    np.random.seed(seed)
    x, y = generate_sample(sequence_count_config=(1, 1), offset_config=MODES[mode])
    _, _, p = disentangle_algorithm(x)
    assert check_disentanglement(y, p)


@pytest.mark.parametrize("name", ALGORITHMS)
def test_speed(name):
    np.random.seed(0)
    x, _ = generate_sample()
    t0 = time.perf_counter()
    ALGORITHMS[name](x)
    assert time.perf_counter() - t0 < 5.0
