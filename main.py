import argparse
import time

import numpy as np

from generator import generate_sample
from human.disentangler import disentangle_algorithm as human_v1
from human.disentangler_v2 import disentangle_algorithm as human_v2
from agent.disentangler_v2 import disentangle_algorithm as agent_v2
from agent.disentangler_v3 import disentangle_algorithm as agent_v3
from agent.disentangler_v4 import disentangle_algorithm as agent_v4

ALGORITHMS = {
    "human_v1": human_v1,
    "human_v2": human_v2,
    "agent_v2": agent_v2,
    "agent_v3": agent_v3,
    "agent_v4": agent_v4,
}

MODES = {
    "no_offset": None,
    "offset": (0, 100),
}


def check_disentanglement(class_entangled, class_entangled_predicted):
    """True iff predicted labels are a one-to-one relabeling of the true ones."""
    pairs = set(zip(np.asarray(class_entangled).tolist(), np.asarray(class_entangled_predicted).tolist()))
    return len(pairs) == len({e for e, _ in pairs}) == len({p for _, p in pairs})


def run(algorithm, seeds, offset_config, budget=None, **generator_kwargs):
    """Success rate and timing; stops early once `budget` seconds are spent."""
    ok, times = 0, []
    for seed in seeds:
        if budget is not None and sum(times) > budget:
            break
        np.random.seed(seed)
        x, y = generate_sample(offset_config=offset_config, **generator_kwargs)
        t0 = time.perf_counter()
        try:
            _, _, p = algorithm(x)
            passed = check_disentanglement(y, p)
        except Exception:
            passed = False
        times.append(time.perf_counter() - t0)
        ok += passed
    return ok / len(times), len(times), 1000 * np.mean(times), 1000 * np.max(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--budget", type=float, default=5.0, help="seconds per algorithm and mode")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    args = parser.parse_args()

    seeds = range(args.seeds)
    print(f"{'algorithm':10s} {'mode':10s} {'success':>8s} {'runs':>5s} {'mean ms':>9s} {'max ms':>9s}")
    for name in args.algorithms:
        for mode in args.modes:
            rate, runs, mean_ms, max_ms = run(ALGORITHMS[name], seeds, MODES[mode], args.budget)
            print(f"{name:10s} {mode:10s} {rate:8.1%} {runs:5d} {mean_ms:9.2f} {max_ms:9.2f}", flush=True)
