# progression-deinterleaving

Split a sorted mix of noisy arithmetic progressions back into its series, without knowing K, steps or noise.

## Layout

| Path | Content |
|---|---|
| `generator.py` | `generate_sample(..., offset_config=None)`; `offset_config=(lo, hi)` adds a random start per series |
| `human/disentangler.py` | v1: multiples of `a[j]` |
| `human/disentangler_v2.py` | v2: difference voting + adaptive per-series tolerance |
| `agent/` | baseline versions written by a neural network (LLM) |
| `hybrid/disentangler_kmeans.py` | human_v2 + k-lines refinement (Hungarian matching to slots), written by an agent from the author's idea |
| `main.py` | benchmark: success rate and timing, both modes |
| `tests/` | pytest, without and with offset |

Code in `human/` is written by a person. Code in `agent/` is a neural-network baseline for comparison.

## human v2

human_v2 is a generalization of human_v1. v1 checks `a[k] ≈ i * a[j]`, so it assumes the series starts at 0 and its step equals the first value `a[j]`. v2 checks differences `a[k] - a[j] ≈ m * d` and finds `d` by voting, so the start can be anything. With start = 0, the difference from 0 to `a[j]` is the step, which gives v1 back.

1. Differences `a[j] - a[i]` cancel the series start.
2. Each difference `D` votes `D / m`, `m = 1..10`, weight `1 / m`; histogram peaks are candidate steps.
3. A step is accepted by the longest chain `x -> x + d` (graph connected components); the series is removed and voting repeats, so K is found without being given.
4. Refinement: each value joins the series with the nearest predicted neighbour `P_k ± d_k`; noise sigma is re-estimated from in-series gaps and the tolerance only shrinks; unassigned values are voted again and may form new series.

## Run

```bash
pip install -r requirements.txt
python main.py                      # 50 seeds, 5 s budget per algorithm and mode
python main.py --seeds 200 --budget 30 --algorithms human_v2 agent_v3 --modes offset
python main.py --noise 0 --budget 300      # noise uniform in [0, noise), default 1
pytest -q
```

## Results (50 seeds, `--budget 300`)

Noise is uniform in [0, 1) per value (`python main.py`); `--noise 0` turns it off. Best value per column in bold.

| algorithm | noise, no_offset | noise, offset | no noise, no_offset | no noise, offset | mean ms |
|---|---|---|---|---|---|
| human_v1 | 36% | 0% | **100%** | 0% | **0.2–1.5** |
| human_v2 | 80% | 68% | 98% | **96%** | 10–11 |
| agent_v2 | 92% | **82%** | **100%** | **96%** | 215–220 |
| agent_v3 | 64% | 54% | 68% | 64% | 2100–2250 |
| agent_v4 | 82% | 78% | 94% | 94% | 4 |
| hybrid_kmeans | **94%** | 78% | **100%** | **96%** | 10–11 |

Success = exact one-to-one label match. The generator gives all series in a sample the same length (4–20), but no algorithm uses that.
agent_v3 takes about 2 s per sample, so the default 5 s budget stops it after a few seeds.
human_v2 failures without noise come mostly from near-integer step ratios (harmonics).

## Degradation with noise (`python main.py --noise a --budget 300`, 50 seeds)

Noise is uniform in [0, a) per value, its standard deviation is sigma = a / sqrt(12). Cells: no_offset / offset. Best per mode in bold.

| algorithm | a = 0 | a = 0.5 (sigma 0.14) | a = 1 (sigma 0.29) | a = 2 (sigma 0.58) | a = 4 (sigma 1.15) |
|---|---|---|---|---|---|
| human_v1 | **100%** / 0% | 42% / 0% | 36% / 0% | 28% / 0% | 22% / 0% |
| human_v2 | 98% / **96%** | 90% / 78% | 80% / 68% | 60% / 48% | 38% / 34% |
| agent_v2 | **100%** / **96%** | 94% / 90% | 92% / **82%** | **82%** / **76%** | 24% / 20% |
| agent_v3 | 68% / 64% | 64% / 58% | 64% / 54% | 58% / 56% | 22% / 24% |
| agent_v4 | 94% / 94% | 92% / 82% | 82% / 78% | 56% / 50% | 10% / 10% |
| hybrid_kmeans | **100%** / **96%** | **98%** / **92%** | **94%** / 78% | 80% / 54% | **60%** / **46%** |

hybrid_kmeans degrades slowest and stays fast. agent_v2 is best with offset at a = 1–2, but collapses at a = 4 and slows to about 1.2 s per sample.

## Equal vs different series lengths (noise a = 1, 50 seeds)

`generate_sample` gives all series one length. Here each series draws its own length in 4–20. Cells: no_offset / offset. Best per column in bold.

| algorithm | equal lengths | different lengths |
|---|---|---|
| human_v2 | 80% / 68% | 72% / 76% |
| agent_v2 | 92% / **82%** | **84%** / **92%** |
| agent_v3 | 64% / 54% | 58% / 56% |
| agent_v4 | 82% / 78% | 72% / 82% |
| hybrid_kmeans | **94%** / 78% | 78% / 86% |
