# Paper evidence notes

These notes map claims in `remaining_sections.tex` to repository evidence and
record comparisons that must not be overstated.

## Direct result sources

- Cached skip-policy table and bounded in-flight scheduler sweep:
  `notebooks/dynamic_rf_IDK_Cascade.ipynb` and
  `notebooks/dynamic_parallel_IDK_scheduler.ipynb`.
- Sequential live RF run: `notebooks/sequential_ronit_IDK_cascades.ipynb`.
- Pipelined sequential run:
  `notebooks/multiprocessor_sequential_idk_cascade.ipynb`.
- Parallel RF route with ResNet-152:
  `notebooks/multiprocessor_ronit_idk_cascades.ipynb`.
- Parallel three-model SVM run: `runs/real_cpu_mps_worker_results.json`.
- Parallel four-model SVM run: `runs/real_parallel_4_worker_results.json`.
- Fixed 30-FPS and 60-FPS runs: `runs/real_cpu_mps_30fps_results.json` and
  `runs/real_cpu_mps_60fps_results.json`.

## Router comparison reproduced from caches

The router table was recomputed from `matched_*`, `top_*`, and
`threshold07_*` artifacts with the hyperparameters already present in
`notebooks/parallel_3_IDK_cascades.ipynb`.

| Router | Router target accuracy | Final cached accuracy | Cached work ms/image |
|---|---:|---:|---:|
| SVM | 72.935% | 73.52% | 6.268 |
| Random Forest | 79.528% | 73.04% | 6.057 |
| Extra Trees | 70.645% | 73.88% | 6.353 |

Training rows: 2,777 (2,666 ResNet-34, 111 ResNet-50). Held-out eligible
router rows: 1,441 (1,374 ResNet-34, 67 ResNet-50). All 4,445 inputs rejected
by ResNet-18 still receive a downstream route in the cascade evaluation.

## Important qualifications

- The 30-FPS run uses ResNet-18/34/50. The 60-FPS run uses
  ResNet-18/34/152. They are not a controlled FPS-only pair.
- Live runs use both 0.7 and 0.9 confidence thresholds. Always show the
  threshold and model set beside a result.
- Logical MPS workers share one physical Apple M1 GPU. Do not describe them as
  dedicated processors or equal hardware partitions.
- The cached scheduler is a replay/simulation. Do not combine its timing with
  live end-to-end results in one speedup claim.
- Only one saved run exists for most live configurations. Do not claim
  statistical significance or hard real-time guarantees.
- The 60-FPS configuration fails to keep up. It ends the arrival interval with
  1,776 unfinished frames and requires 65.316 seconds to drain.
- Dataset artifact keys are disjoint across the three variants (zero pairwise
  overlap, 10,000 keys per variant).
- `paper173_sec7_*` uses calibrated non-IDK success semantics. Its 55.2% overall
  accuracy and 92.98% accuracy-on-successes are not interchangeable with the
  always-classify cascade accuracy in the main tables.

## Before submission

Run every principal configuration at least five times, report mean and 95%
confidence intervals, freeze the exact software versions used for those runs,
and repeat the 30/60-FPS sweep without changing the heavy model.

