# Smoke and reruns

Every command runs from the repo root under the suite interpreter, one package at a time on a machine whose key it writes; sync the store afterwards (`docs/remote-store.md`).

## 1. Smoke

One run per package, tiny n, the two problems and the algorithms every package family shares at one step and one tolerance:

```
python bench.py run --set perf,states,golden_grid -p <pkg> -s lorenz,lorenz96 -g tsit5,kvaerno3,euler,classical-rk4,cash-karp-54 --dt 0.0009765625 --tol 1e-5 -n 128
```

Packages: cubie, cubie_mlir, jax (WSL), pytorch, myokit_cuda, cpp, julia_gpu, julia_cpu. Optimize lines run at the set's own n. Then:

```
python runner_scripts/store.py query "SELECT package, count(*) AS rows, sum(CASE WHEN isnan(min_ms) THEN 1 ELSE 0 END) AS failed FROM results WHERE n = 128 GROUP BY 1"
```

A `reason` naming a controller, precision or algorithm the package does not run is a correct row; a `reason` carrying an exception from the suite's own code is a defect: fix it and rerun that package with `--no-overwrite` before step 2.

## 2. Reruns per key

In cost order; `--resume` skips trials with every row present, `--no-overwrite` also retakes NaN rows. The 2060 key repeats every step but the golden.

```
python runner_scripts/store.py clear '{"package": "julia_cpu", "problem": "nand_gate", "precision": "float64"}'
python bench.py run --set golden -s nand_gate                                        # about 16 h
python bench.py run --set perf,states,golden_grid -p cpp
python bench.py run --set perf,states,golden_grid -p cubie,cubie_mlir
python bench.py run --set golden_grid --resume
python bench.py run --set golden_grid -p julia_cpu --resume
python bench.py run --set perf -p jax --mode adaptive --no-overwrite                # WSL
python bench.py run --set perf,golden_grid -p jax -g kvaerno3 --no-overwrite        # WSL, linux key
python bench.py run --set perf,states,golden_grid -s nand_gate --no-overwrite
```

## 3. After each step

```
python analyses/timing.py --x n --set perf
python analyses/timing.py --x states --set states
python analyses/timing.py --x error --set golden_grid
python analyses/agreement.py --set golden_grid
```

Commit the key's `data/` tree in a data PR.
