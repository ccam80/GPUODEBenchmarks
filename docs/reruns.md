# Smoke and reruns

Every command runs from the repo root under the suite interpreter, one package at a time on a machine whose key it writes; sync the store afterwards (`docs/remote-store.md`).

## 1. Smoke

One run per package, tiny n, the two problems and the algorithms every package family shares at one step and one tolerance:

```
python bench.py run --set perf,states,golden_grid -p <pkg> -s lorenz,lorenz96 -g tsit5,kvaerno3,euler,classical-rk4,cash-karp-54 --dt 0.0009765625 --tol 1e-5 -n 128
```

Run it for cubie, cubie_mlir, jax (WSL), pytorch, myokit_cuda, cpp, julia_gpu and julia_cpu. The first julia_gpu leg precompiles the kernel package. Optimize lines run at the set's own n. Read the rows back before going on:

```
python runner_scripts/store.py query "SELECT package, count(*) AS rows, sum(CASE WHEN isnan(min_ms) THEN 1 ELSE 0 END) AS failed FROM results WHERE n = 128 GROUP BY 1"
```

Every failed row carries a `reason`; fix the cause and rerun the package with `--no-overwrite` before the reruns below.

## 2. Reruns per key

Order of cost. `--resume` skips trials that already have every row; `--no-overwrite` also retakes NaN rows.

| step | command | note |
|---|---|---|
| nand_gate golden | `python runner_scripts/store.py clear '{"package": "julia_cpu", "problem": "nand_gate", "precision": "float64"}'` then `python bench.py run --set golden -s nand_gate` | clears the DFBDF row first; RadauIIA5 takes about 16 h |
| cpp | `python bench.py run --set perf,states,golden_grid -p cpp` | no cpp rows exist |
| cubie, cubie_mlir | `python bench.py run --set perf,states,golden_grid -p cubie,cubie_mlir` | no cubie rows exist |
| golden_grid | `python bench.py run --set golden_grid --resume` | every package; the store holds no finals but the goldens |
| julia_cpu | `python bench.py run --set golden_grid -p julia_cpu --resume` | 1024-point prefix on every problem |
| jax adaptive perf | `python bench.py run --set perf -p jax --mode adaptive --no-overwrite` | WSL |
| jax kvaerno3 | `python bench.py run --set perf,golden_grid -p jax -g kvaerno3 --no-overwrite` | WSL; linux key only |
| nand_gate | `python bench.py run --set perf,states,golden_grid -s nand_gate --no-overwrite` | every package that implements it |

The 2060 key repeats every step but the golden.

## 3. After each step

```
python analyses/timing.py --x n --set perf
python analyses/timing.py --x states --set states
python analyses/timing.py --x error --set golden_grid
python analyses/agreement.py --set golden_grid
```

Commit the key's `data/` tree in a data PR.
