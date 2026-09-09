# Smoke and reruns

Every command runs from the repo root under the suite interpreter, one package at a time on the machine whose key it writes; sync the store afterwards (`docs/remote-store.md`). jax runs on a Linux machine under its own key.

## 1. Smoke

One run per package, tiny n, the two problems and the algorithms every package family shares at one step and one tolerance:

```
python bench.py run --set perf,states,golden_grid -p <pkg> -s lorenz,lorenz96 -g tsit5,kvaerno3,euler,classical-rk4,cash-karp-54 --dt 0.0009765625 --tol 1e-5 -n 128
```

Packages: cubie, cubie_mlir, jax, pytorch, myokit_cuda, cpp, julia_gpu, julia_cpu. Optimize lines run at the set's own n and write no row. Then, with `<rev>` from `git rev-parse --short HEAD`:

```
python runner_scripts/store.py query "SELECT package, count(*) AS rows, sum(CASE WHEN isnan(min_ms) THEN 1 ELSE 0 END) AS failed FROM results WHERE n = 128 AND suite_rev = '<rev>' GROUP BY 1"
```

A `reason` naming a controller, precision or algorithm the package does not run is a correct row; a `reason` carrying an exception from the suite's own code is a defect to report.

## 2. Reruns per key

`--resume` skips trials with every row present, `--no-overwrite` also retakes NaN rows. The goldens stay; every other key repeats the step. Run the sets together; a solve two sets share is recorded once with the finals and transfers of both.

```
python bench.py run --set perf,states,golden_grid --resume
```

## 3. After each step

```
python analyses/timing.py --x n --set perf
python analyses/timing.py --x states --set states
python analyses/timing.py --x error --set golden_grid
python analyses/agreement.py --set golden_grid
```

Commit the key's `data/` tree in a data PR.
