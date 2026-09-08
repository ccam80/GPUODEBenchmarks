# Numerical references and finals

## Golden references

Float64 references per problem (`golden_algorithm` at `golden_tol` from
`runner_scripts/problems.csv`), machine independent:

- `golden_<problem>_131072.csv` — wp ensemble final states, one row per
  trajectory, no header; `runner_scripts/golden/generate_golden.jl`, kept
  unless `--force`. The ne ensemble is its first 1024 rows.
- `golden_<problem>_131072_retcodes.csv` — unconverged golden rows as
  `row,retcode` (1-based).

## Per-machine finals

`<os>_<gpu>/<problem>/<file>.csv` — one package's final states at the N = 32768
timing point (row per trajectory, no header): `cubie_unadaptive.csv`,
`cubie_adaptive.csv` (`cubie_mlir_*` for MLIR), `jax.csv`, `pytorch.csv`,
`julia_fixed.csv`, `julia_adaptive.csv`, `mpgos.csv`, `myokit_cuda.csv`.
`compare_numerical_results.py` compares each pair with `numpy.allclose`
(`rtol=1e-4`, `atol=1e-6`) per machine; exit 3 means fewer than two files.
