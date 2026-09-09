# Numerical references and finals

## Golden references

Float64 references per problem (`golden_algorithm` at `golden_tol` from
`runner_scripts/problems.csv`), machine independent:

- `golden_<problem>_131072.csv` — wp final states, one row per trajectory, no header, from `runner_scripts/golden/generate_golden.jl` (kept unless `--force`); the ne ensemble is its first 1024 rows.
- `golden_<problem>_131072_retcodes.csv` — unconverged golden rows as `row,retcode` (1-based).

Per-machine finals are `data/key=<os>_<gpu>/package=<pkg>/finals/*.parquet`, named by each row's `finals` column.
