# CuBIE vs Myokit-CUDA: Fabbri-Linder

- CellML: `C:\local_working_projects\cubie\tests\fixtures\cellml\Fabbri_Linder.cellml`
- Precision: float32
- Algorithm: forward Euler in both implementations
- Step: `1e-05` s x `1000` (`0.01` s total)
- Trajectories: `512`
- States: `35` (mapped by `component.variable` to `component_variable`)
- Myokit-only CellML metadata repairs: `4`
- CUDA block sizes: Myokit-CUDA `128`, CuBIE `64`
- Timing: minimum of `100` synchronized repeats; compilation and one warmup solve excluded.

## Timing

| Implementation | Minimum (ms) |
|---|---:|
| Myokit-CUDA | 5.444400 |
| CuBIE | 4.553600 |

CuBIE / Myokit-CUDA: `0.836382`.

CuBIE speedup: `1.195625x`.

## Accuracy

| Metric | Value |
|---|---:|
| Maximum absolute error | 5.960464478e-08 |
| Mean absolute error | 6.042163129e-09 |
| RMS error | 1.670976459e-08 |
| Maximum relative error | 7.129495069e-07 |

`numpy.allclose(rtol=1e-06, atol=1e-08)`: `True`.
