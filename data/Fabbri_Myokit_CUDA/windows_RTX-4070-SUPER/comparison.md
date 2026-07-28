# CuBIE vs Myokit-CUDA: Fabbri-Linder scaling

Both implementations use float32 forward Euler with `dt=1e-05` for 1000 steps (0.01 seconds total). Compilation and one warmup solve per trajectory count are excluded; each row is the minimum of 100 synchronized repeats. CUDA block sizes are 128 for Myokit-CUDA and 64 for CuBIE.

| N | Myokit-CUDA (ms) | CuBIE (ms) | CuBIE / Myokit-CUDA | CuBIE speedup | Max abs error | RMS error | allclose |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 5.444400 | 4.553600 | 0.836382 | 1.195625x | 5.960464478e-08 | 1.670976459e-08 | True |
| 2048 | 5.579900 | 4.273600 | 0.765892 | 1.305667x | 5.960464478e-08 | 1.670976459e-08 | True |
| 8192 | 6.393200 | 4.382600 | 0.685510 | 1.458769x | 5.960464478e-08 | 1.670976459e-08 | True |
| 32768 | 14.324700 | 6.315300 | 0.440868 | 2.268253x | 5.960464478e-08 | 1.670976459e-08 | True |
| 131072 | 51.036400 | 19.980900 | 0.391503 | 2.554259x | 5.960464478e-08 | 1.670976459e-08 | True |
