// Sustained-load generator for GPU clock calibration; built and driven by
// calibrate_clocks.py. To build by hand:
//   nvcc -O3 -arch=sm_75 clock_burn.cu -lcublas -o clock_burn
//   ./clock_burn <seconds> [matrix_n]
// A small matrix_n (e.g. 256) gives a light load with launch gaps instead.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(x)                                                             \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at line %d\n",                      \
                    cudaGetErrorString(e), __LINE__);                          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char **argv) {
    double seconds = (argc > 1) ? atof(argv[1]) : 900.0;
    int n = (argc > 2) ? atoi(argv[2]) : 4096;

    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> h((size_t)n * n);
    for (size_t i = 0; i < h.size(); ++i) h[i] = 0.5f + 1e-6f * (float)(i % 97);

    float *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));
    CUDA_OK(cudaMalloc(&dC, bytes));
    CUDA_OK(cudaMemcpy(dA, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dC, 0, bytes));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    const float alpha = 1.0f, beta = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    long long iters = 0;
    double elapsed = 0.0;

    printf("clock_burn: n=%d, target %.0f s\n", n, seconds);
    fflush(stdout);

    while (elapsed < seconds) {
        // Several GEMMs per sync so the GPU never idles between launches.
        for (int k = 0; k < 8; ++k) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n,
                        dB, n, &beta, dC, n);
            ++iters;
        }
        CUDA_OK(cudaDeviceSynchronize());
        elapsed = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
    }

    double gflop = 2.0 * (double)n * n * n * (double)iters / 1e9;
    printf("clock_burn: %lld GEMMs in %.1f s = %.1f GFLOP/s\n", iters, elapsed,
           gflop / elapsed);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
