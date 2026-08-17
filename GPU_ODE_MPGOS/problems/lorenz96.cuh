#ifndef PROBLEM_LORENZ96_CUH
#define PROBLEM_LORENZ96_CUH

// Lorenz 96: cyclic 40-state coupling, the forcing F swept across the ensemble.
#define PROBLEM_NAME "lorenz96"
#define PROBLEM_SD 40
#define PROBLEM_NCP 1
#define PROBLEM_DURATION 1.0
#define PROBLEM_SWEEP_MIN 0.0
#define PROBLEM_SWEEP_MAX 16.0
#define PROBLEM_SWEEP_LOG 0

template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	const int n = PROBLEM_SD;
	for (int i = 0; i < n; i++)
	{
		int ip1 = (i + 1) % n;
		int im1 = (i + n - 1) % n;
		int im2 = (i + n - 2) % n;
		F[i] = (X[ip1] - X[im2]) * X[im1] - X[i] + cPAR[0];
	}
}

// Initial state shared by every trajectory: uniform 8 with X[0] perturbed to 9.
template <class Precision>
__forceinline__ __host__ void ProblemInitialState(Precision* X)
{
	for (int i = 0; i < PROBLEM_SD; i++)
		X[i] = 8.0;
	X[0] = 9.0;
}

#endif
