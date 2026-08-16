#ifndef PROBLEM_LORENZ_CUH
#define PROBLEM_LORENZ_CUH

// Lorenz: sigma = 10, beta = 8/3, rho swept across the ensemble.
#define PROBLEM_NAME "lorenz"
#define PROBLEM_SD 3
#define PROBLEM_NCP 1
#define PROBLEM_DURATION 1.0
#define PROBLEM_SWEEP_MIN 0.0
#define PROBLEM_SWEEP_MAX 21.0
#define PROBLEM_SWEEP_LOG 0

template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	F[0] = 10.0*( X[1]-X[0] );
	F[1] = cPAR[0]*X[0] - X[1] - X[0]*X[2];
	F[2] = X[0]*X[1] - (8.0/3.0) * X[2];
}

// Initial state shared by every trajectory.
template <class Precision>
__forceinline__ __host__ void ProblemInitialState(Precision* X)
{
	X[0] = 1.0;
	X[1] = 0.0;
	X[2] = 0.0;
}

#endif
