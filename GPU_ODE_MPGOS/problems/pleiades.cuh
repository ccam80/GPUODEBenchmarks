#ifndef PROBLEM_PLEIADES_CUH
#define PROBLEM_PLEIADES_CUH

// Pleiades: seven-body gravitation, X = (x, y, x', y') per star, m1 swept.
#define PROBLEM_NAME "pleiades"
#define PROBLEM_SD 28
#define PROBLEM_NCP 1
#define PROBLEM_DURATION 3.0
#define PROBLEM_SWEEP_MIN 0.5
#define PROBLEM_SWEEP_MAX 2.0
#define PROBLEM_SWEEP_LOG 0

template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	for (int i = 0; i < 14; i++)
		F[i] = X[i + 14];
	for (int i = 0; i < 7; i++)
	{
		Precision sumx = 0.0;
		Precision sumy = 0.0;
		for (int j = 0; j < 7; j++)
		{
			if (j == i) continue;
			Precision mj = (j == 0) ? cPAR[0] : (Precision)(j + 1);
			Precision dx = X[j] - X[i];
			Precision dy = X[j + 7] - X[i + 7];
			Precision rij = dx*dx + dy*dy;
			Precision rij32 = rij * sqrt(rij);
			sumx += mj * dx / rij32;
			sumy += mj * dy / rij32;
		}
		F[i + 14] = sumx;
		F[i + 21] = sumy;
	}
}

// Initial state shared by every trajectory.
template <class Precision>
__forceinline__ __host__ void ProblemInitialState(Precision* X)
{
	const double x0[PROBLEM_SD] = {
		3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0,
		3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5,
		0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0 };
	for (int i = 0; i < PROBLEM_SD; i++)
		X[i] = (Precision)x0[i];
}

#endif
