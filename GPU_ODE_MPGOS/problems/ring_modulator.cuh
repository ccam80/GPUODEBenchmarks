#ifndef PROBLEM_RING_MODULATOR_CUH
#define PROBLEM_RING_MODULATOR_CUH

// Ring modulator (Test Set for IVP Solvers II-3); Cs is swept.
#define PROBLEM_NAME "ring_modulator"
#define PROBLEM_SD 15
#define PROBLEM_NCP 1
#define PROBLEM_DURATION 1.0e-3
#define PROBLEM_SWEEP_MIN 2.0e-13
#define PROBLEM_SWEEP_MAX 2.0e-9
#define PROBLEM_SWEEP_LOG 1

#define RM_C 1.6e-8
#define RM_CP 1.0e-8
#define RM_LH 4.45
#define RM_LS1 0.002
#define RM_LS2 5.0e-4
#define RM_LS3 5.0e-4
#define RM_GAMMA 40.67286402e-9
#define RM_R 25000.0
#define RM_RP 50.0
#define RM_RG1 36.3
#define RM_RG2 17.3
#define RM_RG3 17.3
#define RM_RI 50.0
#define RM_RC 600.0
#define RM_DELTA 17.7493332
#define RM_W1 6283.185307179586
#define RM_W2 62831.85307179586

template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	Precision cs = cPAR[0];
	Precision uin1 = 0.5*sin(RM_W1*T);
	Precision uin2 = 2.0*sin(RM_W2*T);
	Precision ud1 = X[2] - X[4] - X[6] - uin2;
	Precision ud2 = -X[3] + X[5] - X[6] - uin2;
	Precision ud3 = X[3] + X[4] + X[6] + uin2;
	Precision ud4 = -X[2] - X[5] + X[6] + uin2;
	Precision q1 = RM_GAMMA*(exp(RM_DELTA*ud1) - 1.0);
	Precision q2 = RM_GAMMA*(exp(RM_DELTA*ud2) - 1.0);
	Precision q3 = RM_GAMMA*(exp(RM_DELTA*ud3) - 1.0);
	Precision q4 = RM_GAMMA*(exp(RM_DELTA*ud4) - 1.0);

	F[0]  = (X[7] - 0.5*X[9] + 0.5*X[10] + X[13] - X[0]/RM_R) / RM_C;
	F[1]  = (X[8] - 0.5*X[11] + 0.5*X[12] + X[14] - X[1]/RM_R) / RM_C;
	F[2]  = (X[9] - q1 + q4) / cs;
	F[3]  = (-X[10] + q2 - q3) / cs;
	F[4]  = (X[11] + q1 - q3) / cs;
	F[5]  = (-X[12] - q2 + q4) / cs;
	F[6]  = (-X[6]/RM_RP + q1 + q2 - q3 - q4) / RM_CP;
	F[7]  = -X[0] / RM_LH;
	F[8]  = -X[1] / RM_LH;
	F[9]  = (0.5*X[0] - X[2] - RM_RG2*X[9]) / RM_LS2;
	F[10] = (-0.5*X[0] + X[3] - RM_RG3*X[10]) / RM_LS3;
	F[11] = (0.5*X[1] - X[4] - RM_RG2*X[11]) / RM_LS2;
	F[12] = (-0.5*X[1] + X[5] - RM_RG3*X[12]) / RM_LS3;
	F[13] = (-X[0] + uin1 - (RM_RI + RM_RG1)*X[13]) / RM_LS1;
	F[14] = (-X[1] - (RM_RC + RM_RG1)*X[14]) / RM_LS1;
}

// Initial state shared by every trajectory.
template <class Precision>
__forceinline__ __host__ void ProblemInitialState(Precision* X)
{
	for (int i = 0; i < PROBLEM_SD; i++)
		X[i] = 0.0;
}

#endif
