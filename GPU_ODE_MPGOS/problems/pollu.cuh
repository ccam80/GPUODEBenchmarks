#ifndef PROBLEM_POLLU_CUH
#define PROBLEM_POLLU_CUH

// Pollution problem (Test Set for IVP Solvers): 25 reactions over 20 species,
// the photolysis rate k1 swept.
#define PROBLEM_NAME "pollu"
#define PROBLEM_SD 20
#define PROBLEM_NCP 1
#define PROBLEM_DURATION 60.0
#define PROBLEM_SWEEP_MIN 3.5e-2
#define PROBLEM_SWEEP_MAX 3.5
#define PROBLEM_SWEEP_LOG 1

#define POLLU_K2 26.6
#define POLLU_K3 1.23e4
#define POLLU_K4 8.6e-4
#define POLLU_K5 8.2e-4
#define POLLU_K6 1.5e4
#define POLLU_K7 1.3e-4
#define POLLU_K8 2.4e4
#define POLLU_K9 1.65e4
#define POLLU_K10 9.0e3
#define POLLU_K11 2.2e-2
#define POLLU_K12 1.2e4
#define POLLU_K13 1.88
#define POLLU_K14 1.63e4
#define POLLU_K15 4.8e6
#define POLLU_K16 3.5e-4
#define POLLU_K17 1.75e-2
#define POLLU_K18 1.0e8
#define POLLU_K19 4.44e11
#define POLLU_K20 1.24e3
#define POLLU_K21 2.1
#define POLLU_K22 5.78
#define POLLU_K23 4.74e-2
#define POLLU_K24 1.78e3
#define POLLU_K25 3.12

template <class Precision>
__forceinline__ __device__ void PerThread_OdeFunction(\
			int tid, int NT, \
			Precision*    F, Precision*    X, Precision     T, \
			Precision* cPAR, Precision* sPAR, int*      sPARi, Precision* ACC, int* ACCi)
{
	Precision r1  = cPAR[0]  * X[0];
	Precision r2  = POLLU_K2  * X[1] * X[3];
	Precision r3  = POLLU_K3  * X[4] * X[1];
	Precision r4  = POLLU_K4  * X[6];
	Precision r5  = POLLU_K5  * X[6];
	Precision r6  = POLLU_K6  * X[6] * X[5];
	Precision r7  = POLLU_K7  * X[8];
	Precision r8  = POLLU_K8  * X[8] * X[5];
	Precision r9  = POLLU_K9  * X[10] * X[1];
	Precision r10 = POLLU_K10 * X[10] * X[0];
	Precision r11 = POLLU_K11 * X[12];
	Precision r12 = POLLU_K12 * X[9] * X[1];
	Precision r13 = POLLU_K13 * X[13];
	Precision r14 = POLLU_K14 * X[0] * X[5];
	Precision r15 = POLLU_K15 * X[2];
	Precision r16 = POLLU_K16 * X[3];
	Precision r17 = POLLU_K17 * X[3];
	Precision r18 = POLLU_K18 * X[15];
	Precision r19 = POLLU_K19 * X[15];
	Precision r20 = POLLU_K20 * X[16] * X[5];
	Precision r21 = POLLU_K21 * X[18];
	Precision r22 = POLLU_K22 * X[18];
	Precision r23 = POLLU_K23 * X[0] * X[3];
	Precision r24 = POLLU_K24 * X[18] * X[0];
	Precision r25 = POLLU_K25 * X[19];

	F[0]  = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25;
	F[1]  = -r2 - r3 - r9 - r12 + r1 + r21;
	F[2]  = -r15 + r1 + r17 + r19 + r22;
	F[3]  = -r2 - r16 - r17 - r23 + r15;
	F[4]  = -r3 + 2.0*r4 + r6 + r7 + r13 + r20;
	F[5]  = -r6 - r8 - r14 - r20 + r3 + 2.0*r18;
	F[6]  = -r4 - r5 - r6 + r13;
	F[7]  = r4 + r5 + r6 + r7;
	F[8]  = -r7 - r8;
	F[9]  = -r12 + r7 + r9;
	F[10] = -r9 - r10 + r8 + r11;
	F[11] = r9;
	F[12] = -r11 + r10;
	F[13] = -r13 + r12;
	F[14] = r14;
	F[15] = -r18 - r19 + r16;
	F[16] = -r20;
	F[17] = r20;
	F[18] = -r21 - r22 - r24 + r23 + r25;
	F[19] = -r25 + r24;
}

// Initial state shared by every trajectory.
template <class Precision>
__forceinline__ __host__ void ProblemInitialState(Precision* X)
{
	for (int i = 0; i < PROBLEM_SD; i++)
		X[i] = 0.0;
	X[1]  = 0.2;
	X[3]  = 0.04;
	X[6]  = 0.1;
	X[7]  = 0.3;
	X[8]  = 0.01;
	X[16] = 0.007;
}

#endif
