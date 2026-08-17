#ifndef PROBLEM_STUBS_CUH
#define PROBLEM_STUBS_CUH

// The event and accessory hooks MPGOS requires. The initialization and
// after-step hooks carry the per-thread run budget: sPARi[0] holds the cycle
// budget in 2^21-cycle units and ACCi[0] the thread's start clock, so a
// thread whose budget is spent poisons its state and jumps to the end time.
#define WATCHDOG_CLOCK_SHIFT 21

template <class Precision>
__forceinline__ __device__ void PerThread_EventFunction(\
			int tid, int NT, Precision* EF, \
			Precision     T, Precision    dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{

}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterEventDetection(\
			int tid, int NT, int IDX, int& UDT, \
			Precision    &T, Precision   &dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{

}

template <class Precision>
__forceinline__ __device__ void PerThread_ActionAfterSuccessfulTimeStep(\
			int tid, int NT, int& UDT, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR, int*       sPARi, Precision* ACC, int* ACCi)
{
	if ((int)(clock64() >> WATCHDOG_CLOCK_SHIFT) - ACCi[0] > sPARi[0])
	{
		for (int c = 0; c < PROBLEM_SD; c++)
			X[c] = (Precision)nan("");
		UDT = 1;
	}
}

template <class Precision>
__forceinline__ __device__ void PerThread_Initialization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{
	ACCi[0] = (int)(clock64() >> WATCHDOG_CLOCK_SHIFT);
}

template <class Precision>
__forceinline__ __device__ void PerThread_Finalization(\
			int tid, int NT, int& DOIDX, \
			Precision&    T, Precision&   dT, Precision*    TD, Precision*   X, \
			Precision* cPAR, Precision* sPAR,       int* sPARi, Precision* ACC, int* ACCi)
{

}

#endif
