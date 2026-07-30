#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "Lorenz_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Solver Configuration
#define SOLVER RKCK45
#define PRECISION float  // float, double
const int NT = 8388608;
const int SD   = 3;     // SystemDimension
const int NCP  = 1;     // NumberOfControlParameters
const int NSP  = 0;     // NumberOfSharedParameters
const int NISP = 0;     // NumberOfIntegerSharedParameters
const int NE   = 0;     // NumberOfEvents
const int NA   = 0;     // NumberOfAccessories
const int NIA  = 0;     // NumberOfIntegerAccessories
const int NDO  = 10;     // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);
void SaveNumericalData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);

// NOTE: the run_ode_cpp runners rewrite lines 15 and 17 of this file by absolute
// line number, so nothing may be inserted above the config block. Keep the
// dataset-key helper below the forward declarations.
#include <cstdio>
#include <cctype>
#include <chrono>
#include <cmath>

// Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
// additively populated across machines without clobbering each other. The GPU
// name comes from nvidia-smi (the single source of truth shared by every
// framework) and is sanitised identically to runner_scripts/bench_key.*:
// tokenise on non-alphanumeric characters, drop the NVIDIA/GeForce vendor words,
// join the rest with '-'. e.g. "NVIDIA GeForce RTX 2060 SUPER" -> "RTX-2060-SUPER".
static std::string DatasetKey()
{
	static std::string cached;
	static bool done = false;
	if (done) return cached;

#ifdef _WIN32
	std::string os = "windows";
#elif defined(__APPLE__)
	std::string os = "macos";
#else
	std::string os = "linux";
#endif

	std::string raw;
#ifdef _WIN32
	FILE* pipe = _popen("nvidia-smi --query-gpu=name --format=csv,noheader", "r");
#else
	FILE* pipe = popen("nvidia-smi --query-gpu=name --format=csv,noheader", "r");
#endif
	if (pipe)
	{
		char buf[256];
		std::string captured;
		if (fgets(buf, sizeof(buf), pipe)) captured = buf;
#ifdef _WIN32
		int rc = _pclose(pipe);
#else
		int rc = pclose(pipe);
#endif
		// Trust the output only when nvidia-smi actually succeeded. With a
		// broken driver it prints its diagnostic ("Failed to initialize NVML:
		// ...") on stdout, which would otherwise be sanitised into a bogus GPU
		// name and silently key this framework's files differently from every
		// other framework's. On failure fall through to "unknown-gpu",
		// matching runner_scripts/bench_key.*.
		if (rc == 0) raw = captured;
	}

	std::string gpu, tok;
	for (size_t i = 0; i <= raw.size(); ++i)
	{
		char c = (i < raw.size()) ? raw[i] : '\0';
		if (std::isalnum((unsigned char)c))
		{
			tok += c;
		}
		else
		{
			if (!tok.empty() && tok != "NVIDIA" && tok != "GeForce")
			{
				if (!gpu.empty()) gpu += "-";
				gpu += tok;
			}
			tok.clear();
		}
	}
	if (gpu.empty()) gpu = "unknown-gpu";

	cached = os + "_" + gpu;
	done = true;
	return cached;
}

int main(int argc, char *argv[])
{
	int NumberOfProblems = NT;
	int BlockSize        = 32;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	int NumberOfParameters_R = NumberOfProblems;
	PRECISION R_RangeLower = 0.0;
    PRECISION R_RangeUpper = 21.0;
		vector<PRECISION> Parameters_R_Values(NumberOfParameters_R,0);
		Linspace(Parameters_R_Values, R_RangeLower, R_RangeUpper, NumberOfParameters_R);
	
	
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanLorenz(SelectedDevice);
	
	ScanLorenz.SolverOption(ThreadsPerBlock, BlockSize);
	ScanLorenz.SolverOption(InitialTimeStep, 1.0e-3);

	// ========================================
	// WORK-PRECISION (wp) MODE
	// ========================================
	// `Lorenz.exe 32768 wp` sweeps the fixed step size (RK4 build) or the
	// adaptive tolerance (RKCK45 build) at NT=32768 and records
	// "<setting> <time_ms> <error-vs-golden>" per point. dt and tolerances are
	// runtime SolverOptions, so only the solver family needs a rebuild. Grids
	// and protocol mirror runner_scripts/wp_common.py — keep in sync. wp takes
	// the minimum of 10 repeats after one warm-up; the N sweep below uses 20 per
	// transfer variant.
	if (argc > 2 && string(argv[2]) == string("wp"))
	{
		if (NT != 32768)
		{
			cerr << "wp mode must be built with NT = 32768" << endl;
			return 1;
		}
		vector<double> gx(NT), gy(NT), gz(NT);
		{
			ifstream gf("./data/numerical/golden_lorenz_32768.csv");
			if (!gf)
			{
				cerr << "golden reference missing - run "
				        "runner_scripts/golden/generate_golden.jl first" << endl;
				return 1;
			}
			char comma;
			for (int i = 0; i < NT; i++)
				gf >> gx[i] >> comma >> gy[i] >> comma >> gz[i];
		}

		const bool FixedMode = (SOLVER == RK4);
		vector<double> Settings;
		if (FixedMode)
			for (int k = 4; k <= 13; k++) Settings.push_back(pow(2.0, -k));
		else
			for (int k = 2; k <= 8; k++) Settings.push_back(pow(10.0, -k));

		string Mode = FixedMode ? "fixed" : "adaptive";
		ofstream wpfile(("./data/CPP/MPGOS_wp_" + Mode + "_" + DatasetKey() + ".txt").c_str());
		wpfile.precision(12);

		const int Repeats = 10;
		for (size_t si = 0; si < Settings.size(); si++)
		{
			double Setting = Settings[si];
			ScanLorenz.SolverOption(InitialTimeStep, FixedMode ? Setting : 1.0e-3);
			if (!FixedMode)
			{
				for (int c = 0; c < SD; c++)
				{
					ScanLorenz.SolverOption(RelativeTolerance, c, Setting);
					ScanLorenz.SolverOption(AbsoluteTolerance, c, Setting);
				}
			}

			double BestMs = 1.0e300;
			for (int r = 0; r <= Repeats; r++)
			{
				// Reset states/time domain: Solve() advances in place.
				FillSolverObject(ScanLorenz, Parameters_R_Values, NT);
				ScanLorenz.SynchroniseFromHostToDevice(All);

				auto T0 = std::chrono::steady_clock::now();
				ScanLorenz.Solve();
				ScanLorenz.InsertSynchronisationPoint();
				ScanLorenz.SynchroniseSolver();
				// ActualState only: All would also copy the NDO dense-output
				// registers, which no other package stores or transfers.
				ScanLorenz.SynchroniseFromDeviceToHost(ActualState);
				ScanLorenz.SynchroniseDevice();
				auto T1 = std::chrono::steady_clock::now();

				cudaError_t WpErr = cudaGetLastError();
				if (WpErr != cudaSuccess)
					cerr << "CUDA launch error: " << cudaGetErrorString(WpErr) << endl;

				double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
				if (r > 0 && Ms < BestMs) BestMs = Ms;   // r == 0 is warm-up
			}

			double Sum2 = 0.0;
			for (int i = 0; i < NT; i++)
			{
				double DX = (double)ScanLorenz.GetHost<PRECISION>(i, ActualState, 0) - gx[i];
				double DY = (double)ScanLorenz.GetHost<PRECISION>(i, ActualState, 1) - gy[i];
				double DZ = (double)ScanLorenz.GetHost<PRECISION>(i, ActualState, 2) - gz[i];
				Sum2 += DX*DX + DY*DY + DZ*DZ;
			}
			double Err = sqrt(Sum2 / (NT * 3.0));

			wpfile << Setting << " " << BestMs << " " << scientific << Err << fixed << "\n";
			cout << "wp " << Mode << " setting=" << Setting << ": " << BestMs
			     << " ms, err=" << scientific << Err << fixed << endl;
		}
		wpfile.close();

		cout << "wp sweep finished!" << endl;
		return 0;
	}

	// Minimum of repeated solves, discarding r == 0. The runners export
	// CUDA_MODULE_LOADING=EAGER, which removes the lazy-load cost that otherwise
	// lands on the first launches (1.14 ms decaying to 0.158 ms over three solves
	// at NT=8). 20 matches every other framework in this suite.
	const int TimingRepeats = 20;

	// Device-only: the kernel with neither transfer. The h2d ahead of each run
	// is untimed and also resets ActualTime, which Solve() advances in place.
	double ElapsedDeviceMs = 1.0e300;
	for (int r = 0; r <= TimingRepeats; r++)
	{
		FillSolverObject(ScanLorenz, Parameters_R_Values, NT);
		ScanLorenz.SynchroniseFromHostToDevice(All);

		auto T0 = std::chrono::steady_clock::now();
		ScanLorenz.Solve();
		ScanLorenz.InsertSynchronisationPoint();
		ScanLorenz.SynchroniseSolver();
		ScanLorenz.SynchroniseDevice();
		auto T1 = std::chrono::steady_clock::now();

		double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
		if (r > 0 && Ms < ElapsedDeviceMs) ElapsedDeviceMs = Ms;
	}

	// End-to-end: h2d, kernel, d2h. Only ActualState returns, matching the
	// final-state transfer every other package times.
	double ElapsedMs = 1.0e300;
	for (int r = 0; r <= TimingRepeats; r++)
	{
		FillSolverObject(ScanLorenz, Parameters_R_Values, NT);

		auto T0 = std::chrono::steady_clock::now();
		ScanLorenz.SynchroniseFromHostToDevice(All);
		ScanLorenz.Solve();
		ScanLorenz.InsertSynchronisationPoint();
		ScanLorenz.SynchroniseSolver();
		ScanLorenz.SynchroniseFromDeviceToHost(ActualState);
		ScanLorenz.SynchroniseDevice();
		auto T1 = std::chrono::steady_clock::now();

		double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
		if (r > 0 && Ms < ElapsedMs) ElapsedMs = Ms;
	}

	// Untimed: the ActualTime print and SaveData need the registers the timed
	// d2h deliberately left on the device.
	ScanLorenz.SynchroniseFromDeviceToHost(All);
	ScanLorenz.SynchroniseDevice();
		// Check for kernel launch errors
	cudaError_t _lastErr = cudaGetLastError();
	if (_lastErr != cudaSuccess) {
		std::cerr << "CUDA launch error: " << cudaGetErrorString(_lastErr) << std::endl;
		// Exit non-zero without recording. The timings above are meaningless
		// after a failed launch, and a zero exit would let the runner continue
		// the N sweep writing bogus rows instead of stopping at the ceiling.
		std::cerr << "No timing recorded for NT = " << NT << "." << std::endl;
		return 1;
	}
	std::cout << ScanLorenz.GetHost<PRECISION>(0, ActualTime) << std::endl;
	cout << "Total simulation time:           " << ElapsedMs << "ms" << endl;
	cout << "Device-only time (no h2d/d2h):   " << ElapsedDeviceMs << "ms" << endl;
	cout << "Ensemble size:                   " << NT << endl << endl;
		
	
	ofstream datafile;
	if (SOLVER == RK4){
		datafile.open (("./data/CPP/MPGOS_times_unadaptive_" + DatasetKey() + ".txt").c_str(),ios::app);
		datafile << NT << "\t" << ElapsedMs << "\t" << ElapsedDeviceMs << "\n";
		datafile.close();
	}else{
		
		datafile.open (("./data/CPP/MPGOS_times_adaptive_" + DatasetKey() + ".txt").c_str(),ios::app);
		datafile << NT << "\t" << ElapsedMs << "\t" << ElapsedDeviceMs << "\n";
		datafile.close();
	}
	
	//SaveData(ScanLorenz, NT);
	
	// Save numerical data for 32768-trajectory run
	if (NT == 32768) {
		SaveNumericalData(ScanLorenz, NT);
		SaveData(ScanLorenz, NT);
		// save per-trajectory step counts (total steps, rejected steps)
	}
	
	cout << "Test finished!" << endl;
}

// AUXILIARY FUNCTION -----------------------------------------------------------------------------

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
	x[0]   = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& R_Values, int NumberOfThreads)
{
	int ProblemNumber = 0;
	for (int k=0; k<NumberOfThreads; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, 0.001*1000.0 );
		// MPGOS Solve() continues from ActualTime (by design, for
		// continuation runs); reset it so repeated solves in the wp sweep
		// re-integrate from t=0 instead of no-opping at t=t_end.
		Solver.SetHost(ProblemNumber, ActualTime, 0 );
		
		Solver.SetHost(ProblemNumber, ActualState, 0, 1.0 );
		Solver.SetHost(ProblemNumber, ActualState, 1, 0.0 );
		Solver.SetHost(ProblemNumber, ActualState, 2, 0.0 );
		
		Solver.SetHost(ProblemNumber, ControlParameters, 0, R_Values[k] );
		
		ProblemNumber++;
	}
}

void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, int NumberOfThreads)
{
	ofstream DataFile;
	// Create directory if it doesn't exist (assumes unix-like system)
	system("mkdir -p ./data/numerical");
	DataFile.open ( ("./data/numerical/mpgos_internalsave_" + DatasetKey() + ".csv").c_str() );
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 2);
		DataFile << '\n';
	}
	
	DataFile.close();
}

void SaveNumericalData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, int NumberOfThreads)
{
	ofstream DataFile;
	// Create directory if it doesn't exist (assumes unix-like system)
	system("mkdir -p ./data/numerical");
	DataFile.open ( ("./data/numerical/mpgos_" + DatasetKey() + ".csv").c_str() );
	
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 0) << ',';
		DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 1) << ',';
		DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 2);
		DataFile << '\n';
	}
	
	DataFile.close();
}
