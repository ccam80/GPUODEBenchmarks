#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

// Build with -DPROBLEM_HEADER=\"problems/lorenz.cuh\" -DSOLVER_CHOICE=RK4 -DNT_VALUE=32768.
#ifndef PROBLEM_HEADER
	#error "define PROBLEM_HEADER, e.g. -DPROBLEM_HEADER=\"problems/lorenz.cuh\""
#endif
#ifndef SOLVER_CHOICE
	#define SOLVER_CHOICE RKCK45
#endif
#ifndef NT_VALUE
	#define NT_VALUE 8388608
#endif

#include PROBLEM_HEADER
#include "problems/stubs.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Solver Configuration
#define SOLVER SOLVER_CHOICE
#define PRECISION float  // float, double
const int NT = NT_VALUE;
const int SD   = PROBLEM_SD;   // SystemDimension
const int NCP  = PROBLEM_NCP;  // NumberOfControlParameters
const int NSP  = 0;     // NumberOfSharedParameters
const int NISP = 1;     // NumberOfIntegerSharedParameters (run budget)
const int NE   = 0;     // NumberOfEvents
const int NA   = 0;     // NumberOfAccessories
const int NIA  = 1;     // NumberOfIntegerAccessories (start clock)
const int NDO  = 10;     // NumberOfPointsOfDenseOutput

const PRECISION DURATION = (PRECISION)PROBLEM_DURATION;
// The N sweep steps duration * 2^-10, matching the other frameworks.
const PRECISION TIMING_DT = (PRECISION)(PROBLEM_DURATION / 1024.0);

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);
void SaveNumericalData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>
#include <thread>

// Dataset key "<os>_<gpu>" from nvidia-smi, sanitised as in runner_scripts/bench_key.*.
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
		// Only a successful nvidia-smi names the GPU; anything else is "unknown-gpu".
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

// Directory holding this machine's files for a package and problem; creates it.
static std::string DataDir(const std::string& package)
{
	std::string dir = "./data/" + package + "/" + DatasetKey() + "/" + PROBLEM_NAME;
#ifdef _WIN32
	// cmd needs backslashes and creates intermediate directories itself.
	std::string win = dir;
	std::replace(win.begin(), win.end(), '/', '\\');
	system(("if not exist \"" + win + "\" mkdir \"" + win + "\"").c_str());
#else
	system(("mkdir -p \"" + dir + "\"").c_str());
#endif
	return dir + "/";
}

// Per-run wall-clock watchdog; a hung kernel can only be stopped by process exit.
static double WatchdogSeconds()
{
	const char* env = std::getenv("BENCH_WATCHDOG_SECONDS");
	return env ? atof(env) : 120.0;
}

static std::mutex WatchdogLock;
static std::string WatchdogFile;
static std::vector<std::string> WatchdogRows;
static std::atomic<long long> WatchdogDeadlineMs(0);   // 0 = disarmed

static long long NowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Arm with the rows to append if the run never returns; margin over the soft cap.
static void ArmWatchdog(const std::string& file, const std::vector<std::string>& rows)
{
	std::lock_guard<std::mutex> hold(WatchdogLock);
	WatchdogFile = file;
	WatchdogRows = rows;
	WatchdogDeadlineMs = NowMs() + (long long)((WatchdogSeconds() * 2.0 + 30.0) * 1000.0);
}

static void DisarmWatchdog()
{
	WatchdogDeadlineMs = 0;
}

static void WatchdogMain()
{
	for (;;)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
		long long deadline = WatchdogDeadlineMs;
		if (deadline == 0 || NowMs() < deadline) continue;
		std::lock_guard<std::mutex> hold(WatchdogLock);
		std::ofstream out(WatchdogFile.c_str(), std::ios::app);
		for (size_t i = 0; i < WatchdogRows.size(); ++i)
			out << WatchdogRows[i] << "\n";
		out.close();
		std::cout << "WATCHDOG " << PROBLEM_NAME
		          << ": run never returned" << std::endl;
		std::_Exit(0);
	}
}

int main(int argc, char *argv[])
{
	int NumberOfProblems = NT;
	int BlockSize        = 32;

	// `<exe> states <build_s>` writes an SD-keyed row with the build time.
	bool StatesMode = (argc > 1 && string(argv[1]) == string("states"));
	string StatesBuild = (StatesMode && argc > 2) ? string(argv[2])
	                                              : string("nan");

	std::thread(WatchdogMain).detach();

	ListCUDADevices();

	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);

	PrintPropertiesOfSpecificDevice(SelectedDevice);


	int NumberOfParameters_R = NumberOfProblems;
	PRECISION R_RangeLower = (PRECISION)PROBLEM_SWEEP_MIN;
    PRECISION R_RangeUpper = (PRECISION)PROBLEM_SWEEP_MAX;
		vector<PRECISION> Parameters_R_Values(NumberOfParameters_R,0);
#if PROBLEM_SWEEP_LOG
		Logspace(Parameters_R_Values, R_RangeLower, R_RangeUpper, NumberOfParameters_R);
#else
		Linspace(Parameters_R_Values, R_RangeLower, R_RangeUpper, NumberOfParameters_R);
#endif


	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> Scan(SelectedDevice);

	Scan.SolverOption(ThreadsPerBlock, BlockSize);
	Scan.SolverOption(InitialTimeStep, TIMING_DT);
	// Adaptive N-sweep tolerance; mirrors TIMING_TOL in runner_scripts/wp_common.py.
	if (SOLVER != RK4)
		for (int c = 0; c < SD; c++)
		{
			Scan.SolverOption(RelativeTolerance, c, 1.0e-5);
			Scan.SolverOption(AbsoluteTolerance, c, 1.0e-5);
		}

	// Device-side run budget, 1.25 over the host cap; see problems/stubs.cuh.
	int ClockKHz = 0;
	cudaDeviceGetAttribute(&ClockKHz, cudaDevAttrClockRate, SelectedDevice);
	if (ClockKHz <= 0) ClockKHz = 3000000;
	long long BudgetCycles = (long long)(WatchdogSeconds() * 1.25 * ClockKHz * 1000.0);
	Scan.SetHost(IntegerSharedParameters, 0, (int)(BudgetCycles >> WATCHDOG_CLOCK_SHIFT));

	// `<exe> wp` sweeps step size (RK4) or tolerance (RKCK45); grids mirror runner_scripts/wp_common.py.
	if (argc > 1 && string(argv[1]) == string("wp"))
	{
		vector< vector<double> > golden(NT, vector<double>(SD, 0.0));
		{
			string gpath = "./data/numerical/golden_" + string(PROBLEM_NAME) + "_131072.csv";
			ifstream gf(gpath.c_str());
			if (!gf)
			{
				cerr << gpath << " missing - run "
				        "runner_scripts/golden/generate_golden.jl first" << endl;
				return 1;
			}
			char comma;
			for (int i = 0; i < NT; i++)
				for (int c = 0; c < SD; c++)
				{
					gf >> golden[i][c];
					if (c + 1 < SD) gf >> comma;
				}
		}

		const bool FixedMode = (SOLVER == RK4);
		vector<double> Settings;
		if (FixedMode)
			for (int k = 4; k <= 13; k++) Settings.push_back(PROBLEM_DURATION * pow(2.0, -k));
		else
			for (int k = 2; k <= 8; k++) Settings.push_back(pow(10.0, -k));

		// Filenames carry the cubie-vocabulary algorithm name.
		string Mode = FixedMode ? "fixed" : "adaptive";
		string Algorithm = FixedMode ? "classical-rk4" : "cash-karp-54";
		const std::string WpPath = DataDir("CPP") + "MPGOS_wp_" + Mode + "_" + Algorithm + ".txt";
		ofstream wpfile(WpPath.c_str());
		wpfile.precision(12);

		const int Repeats = 10;
		for (size_t si = 0; si < Settings.size(); si++)
		{
			double Setting = Settings[si];
			Scan.SolverOption(InitialTimeStep, FixedMode ? Setting : (double)TIMING_DT);
			if (!FixedMode)
			{
				for (int c = 0; c < SD; c++)
				{
					Scan.SolverOption(RelativeTolerance, c, Setting);
					Scan.SolverOption(AbsoluteTolerance, c, Setting);
				}
			}

			// Later settings are slower, so a breach abandons the sweep as NaN rows.
			std::vector<std::string> NanRows;
			for (size_t sj = si; sj < Settings.size(); sj++)
			{
				std::ostringstream row;
				row.precision(12);
				row << Settings[sj] << " nan nan";
				NanRows.push_back(row.str());
			}

			bool Breached = false;
			double BestMs = 1.0e300;
			for (int r = 0; r <= Repeats; r++)
			{
				// Reset states/time domain: Solve() advances in place.
				FillSolverObject(Scan, Parameters_R_Values, NT);
				Scan.SynchroniseFromHostToDevice(All);

				ArmWatchdog(WpPath, NanRows);
				auto T0 = std::chrono::steady_clock::now();
				Scan.Solve();
				Scan.InsertSynchronisationPoint();
				Scan.SynchroniseSolver();
				// ActualState only: All would also copy the NDO dense-output registers.
				Scan.SynchroniseFromDeviceToHost(ActualState);
				Scan.SynchroniseDevice();
				auto T1 = std::chrono::steady_clock::now();
				DisarmWatchdog();

				cudaError_t WpErr = cudaGetLastError();
				if (WpErr != cudaSuccess)
				{
					cerr << "CUDA launch error: " << cudaGetErrorString(WpErr) << endl;
					cerr << "No wp row recorded for setting = " << Setting << "." << endl;
					return 1;
				}

				double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
				if (Ms > WatchdogSeconds() * 1000.0) { Breached = true; break; }
				if (r > 0 && Ms < BestMs) BestMs = Ms;   // r == 0 is warm-up
			}

			if (Breached)
			{
				for (size_t i = 0; i < NanRows.size(); ++i)
					wpfile << NanRows[i] << "\n";
				wpfile.flush();
				cout << "WATCHDOG wp setting=" << Setting
				     << ": run exceeded the cap" << endl;
				break;
			}

			double Sum2 = 0.0;
			for (int i = 0; i < NT; i++)
				for (int c = 0; c < SD; c++)
				{
					double D = (double)Scan.GetHost<PRECISION>(i, ActualState, c) - golden[i][c];
					Sum2 += D*D;
				}
			double Err = sqrt(Sum2 / (NT * (double)SD));

			wpfile << Setting << " " << BestMs << " " << scientific << Err << fixed << "\n";
			wpfile.flush();
			cout << "wp " << Mode << " setting=" << Setting << ": " << BestMs
			     << " ms, err=" << scientific << Err << fixed << endl;
		}
		wpfile.close();

		cout << "wp sweep finished!" << endl;
		return 0;
	}

	// Minimum of TimingRepeats solves; r == 0 is a discarded warm-up.
	const int TimingRepeats = 20;

	const std::string TimesPath = DataDir("CPP") + "MPGOS_" +
		(StatesMode ? string("states") : string("times")) +
		(SOLVER == RK4 ? "_fixed_classical-rk4.txt"
		               : "_adaptive_cash-karp-54.txt");
	std::vector<std::string> TimesNanRow;
	{
		std::ostringstream row;
		if (StatesMode)
			row << SD << "\tnan\tnan\t" << StatesBuild;
		else
			row << NT << "\tnan\tnan";
		TimesNanRow.push_back(row.str());
	}
	bool TimesBreached = false;

	// Device-only timing: the untimed h2d resets the in-place solver state.
	double ElapsedDeviceMs = 1.0e300;
	for (int r = 0; r <= TimingRepeats; r++)
	{
		FillSolverObject(Scan, Parameters_R_Values, NT);
		Scan.SynchroniseFromHostToDevice(All);

		ArmWatchdog(TimesPath, TimesNanRow);
		auto T0 = std::chrono::steady_clock::now();
		Scan.Solve();
		Scan.InsertSynchronisationPoint();
		Scan.SynchroniseSolver();
		Scan.SynchroniseDevice();
		auto T1 = std::chrono::steady_clock::now();
		DisarmWatchdog();

		double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
		if (Ms > WatchdogSeconds() * 1000.0) { TimesBreached = true; break; }
		if (r > 0 && Ms < ElapsedDeviceMs) ElapsedDeviceMs = Ms;
	}

	// End-to-end timing: h2d, kernel, ActualState d2h.
	double ElapsedMs = 1.0e300;
	for (int r = 0; !TimesBreached && r <= TimingRepeats; r++)
	{
		FillSolverObject(Scan, Parameters_R_Values, NT);

		ArmWatchdog(TimesPath, TimesNanRow);
		auto T0 = std::chrono::steady_clock::now();
		Scan.SynchroniseFromHostToDevice(All);
		Scan.Solve();
		Scan.InsertSynchronisationPoint();
		Scan.SynchroniseSolver();
		Scan.SynchroniseFromDeviceToHost(ActualState);
		Scan.SynchroniseDevice();
		auto T1 = std::chrono::steady_clock::now();
		DisarmWatchdog();

		double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
		if (Ms > WatchdogSeconds() * 1000.0) { TimesBreached = true; break; }
		if (r > 0 && Ms < ElapsedMs) ElapsedMs = Ms;
	}

	if (TimesBreached)
	{
		std::ofstream out(TimesPath.c_str(), std::ios::app);
		out << TimesNanRow[0] << "\n";
		out.close();
		cout << "WATCHDOG N=" << NT << ": run exceeded the cap" << endl;
		return 0;
	}

	// Untimed full d2h for the ActualTime print and SaveData.
	Scan.SynchroniseFromDeviceToHost(All);
	Scan.SynchroniseDevice();
		// Check for kernel launch errors
	cudaError_t _lastErr = cudaGetLastError();
	if (_lastErr != cudaSuccess) {
		std::cerr << "CUDA launch error: " << cudaGetErrorString(_lastErr) << std::endl;
		std::cerr << "No timing recorded for NT = " << NT << "." << std::endl;
		return 1;
	}
	std::cout << Scan.GetHost<PRECISION>(0, ActualTime) << std::endl;
	cout << "Total simulation time:           " << ElapsedMs << "ms" << endl;
	cout << "Device-only time (no h2d/d2h):   " << ElapsedDeviceMs << "ms" << endl;
	cout << "Ensemble size:                   " << NT << endl << endl;


	ofstream datafile(TimesPath.c_str(), ios::app);
	if (StatesMode)
		datafile << SD << "\t" << ElapsedMs << "\t" << ElapsedDeviceMs
		         << "\t" << StatesBuild << "\n";
	else
		datafile << NT << "\t" << ElapsedMs << "\t" << ElapsedDeviceMs << "\n";
	datafile.close();

	//SaveData(Scan, NT);

	// Save numerical data for 32768-trajectory run
	if (NT == 32768 && !StatesMode) {
		SaveNumericalData(Scan, NT);
		SaveData(Scan, NT);
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

// Geometric grid, matching problems.py for a log-scaled sweep.
void Logspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
	x[0] = B;

	if ( N>1 )
	{
		x[N-1] = E;
		double LogB = log10((double)B);
		double Increment = (log10((double)E) - LogB)/(N-1);

		for (int i=1; i<N-1; i++)
		{
			x[i] = (PRECISION)pow(10.0, LogB + i*Increment);
		}
	}
}

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& R_Values, int NumberOfThreads)
{
	PRECISION X0[SD];
	ProblemInitialState<PRECISION>(X0);

	int ProblemNumber = 0;
	for (int k=0; k<NumberOfThreads; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, DURATION );
		// Solve() continues from ActualTime, so reset it to re-integrate from t=0.
		Solver.SetHost(ProblemNumber, ActualTime, 0 );

		for (int c=0; c<SD; c++)
			Solver.SetHost(ProblemNumber, ActualState, c, X0[c] );

		Solver.SetHost(ProblemNumber, ControlParameters, 0, R_Values[k] );

		ProblemNumber++;
	}
}

void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, int NumberOfThreads)
{
	ofstream DataFile;
	// Create directory if it doesn't exist (assumes unix-like system)
	DataFile.open ( (DataDir("numerical") + "mpgos_internalsave.csv").c_str() );

	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);

	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ControlParameters, 0) << ',';
		for (int c=0; c<SD; c++)
		{
			DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, c);
			if (c + 1 < SD) DataFile << ',';
		}
		DataFile << '\n';
	}

	DataFile.close();
}

void SaveNumericalData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, int NumberOfThreads)
{
	ofstream DataFile;
	// Create directory if it doesn't exist (assumes unix-like system)
	DataFile.open ( (DataDir("numerical") + "mpgos.csv").c_str() );

	DataFile.precision(10);
	DataFile.flags(ios::scientific);

	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		for (int c=0; c<SD; c++)
		{
			DataFile << Solver.GetHost<PRECISION>(tid, ActualState, c);
			if (c + 1 < SD) DataFile << ',';
		}
		DataFile << '\n';
	}

	DataFile.close();
}
