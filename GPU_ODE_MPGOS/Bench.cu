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
// Generated from runner_scripts/protocol.toml by the launcher before every build.
#include "protocol.h"

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
const int NDO  = 0;      // NumberOfPointsOfDenseOutput: nothing reads it, and
                         // storing it is work the other suites do not do

const PRECISION DURATION = (PRECISION)PROBLEM_DURATION;
// The N sweep steps duration * 2^-timing_k.
const PRECISION TIMING_DT = (PRECISION)(PROBLEM_DURATION * pow(2.0, -PROTOCOL_TIMING_K));

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void Logspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);
void SaveNumericalData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
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
	return env ? atof(env) : PROTOCOL_WATCHDOG_SECONDS;
}

// Repeat floor and ceiling from the first timed run's seconds, per the protocol schedule.
static void RepeatBounds(double FirstMs, int Cap, int& Floor, int& Ceiling)
{
	double FirstS = FirstMs / 1000.0;
	for (int i = 0; i < PROTOCOL_REPEAT_SCHEDULE_ROWS; i++)
	{
		if (FirstS < PROTOCOL_REPEAT_SCHEDULE[i][0])
		{
			Floor = (int)PROTOCOL_REPEAT_SCHEDULE[i][1];
			Ceiling = (int)PROTOCOL_REPEAT_SCHEDULE[i][2];
			break;
		}
	}
	if (Floor > Cap) Floor = Cap;
	if (Ceiling > Cap) Ceiling = Cap;
}

static double MedianMs(std::vector<double> Timed)   // by value: nth_element permutes
{
	size_t Half = Timed.size() / 2;
	std::nth_element(Timed.begin(), Timed.begin() + Half, Timed.end());
	double Upper = Timed[Half];
	if (Timed.size() % 2) return Upper;
	std::nth_element(Timed.begin(), Timed.begin() + Half - 1, Timed.end());
	return 0.5 * (Timed[Half - 1] + Upper);
}

// True at the ceiling, or past the floor with median/min - 1 within the protocol spread.
static bool RepeatsDone(const std::vector<double>& Timed, int Floor, int Ceiling)
{
	if ((int)Timed.size() >= Ceiling) return true;
	if ((int)Timed.size() < Floor) return false;
	double Min = *std::min_element(Timed.begin(), Timed.end());
	return MedianMs(Timed) / Min - 1.0 <= PROTOCOL_REPEAT_SPREAD;
}

// The token as a double; nan when it does not parse.
static double ParseTime(const std::string& Token)
{
	const char* Start = Token.c_str();
	char* End = NULL;
	double Value = strtod(Start, &End);
	return (End == Start) ? std::nan("") : Value;
}

static std::string Fmt(double Value)
{
	if (std::isnan(Value)) return "nan";
	char Text[32];
	snprintf(Text, sizeof(Text), "%.10g", Value);
	return Text;
}

// One store row through runner_scripts/results.py, Samples as every attempt in ms (warm-up first); BENCH_FLOOR is honoured there.
static void RecordResult(const std::string& Analysis, const std::string& Mode,
                         const std::string& Algorithm, const std::string& SettingKind,
                         double Setting, int N, int States, const std::string& Transfers,
                         double MinMs, const std::vector<double>* Samples,
                         double ErroredPercent, double Err, double BuildS)
{
#ifdef _WIN32
	const char* Python = "python";
#else
	const char* Python = "python3";
#endif
	std::ostringstream Cmd;
	Cmd << Python << " runner_scripts/results.py record cpp " << DatasetKey()
	    << " " << Analysis << " " << PROBLEM_NAME << " " << Algorithm << " " << Mode
	    << " " << SettingKind << " " << Fmt(Setting) << " " << N << " " << States
	    << " default " << Transfers << " min_ms=" << Fmt(MinMs)
	    << " errored_pct=" << Fmt(ErroredPercent) << " error=" << Fmt(Err)
	    << " build_s=" << Fmt(BuildS);
	if (Samples && !Samples->empty())
	{
		Cmd << " samples=";
		for (size_t i = 0; i < Samples->size(); ++i)
			Cmd << (i ? ";" : "") << Fmt((*Samples)[i]);
	}
	int Status = system(Cmd.str().c_str());
	if (Status != 0)
		std::cerr << "results.py record failed (" << Status << "): " << Cmd.str() << std::endl;
}

// Percent of trajectories whose final state is not finite; mirrors errored_pct in runner_scripts/wp_common.py.
template<class SolverT>
static double ErroredPct(SolverT& Solver, int Threads, int States)
{
	int Bad = 0;
	for (int tid = 0; tid < Threads; ++tid)
	{
		for (int c = 0; c < States; ++c)
		{
			if (!std::isfinite((double) Solver.template GetHost<PRECISION>(
					tid, ActualState, c)))
			{
				++Bad;
				break;
			}
		}
	}
	return 100.0 * (double) Bad / (double) Threads;
}

// Breach exit code from the protocol; the runner NaN-fills the leg's remaining sizes.
static const int WatchdogExitCode = PROTOCOL_WATCHDOG_EXIT_CODE;

// Mode and algorithm names for the store and watchdog messages.
static const char* ModeName      = (SOLVER == RK4) ? "fixed" : "adaptive";
static const char* AlgorithmName = (SOLVER == RK4) ? "classical-rk4"
                                                   : "cash-karp-54";
static bool StatesRun = false;   // set in main; states rows are SD-keyed

static std::mutex WatchdogLock;
static std::function<void()> WatchdogRecord;
static std::atomic<long long> WatchdogDeadlineMs(0);   // 0 = disarmed

static long long NowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Arm with the NaN rows to record if the run never returns; margin over the soft cap.
static void ArmWatchdog(std::function<void()> record)
{
	std::lock_guard<std::mutex> hold(WatchdogLock);
	WatchdogRecord = record;
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
		if (WatchdogRecord) WatchdogRecord();
		std::cout << "WATCHDOG " << PROBLEM_NAME;
		if (StatesRun) std::cout << " states=" << SD;
		std::cout << " " << ModeName << " " << AlgorithmName << " N=" << NT
		          << ": run never returned" << std::endl;
		std::_Exit(WatchdogExitCode);
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
	StatesRun = StatesMode;

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
	// Adaptive N-sweep tolerance.
	if (SOLVER != RK4)
		for (int c = 0; c < SD; c++)
		{
			Scan.SolverOption(RelativeTolerance, c, PROTOCOL_TIMING_TOL);
			Scan.SolverOption(AbsoluteTolerance, c, PROTOCOL_TIMING_TOL);
		}

	// Device-side run budget, 1.25 over the host cap; see problems/stubs.cuh.
	int ClockKHz = 0;
	cudaDeviceGetAttribute(&ClockKHz, cudaDevAttrClockRate, SelectedDevice);
	if (ClockKHz <= 0) ClockKHz = 3000000;
	long long BudgetCycles = (long long)(WatchdogSeconds() * 1.25 * ClockKHz * 1000.0);
	Scan.SetHost(IntegerSharedParameters, 0, (int)(BudgetCycles >> WATCHDOG_CLOCK_SHIFT));

	// `<exe> wp` sweeps step size (RK4) or tolerance (RKCK45) over the protocol grids.
	if (argc > 1 && string(argv[1]) == string("wp"))
	{
		vector< vector<double> > golden(NT, vector<double>(SD, 0.0));
		{
			string gpath = "./data/numerical/golden_" + string(PROBLEM_NAME) + "_"
				+ std::to_string(PROTOCOL_N_WP) + ".csv";
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
			for (int k = PROTOCOL_WP_K_LO; k <= PROTOCOL_WP_K_HI; k++)
				Settings.push_back(PROBLEM_DURATION * pow(2.0, -k));
		else
			for (int k = PROTOCOL_TOL_K_LO; k <= PROTOCOL_TOL_K_HI; k++)
				Settings.push_back(pow(10.0, -k));

		// The store carries the cubie-vocabulary algorithm name.
		string Mode = ModeName;
		string Algorithm = AlgorithmName;
		const std::string SettingKind = FixedMode ? "dt" : "tol";
		// NaN rows for the settings from si on, for a breach or a hard exit.
		auto NanFrom = [&](size_t si) {
			for (size_t sj = si; sj < Settings.size(); sj++)
				RecordResult("wp", Mode, Algorithm, SettingKind, Settings[sj], NT, SD,
					"d2h", std::nan(""), NULL, 100.0, std::nan(""), std::nan(""));
		};

		// Repeat ceiling; the count follows the first timed run's duration.
		const int Repeats = PROTOCOL_REPEAT_CAP;
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

			bool Breached = false;
			double BestMs = 1.0e300;
			std::vector<double> WpSamples;
			std::vector<double> WpTimed;
			int WpFloor = 0, WpCeiling = 0;
			for (int r = 0; ; r++)
			{
				// Reset states/time domain: Solve() advances in place.
				FillSolverObject(Scan, Parameters_R_Values, NT);
				Scan.SynchroniseFromHostToDevice(All);

				// Later settings are slower, so a hard exit abandons the sweep as NaN rows.
				ArmWatchdog([&, si]() { NanFrom(si); });
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
				WpSamples.push_back(Ms);
				if (Ms > WatchdogSeconds() * 1000.0) { Breached = true; break; }
				if (r == 0) continue;   // r == 0 is warm-up
				WpTimed.push_back(Ms);
				if (Ms < BestMs) BestMs = Ms;
				if (r == 1) RepeatBounds(WpTimed[0], Repeats, WpFloor, WpCeiling);
				if (RepeatsDone(WpTimed, WpFloor, WpCeiling)) break;
			}
			if (Breached)
			{
				NanFrom(si);
				cout << "WATCHDOG " << PROBLEM_NAME << " " << Mode << " "
				     << Algorithm << " wp setting=" << Setting
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
			const double WpErroredPct = ErroredPct(Scan, NT, SD);

			RecordResult("wp", Mode, Algorithm, SettingKind, Setting, NT, SD, "d2h",
				BestMs, &WpSamples, WpErroredPct, Err, std::nan(""));
			cout << "wp " << Mode << " setting=" << Setting << ": " << BestMs
			     << " ms, err=" << scientific << Err << fixed
			     << ", errored=" << WpErroredPct << "%" << endl;
		}

		cout << "wp sweep finished!" << endl;
		return 0;
	}

	// Repeat ceiling; the count per leg follows its first timed run.
	const int TimingRepeats = PROTOCOL_REPEAT_CAP;

	const std::string TimesAnalysis = StatesMode ? "states" : "times";
	const std::string TimesMode = ModeName;
	const std::string TimesAlgorithm = AlgorithmName;
	const std::string TimesSettingKind = (SOLVER == RK4) ? "dt" : "tol";
	const double TimesSetting = (SOLVER == RK4) ? (double)TIMING_DT
	                                            : (double)PROTOCOL_TIMING_TOL;
	const double BuildS = ParseTime(StatesBuild);
	// Both transfer legs of this point, NaN-timed, for a breach or a hard exit.
	auto RecordNan = [&]() {
		RecordResult(TimesAnalysis, TimesMode, TimesAlgorithm, TimesSettingKind,
			TimesSetting, NT, SD, "both", std::nan(""), NULL, 100.0, std::nan(""), BuildS);
		RecordResult(TimesAnalysis, TimesMode, TimesAlgorithm, TimesSettingKind,
			TimesSetting, NT, SD, "none", std::nan(""), NULL, 100.0, std::nan(""), BuildS);
	};
	bool TimesBreached = false;

	// Device-only timing: the untimed h2d resets the in-place solver state.
	double ElapsedDeviceMs = 1.0e300;
	std::vector<double> DeviceSamples;
	std::vector<double> DeviceTimed;
	int DeviceFloor = 0, DeviceCeiling = 0;
	for (int r = 0; ; r++)
	{
		FillSolverObject(Scan, Parameters_R_Values, NT);
		Scan.SynchroniseFromHostToDevice(All);

		ArmWatchdog(RecordNan);
		auto T0 = std::chrono::steady_clock::now();
		Scan.Solve();
		Scan.InsertSynchronisationPoint();
		Scan.SynchroniseSolver();
		Scan.SynchroniseDevice();
		auto T1 = std::chrono::steady_clock::now();
		DisarmWatchdog();

		double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
		DeviceSamples.push_back(Ms);
		if (Ms > WatchdogSeconds() * 1000.0) { TimesBreached = true; break; }
		if (r == 0) continue;   // r == 0 is warm-up
		DeviceTimed.push_back(Ms);
		if (Ms < ElapsedDeviceMs) ElapsedDeviceMs = Ms;
		if (r == 1) RepeatBounds(DeviceTimed[0], TimingRepeats, DeviceFloor,
			DeviceCeiling);
		if (RepeatsDone(DeviceTimed, DeviceFloor, DeviceCeiling)) break;
	}

	// End-to-end timing: h2d, kernel, ActualState d2h.
	double ElapsedMs = 1.0e300;
	std::vector<double> EndToEndSamples;
	std::vector<double> EndToEndTimed;
	int EndToEndFloor = 0, EndToEndCeiling = 0;
	for (int r = 0; !TimesBreached; r++)
	{
		FillSolverObject(Scan, Parameters_R_Values, NT);

		ArmWatchdog(RecordNan);
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
		EndToEndSamples.push_back(Ms);
		if (Ms > WatchdogSeconds() * 1000.0) { TimesBreached = true; break; }
		if (r == 0) continue;   // r == 0 is warm-up
		EndToEndTimed.push_back(Ms);
		if (Ms < ElapsedMs) ElapsedMs = Ms;
		if (r == 1) RepeatBounds(EndToEndTimed[0], TimingRepeats,
			EndToEndFloor, EndToEndCeiling);
		if (RepeatsDone(EndToEndTimed, EndToEndFloor, EndToEndCeiling)) break;
	}

	if (TimesBreached)
	{
		RecordNan();
		cout << "WATCHDOG " << PROBLEM_NAME;
		if (StatesMode) cout << " states=" << SD;
		cout << " " << TimesMode << " " << TimesAlgorithm << " N=" << NT
		     << ": run exceeded the cap" << endl;
		return WatchdogExitCode;
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
	const double ErroredPercent = ErroredPct(Scan, NT, SD);
	std::cout << Scan.GetHost<PRECISION>(0, ActualTime) << std::endl;
	cout << "Total simulation time:           " << ElapsedMs << "ms" << endl;
	cout << "Device-only time (no h2d/d2h):   " << ElapsedDeviceMs << "ms" << endl;
	cout << "Ensemble size:                   " << NT << endl << endl;


	RecordResult(TimesAnalysis, TimesMode, TimesAlgorithm, TimesSettingKind,
		TimesSetting, NT, SD, "both", ElapsedMs, &EndToEndSamples, ErroredPercent,
		std::nan(""), BuildS);
	RecordResult(TimesAnalysis, TimesMode, TimesAlgorithm, TimesSettingKind,
		TimesSetting, NT, SD, "none", ElapsedDeviceMs, &DeviceSamples, ErroredPercent,
		std::nan(""), BuildS);

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
