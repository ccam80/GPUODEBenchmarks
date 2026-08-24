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
#include <cstring>
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

// Per-repeat timing log, as in runner_scripts/wp_common.py.
static const char* SampleHeader =
	"analysis,problem,algorithm,mode,transfers,setting_kind,setting,n,states,repeat,ms";

// The identity of one timed point, shared by its timed legs.
struct SamplePoint
{
	std::string Analysis;
	std::string Algorithm;
	std::string Mode;
	std::string SettingKind;
	double Setting;
	int N;
	int States;
};

// Drop a leg's log, for the sweeps whose reduced file is rewritten.
static void ResetSamples(const std::string& Path)
{
	remove(Path.c_str());
}

// Append one row per attempt of one timed leg, warm-up as repeat 0.
static void AppendSamples(const std::string& Path, const SamplePoint& Point,
                          const std::string& Transfers,
                          const std::vector<double>& Samples)
{
	std::ifstream Probe(Path.c_str());
	bool Header = !Probe.good();
	Probe.close();

	char SettingText[32];
	snprintf(SettingText, sizeof(SettingText), "%.10g", Point.Setting);

	std::ofstream Out(Path.c_str(), std::ios::app);
	if (Header) Out << SampleHeader << "\n";
	for (size_t r = 0; r < Samples.size(); ++r)
	{
		char MsText[32];
		snprintf(MsText, sizeof(MsText), "%.6f", Samples[r]);
		Out << Point.Analysis << "," << PROBLEM_NAME << "," << Point.Algorithm
		    << "," << Point.Mode << "," << Transfers << "," << Point.SettingKind
		    << "," << SettingText << "," << Point.N << "," << Point.States
		    << "," << r << "," << MsText << "\n";
	}
}

// Per-run wall-clock watchdog; a hung kernel can only be stopped by process exit.
static double WatchdogSeconds()
{
	const char* env = std::getenv("BENCH_WATCHDOG_SECONDS");
	return env ? atof(env) : 120.0;
}

// Repeat floor and ceiling from the first timed run's duration, and the
// median/min spread that ends a leg once the floor is reached; mirrored in
// runner_scripts/wp_common.py and runner_scripts/watchdog.jl.
static void RepeatBounds(double FirstMs, int Cap, int& Floor, int& Ceiling)
{
	if      (FirstMs < 100.0)  { Floor = 20; Ceiling = 20; }
	else if (FirstMs < 3000.0) { Floor = 10; Ceiling = 10; }
	else if (FirstMs < 5000.0) { Floor = 5;  Ceiling = 10; }
	else                       { Floor = 3;  Ceiling = 10; }
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

// True when the timed runs settle the leg's minimum: the ceiling is reached,
// or the floor is and median/min - 1 is within 2%.
static bool RepeatsDone(const std::vector<double>& Timed, int Floor, int Ceiling)
{
	if ((int)Timed.size() >= Ceiling) return true;
	if ((int)Timed.size() < Floor) return false;
	double Min = *std::min_element(Timed.begin(), Timed.end());
	return MedianMs(Timed) / Min - 1.0 <= 0.02;
}

// --floor (BENCH_FLOOR): merge re-runs into the recorded file, keeping the
// lower time; mirrored in runner_scripts/resume.py.
static bool FloorEnabled()
{
	const char* env = std::getenv("BENCH_FLOOR");
	return env && *env && strcmp(env, "0") != 0;
}

// The lower of two times; nan loses to any finite value.
static double LowerTime(double Recorded, double New)
{
	if (std::isnan(Recorded)) return New;
	if (std::isnan(New)) return Recorded;
	return std::min(Recorded, New);
}

static std::vector<std::string> ReadLines(const std::string& Path)
{
	std::vector<std::string> Lines;
	std::ifstream In(Path.c_str());
	std::string Line;
	while (std::getline(In, Line)) Lines.push_back(Line);
	return Lines;
}

// The token as a double; nan when it does not parse.
static double ParseTime(const std::string& Token)
{
	const char* Start = Token.c_str();
	char* End = NULL;
	double Value = strtod(Start, &End);
	return (End == Start) ? std::nan("") : Value;
}

// First whitespace-separated token as a double; nan when there is none.
static double RowKey(const std::string& Line)
{
	std::istringstream Fields(Line);
	std::string Token;
	if (Fields >> Token) return ParseTime(Token);
	return std::nan("");
}

// --floor: merge one tab-separated times/states row, keeping the lower value
// per column; recorded fields beyond the merged columns survive.
static void MergeMinRow(const std::string& Path, long long Key,
                        const std::vector<double>& Values)
{
	std::vector<std::string> Lines = ReadLines(Path);
	bool Merged = false;
	for (size_t i = 0; i < Lines.size() && !Merged; ++i)
	{
		double Parsed = RowKey(Lines[i]);
		if (std::isnan(Parsed) || (long long)llround(Parsed) != Key) continue;
		std::istringstream Fields(Lines[i]);
		std::string Token;
		std::vector<std::string> Tokens;
		while (Fields >> Token) Tokens.push_back(Token);
		std::ostringstream Row;
		Row << Key;
		for (size_t c = 0; c < Values.size(); ++c)
		{
			double Recorded = std::nan("");
			if (c + 1 < Tokens.size()) Recorded = ParseTime(Tokens[c + 1]);
			Row << "\t" << LowerTime(Recorded, Values[c]);
		}
		for (size_t c = Values.size() + 1; c < Tokens.size(); ++c)
			Row << "\t" << Tokens[c];
		Lines[i] = Row.str();
		Merged = true;
	}
	if (!Merged)
	{
		std::ostringstream Row;
		Row << Key;
		for (size_t c = 0; c < Values.size(); ++c) Row << "\t" << Values[c];
		Lines.push_back(Row.str());
	}
	std::ofstream Out(Path.c_str());
	for (size_t i = 0; i < Lines.size(); ++i) Out << Lines[i] << "\n";
}

// --floor: merge one wp row, keeping the (time, error) pair with the lower time.
static void MergeWpRow(const std::string& Path, double Setting, double Ms,
                       double Err)
{
	std::ostringstream NewRow;
	NewRow.precision(12);
	NewRow << Setting << " " << Ms << " " << std::scientific << Err;
	std::vector<std::string> Lines = ReadLines(Path);
	bool Merged = false;
	for (size_t i = 0; i < Lines.size() && !Merged; ++i)
	{
		double Parsed = RowKey(Lines[i]);
		if (std::isnan(Parsed)) continue;
		double Tolerance = 1.0e-8 * std::max(std::fabs(Parsed), std::fabs(Setting));
		if (std::fabs(Parsed - Setting) > Tolerance) continue;
		Merged = true;
		std::istringstream Fields(Lines[i]);
		std::string Token;
		double Recorded = std::nan("");
		if (Fields >> Token && Fields >> Token) Recorded = ParseTime(Token);
		if (std::isnan(Recorded) || (!std::isnan(Ms) && Ms < Recorded))
			Lines[i] = NewRow.str();
	}
	if (!Merged) Lines.push_back(NewRow.str());
	std::ofstream Out(Path.c_str());
	for (size_t i = 0; i < Lines.size(); ++i) Out << Lines[i] << "\n";
}

// --floor's watchdog path: append only the rows whose key is not yet recorded.
static void AppendMissingRows(const std::string& Path,
                              const std::vector<std::string>& Rows)
{
	std::vector<double> Keys;
	{
		std::vector<std::string> Lines = ReadLines(Path);
		for (size_t i = 0; i < Lines.size(); ++i)
			Keys.push_back(RowKey(Lines[i]));
	}
	std::ofstream Out(Path.c_str(), std::ios::app);
	for (size_t i = 0; i < Rows.size(); ++i)
	{
		double Key = RowKey(Rows[i]);
		bool Present = false;
		for (size_t k = 0; k < Keys.size() && !Present; ++k)
		{
			if (std::isnan(Keys[k])) continue;
			double Tolerance = 1.0e-8 * std::max(std::fabs(Keys[k]),
			                                     std::fabs(Key));
			Present = std::fabs(Keys[k] - Key) <= Tolerance;
		}
		if (!Present) Out << Rows[i] << "\n";
	}
}

// Breach exit code; the runner NaN-fills the leg's remaining sizes.
static const int WatchdogExitCode = 42;

// Mode and algorithm names for filenames and watchdog messages.
static const char* ModeName      = (SOLVER == RK4) ? "fixed" : "adaptive";
static const char* AlgorithmName = (SOLVER == RK4) ? "classical-rk4"
                                                   : "cash-karp-54";
static bool StatesRun = false;   // set in main; states rows are SD-keyed

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
		if (FloorEnabled())
		{
			// The recorded rows already cover these points; NaN adds nothing.
			AppendMissingRows(WatchdogFile, WatchdogRows);
		}
		else
		{
			std::ofstream out(WatchdogFile.c_str(), std::ios::app);
			for (size_t i = 0; i < WatchdogRows.size(); ++i)
				out << WatchdogRows[i] << "\n";
			out.close();
		}
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
		string Mode = ModeName;
		string Algorithm = AlgorithmName;
		const std::string WpDir = DataDir("CPP");
		const std::string WpPath = WpDir + "MPGOS_wp_" + Mode + "_" + Algorithm + ".txt";
		// --floor merges into the recorded file, so it must not be truncated.
		ofstream wpfile(WpPath.c_str(), FloorEnabled()
			? (std::ios::out | std::ios::app) : std::ios::out);
		wpfile.precision(12);
		const std::string WpSamplesPath = WpDir + "MPGOS_samples_wp_" + Mode
			+ "_" + Algorithm + ".csv";
		// --floor re-runs gain a fresh series in the log instead.
		if (!FloorEnabled()) ResetSamples(WpSamplesPath);
		const std::string SettingKind = FixedMode ? "dt" : "tol";

		// Repeat ceiling; the count follows the first timed run's duration.
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
			std::vector<double> WpSamples;
			std::vector<double> WpTimed;
			int WpFloor = 0, WpCeiling = 0;
			for (int r = 0; ; r++)
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
				WpSamples.push_back(Ms);
				if (Ms > WatchdogSeconds() * 1000.0) { Breached = true; break; }
				if (r == 0) continue;   // r == 0 is warm-up
				WpTimed.push_back(Ms);
				if (Ms < BestMs) BestMs = Ms;
				if (r == 1) RepeatBounds(WpTimed[0], Repeats, WpFloor, WpCeiling);
				if (RepeatsDone(WpTimed, WpFloor, WpCeiling)) break;
			}
			// The h2d is outside the timed region; the ActualState d2h is inside.
			SamplePoint WpPoint = {"wp", Algorithm, Mode, SettingKind, Setting,
				NT, SD};
			AppendSamples(WpSamplesPath, WpPoint, "d2h", WpSamples);

			if (Breached)
			{
				if (FloorEnabled())
				{
					for (size_t sj = si; sj < Settings.size(); sj++)
						MergeWpRow(WpPath, Settings[sj], std::nan(""),
							std::nan(""));
				}
				else
				{
					for (size_t i = 0; i < NanRows.size(); ++i)
						wpfile << NanRows[i] << "\n";
					wpfile.flush();
				}
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

			if (FloorEnabled())
			{
				MergeWpRow(WpPath, Setting, BestMs, Err);
			}
			else
			{
				wpfile << Setting << " " << BestMs << " " << scientific << Err << fixed << "\n";
				wpfile.flush();
			}
			cout << "wp " << Mode << " setting=" << Setting << ": " << BestMs
			     << " ms, err=" << scientific << Err << fixed << endl;
		}
		wpfile.close();

		cout << "wp sweep finished!" << endl;
		return 0;
	}

	// Repeat ceiling; the count per leg follows its first timed run's
	// duration, and r == 0 is a discarded warm-up.
	const int TimingRepeats = 20;

	const std::string TimesAnalysis = StatesMode ? "states" : "times";
	const std::string TimesMode = ModeName;
	const std::string TimesAlgorithm = AlgorithmName;
	const std::string TimesDir = DataDir("CPP");
	const std::string TimesPath = TimesDir + "MPGOS_" + TimesAnalysis +
		"_" + TimesMode + "_" + TimesAlgorithm + ".txt";
	const std::string TimesSamplesPath = TimesDir + "MPGOS_samples_" +
		TimesAnalysis + "_" + TimesMode + "_" + TimesAlgorithm + ".csv";
	SamplePoint TimesPoint = {TimesAnalysis, TimesAlgorithm, TimesMode,
		"none", std::nan(""), NT, SD};
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
	std::vector<double> DeviceSamples;
	std::vector<double> DeviceTimed;
	int DeviceFloor = 0, DeviceCeiling = 0;
	for (int r = 0; ; r++)
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
		DeviceSamples.push_back(Ms);
		if (Ms > WatchdogSeconds() * 1000.0) { TimesBreached = true; break; }
		if (r == 0) continue;   // r == 0 is warm-up
		DeviceTimed.push_back(Ms);
		if (Ms < ElapsedDeviceMs) ElapsedDeviceMs = Ms;
		if (r == 1) RepeatBounds(DeviceTimed[0], TimingRepeats, DeviceFloor,
			DeviceCeiling);
		if (RepeatsDone(DeviceTimed, DeviceFloor, DeviceCeiling)) break;
	}
	AppendSamples(TimesSamplesPath, TimesPoint, "none", DeviceSamples);

	// End-to-end timing: h2d, kernel, ActualState d2h.
	double ElapsedMs = 1.0e300;
	std::vector<double> EndToEndSamples;
	std::vector<double> EndToEndTimed;
	int EndToEndFloor = 0, EndToEndCeiling = 0;
	for (int r = 0; !TimesBreached; r++)
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
		EndToEndSamples.push_back(Ms);
		if (Ms > WatchdogSeconds() * 1000.0) { TimesBreached = true; break; }
		if (r == 0) continue;   // r == 0 is warm-up
		EndToEndTimed.push_back(Ms);
		if (Ms < ElapsedMs) ElapsedMs = Ms;
		if (r == 1) RepeatBounds(EndToEndTimed[0], TimingRepeats,
			EndToEndFloor, EndToEndCeiling);
		if (RepeatsDone(EndToEndTimed, EndToEndFloor, EndToEndCeiling)) break;
	}
	AppendSamples(TimesSamplesPath, TimesPoint, "both", EndToEndSamples);

	if (TimesBreached)
	{
		if (FloorEnabled())
		{
			std::vector<double> NanValues(2, std::nan(""));
			if (StatesMode) NanValues.push_back(ParseTime(StatesBuild));
			MergeMinRow(TimesPath, StatesMode ? SD : NT, NanValues);
		}
		else
		{
			std::ofstream out(TimesPath.c_str(), std::ios::app);
			out << TimesNanRow[0] << "\n";
			out.close();
		}
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
	std::cout << Scan.GetHost<PRECISION>(0, ActualTime) << std::endl;
	cout << "Total simulation time:           " << ElapsedMs << "ms" << endl;
	cout << "Device-only time (no h2d/d2h):   " << ElapsedDeviceMs << "ms" << endl;
	cout << "Ensemble size:                   " << NT << endl << endl;


	if (FloorEnabled())
	{
		std::vector<double> RowValues;
		RowValues.push_back(ElapsedMs);
		RowValues.push_back(ElapsedDeviceMs);
		if (StatesMode) RowValues.push_back(ParseTime(StatesBuild));
		MergeMinRow(TimesPath, StatesMode ? SD : NT, RowValues);
	}
	else
	{
		ofstream datafile(TimesPath.c_str(), ios::app);
		if (StatesMode)
			datafile << SD << "\t" << ElapsedMs << "\t" << ElapsedDeviceMs
			         << "\t" << StatesBuild << "\n";
		else
			datafile << NT << "\t" << ElapsedMs << "\t" << ElapsedDeviceMs << "\n";
		datafile.close();
	}

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
