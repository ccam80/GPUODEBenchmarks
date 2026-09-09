// One MPGOS trial: Bench.exe --trials <jsonl> --trial <trial_id> --key <key> --transfers both,none --python <exe> --package-version <v> --suite-rev <r> --outcome <path> [--floor] [--build-s <s>]
// The trial's problem, algorithm, trajectory count, state count and precision are compile-time constants; the binary refuses a trial it was not built for.
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#ifndef PROBLEM_HEADER
	#error "define PROBLEM_HEADER, e.g. -DPROBLEM_HEADER=\"problems/lorenz.cuh\""
#endif
#ifndef SOLVER_CHOICE
	#define SOLVER_CHOICE RKCK45
#endif
#ifndef NT_VALUE
	#define NT_VALUE 8388608
#endif
#ifndef PRECISION_TYPE
	#define PRECISION_TYPE float
#endif

#include PROBLEM_HEADER
#include "problems/stubs.cuh"
#include "SingleSystem_PerThread_Interface.cuh"
// Generated from runner_scripts/protocol.toml by protocol.py before every build.
#include "protocol.h"
#include "grid.cuh"
#include "trial.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace std;

// Solver Configuration
#define SOLVER SOLVER_CHOICE
typedef PRECISION_TYPE PRECISION;
const int NT = NT_VALUE;
const int SD   = PROBLEM_SD;   // SystemDimension
const int NCP  = PROBLEM_NCP;  // NumberOfControlParameters
const int NSP  = 0;     // NumberOfSharedParameters
const int NISP = 1;     // NumberOfIntegerSharedParameters (run budget)
const int NE   = 0;     // NumberOfEvents
const int NA   = 0;     // NumberOfAccessories
const int NIA  = 1;     // NumberOfIntegerAccessories (start clock)
const int NDO  = 0;     // NumberOfPointsOfDenseOutput: nothing reads it, and
                        // storing it is work the other suites do not do

typedef ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> Solver;

static const bool FixedSolver = (SOLVER == RK4);
static const char* const SolverAlgorithm = FixedSolver ? "classical-rk4" : "cash-karp-54";
static const char* const BuiltPrecision = (sizeof(PRECISION) == 8) ? "float64" : "float32";
static const int WatchdogExitCode = PROTOCOL_WATCHDOG_EXIT_CODE;

// ------------------------------------------------------------------ options

struct Options
{
	std::string trials, trial_id, key, python, package_version, suite_rev, outcome;
	std::vector<std::string> transfers;
	bool floor;
	double build_s;
	Options() : floor(false), build_s(std::nan("")) {}
};

static void Usage()
{
	std::cerr << "usage: Bench.exe --trials <jsonl> --trial <trial_id> --key <key> --transfers both,none "
	             "--python <exe> --package-version <v> --suite-rev <r> --outcome <path> [--floor] [--build-s <s>]"
	          << std::endl;
	exit(2);
}

static Options ParseOptions(int argc, char* argv[])
{
	Options o;
	for (int i = 1; i < argc; i++)
	{
		std::string flag = argv[i];
		if (flag == "--floor") { o.floor = true; continue; }
		if (i + 1 >= argc) Usage();
		std::string value = argv[++i];
		if (flag == "--trials") o.trials = value;
		else if (flag == "--trial") o.trial_id = value;
		else if (flag == "--key") o.key = value;
		else if (flag == "--python") o.python = value;
		else if (flag == "--package-version") o.package_version = value;
		else if (flag == "--suite-rev") o.suite_rev = value;
		else if (flag == "--outcome") o.outcome = value;
		else if (flag == "--build-s") o.build_s = std::strtod(value.c_str(), NULL);
		else if (flag == "--transfers")
		{
			std::stringstream parts(value);
			std::string item;
			while (std::getline(parts, item, ','))
				if (!item.empty()) o.transfers.push_back(item);
		}
		else Usage();
	}
	if (o.trials.empty() || o.trial_id.empty() || o.key.empty() || o.python.empty()
		|| o.outcome.empty() || o.transfers.empty())
		Usage();
	for (size_t i = 0; i < o.transfers.size(); i++)
		if (o.transfers[i] != "both" && o.transfers[i] != "none")
		{
			std::cerr << "transfers must be both or none, got " << o.transfers[i] << std::endl;
			exit(2);
		}
	return o;
}

// ----------------------------------------------------------------- store CLI

// Run a command line; on Windows the whole line is wrapped in quotes so cmd keeps the quoted program path.
static int Shell(const std::string& command)
{
#ifdef _WIN32
	return system(("\"" + command + "\"").c_str());
#else
	return system(command.c_str());
#endif
}

static std::string Quote(const std::string& text)
{
	return "\"" + text + "\"";
}

static void WriteText(const std::string& path, const std::string& text)
{
	std::ofstream out(path.c_str(), std::ios::binary);
	if (!out)
		throw std::runtime_error("cannot write " + path);
	out << text;
}

// Rows through `store.py record`; a failed record is fatal because the row would be lost.
static void RecordRows(const Options& o, const std::vector<std::string>& rows)
{
	if (rows.empty()) return;
	std::string text = "[";
	for (size_t i = 0; i < rows.size(); i++)
		text += (i ? ",\n" : "\n") + rows[i];
	text += "\n]\n";
	std::string path = o.trials + ".cpp_rows.json";
	WriteText(path, text);
	std::string command = Quote(o.python) + " runner_scripts/store.py record " + Quote(path);
	if (o.floor) command += " --floor";
	int status = Shell(command);
	if (status != 0)
	{
		std::cerr << "store.py record failed (" << status << "): " << command << std::endl;
		exit(2);
	}
}

// Finals through `store.py finals`: every trajectory's final state, final time and an empty retcode (MPGOS reports none).
static std::string RecordFinals(const Options& o, const Trial& trial, Solver& Scan)
{
	std::string csv = o.trials + ".cpp_finals.csv";
	std::string spec = o.trials + ".cpp_finals_spec.json";
	{
		std::ofstream out(csv.c_str(), std::ios::binary);
		if (!out)
			throw std::runtime_error("cannot write " + csv);
		for (int c = 0; c < SD; c++)
			out << "s" << (c + 1) << ",";
		out << "t_final,retcode\n";
		const char* format = (sizeof(PRECISION) == 8) ? "%.17g" : "%.9g";
		char buf[40];
		for (int tid = 0; tid < NT; tid++)
		{
			for (int c = 0; c < SD; c++)
			{
				snprintf(buf, sizeof(buf), format, (double)Scan.GetHost<PRECISION>(tid, ActualState, c));
				out << buf << ",";
			}
			snprintf(buf, sizeof(buf), "%.17g", (double)Scan.GetHost<PRECISION>(tid, ActualTime));
			out << buf << ",\n";
		}
	}
	WriteText(spec, FinalsSpecText(trial, o.key) + "\n");
	std::string command = Quote(o.python) + " runner_scripts/store.py finals " + Quote(spec) + " " + Quote(csv);
	int status = Shell(command);
	if (status != 0)
	{
		std::cerr << "store.py finals failed (" << status << "): " << command << std::endl;
		exit(2);
	}
	// The CSV is n rows of every state; the parquet file now holds them.
	std::remove(csv.c_str());
	return "finals/" + trial.trial_id + ".parquet";
}

// `<trials>.progress`: the trial under way, for the driver's hard-exit bookkeeping.
static void WriteProgress(const Options& o)
{
	time_t now = time(NULL);
	char stamp[32];
	strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
	WriteText(o.trials + ".progress",
	          "{\"trial_id\": " + JsonString(o.trial_id) + ", \"started_utc\": \"" + stamp + "\"}\n");
}

// ------------------------------------------------------------------ repeats

static double WatchdogSeconds()
{
	return PROTOCOL_WATCHDOG_SECONDS;
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

// ----------------------------------------------------------------- watchdog

// A hung kernel can only be stopped by process exit; the driver records the abandoned rows after exit 3.
static std::atomic<long long> WatchdogDeadlineMs(0);   // 0 = disarmed
static std::string WatchdogLabel;

static long long NowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Margin over the soft cap, so a run that returns late is a timeout row, not a hard exit.
static void ArmWatchdog()
{
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
		std::cout << "WATCHDOG " << WatchdogLabel << ": run never returned" << std::endl;
		std::_Exit(WatchdogExitCode);
	}
}

// ------------------------------------------------------------------- solve

enum Outcome { Ok, Timeout, Oom, Error };

static const char* OutcomeName(Outcome outcome)
{
	switch (outcome)
	{
		case Ok: return "ok";
		case Timeout: return "timeout";
		case Oom: return "oom";
		default: return "error";
	}
}

struct LegResult
{
	Outcome outcome;
	double min_ms;
	std::vector<double> samples;
	std::string reason;
	LegResult() : outcome(Ok), min_ms(std::nan("")) {}
};

static bool IsOom(const std::string& name, const std::string& message)
{
	std::string lower = message;
	std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
	return name == "cudaErrorMemoryAllocation" || lower.find("out of memory") != std::string::npos;
}

// error: <Type>: <message[:200]>, or oom: <Type>: <message> when the message names memory.
static std::string FailureReason(const std::string& type, const std::string& message, Outcome& outcome)
{
	outcome = IsOom(type, message) ? Oom : Error;
	return std::string(outcome == Oom ? "oom: " : "error: ") + type + ": " + message.substr(0, 200);
}

void FillSolverObject(Solver& Scan, const std::vector<PRECISION>& Values, PRECISION Duration)
{
	PRECISION X0[SD];
	ProblemInitialState<PRECISION>(X0);
	for (int k = 0; k < NT; k++)
	{
		Scan.SetHost(k, TimeDomain, 0, (PRECISION)0);
		Scan.SetHost(k, TimeDomain, 1, Duration);
		// Solve() continues from ActualTime, so reset it to re-integrate from t=0.
		Scan.SetHost(k, ActualTime, (PRECISION)0);
		for (int c = 0; c < SD; c++)
			Scan.SetHost(k, ActualState, c, X0[c]);
		Scan.SetHost(k, ControlParameters, 0, Values[k]);
	}
}

// Time one transfers leg: `both` is h2d, kernel and the ActualState d2h; `none` the kernel alone after an untimed h2d. One untimed warm-up, then the protocol repeat schedule.
static LegResult TimeLeg(Solver& Scan, const std::vector<PRECISION>& Values, PRECISION Duration, bool Both)
{
	LegResult result;
	std::vector<double> Timed;
	int Floor = 0, Ceiling = 0;
	const double CapMs = WatchdogSeconds() * 1000.0;
	for (int r = 0; ; r++)
	{
		FillSolverObject(Scan, Values, Duration);
		if (!Both)
			Scan.SynchroniseFromHostToDevice(All);

		ArmWatchdog();
		auto T0 = std::chrono::steady_clock::now();
		if (Both)
			Scan.SynchroniseFromHostToDevice(All);
		Scan.Solve();
		Scan.InsertSynchronisationPoint();
		Scan.SynchroniseSolver();
		if (Both)
			Scan.SynchroniseFromDeviceToHost(ActualState);
		Scan.SynchroniseDevice();
		auto T1 = std::chrono::steady_clock::now();
		DisarmWatchdog();

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			result.reason = FailureReason(cudaGetErrorName(err), cudaGetErrorString(err), result.outcome);
			result.min_ms = std::nan("");
			return result;
		}

		double Ms = std::chrono::duration<double, std::milli>(T1 - T0).count();
		result.samples.push_back(Ms);
		if (Ms > CapMs)
		{
			result.outcome = Timeout;
			result.min_ms = std::nan("");
			char text[96];
			snprintf(text, sizeof(text), "timeout: %.1f ms exceeded the %g s cap", Ms, WatchdogSeconds());
			result.reason = text;
			return result;
		}
		if (r == 0) continue;   // r == 0 is warm-up
		Timed.push_back(Ms);
		if (std::isnan(result.min_ms) || Ms < result.min_ms) result.min_ms = Ms;
		if (r == 1) RepeatBounds(Timed[0], PROTOCOL_REPEAT_CAP, Floor, Ceiling);
		if (RepeatsDone(Timed, Floor, Ceiling)) break;
	}
	return result;
}

// Percent of trajectories the store's errored_mask flags: a non-finite state, or a final time off the duration by over 1e-4 relative. MPGOS reports no retcode.
static double ErroredPct(Solver& Scan, double Duration)
{
	int Bad = 0;
	for (int tid = 0; tid < NT; ++tid)
	{
		bool bad = false;
		for (int c = 0; c < SD && !bad; ++c)
			if (!std::isfinite((double) Scan.GetHost<PRECISION>(tid, ActualState, c)))
				bad = true;
		double t = (double) Scan.GetHost<PRECISION>(tid, ActualTime);
		if (!(std::fabs(t - Duration) <= 1e-4 * std::fabs(Duration)))
			bad = true;
		if (bad) ++Bad;
	}
	return 100.0 * (double) Bad / (double) NT;
}

// --------------------------------------------------------------------- rows

static RowValues BaseValues(const Options& o)
{
	RowValues v;
	v.states = SD;
	v.min_ms = std::nan("");
	v.errored_pct = std::nan("");
	v.build_s = o.build_s;
	v.package_version = o.package_version;
	v.suite_rev = o.suite_rev;
	return v;
}

// NaN rows for every higher ordinal of the leg that lists these transfers: they are not run.
static std::vector<std::string> AbandonRows(const Options& o, const std::vector<Trial>& trials,
                                            const Trial& mine, const std::string& transfers, Outcome why)
{
	std::vector<std::string> rows;
	RowValues v = BaseValues(o);
	v.reason = std::string("abandoned: ") + OutcomeName(why) + " at ordinal " + std::to_string(mine.ordinal);
	for (size_t i = 0; i < trials.size(); i++)
	{
		const Trial& t = trials[i];
		if (t.kind != "solve" || t.leg != mine.leg || t.ordinal <= mine.ordinal || !t.Lists(transfers))
			continue;
		rows.push_back(RowText(t, transfers, o.key, v));
	}
	return rows;
}

struct OutcomeLog
{
	std::string path;
	std::vector<std::string> lines;
	void Add(const std::string& transfers, Outcome outcome)
	{
		lines.push_back(transfers + " " + OutcomeName(outcome));
		std::string text;
		for (size_t i = 0; i < lines.size(); i++) text += lines[i] + "\n";
		WriteText(path, text);
	}
};

// Rows with one reason for every requested transfers leg, recorded and logged; the trial ran nothing.
static int FailAll(const Options& o, const std::vector<Trial>& trials, const Trial& trial, OutcomeLog& log,
                   const std::string& reason, Outcome outcome)
{
	std::vector<std::string> rows;
	RowValues v = BaseValues(o);
	v.reason = reason;
	for (size_t i = 0; i < o.transfers.size(); i++)
	{
		rows.push_back(RowText(trial, o.transfers[i], o.key, v));
		if (outcome == Oom || outcome == Timeout)
		{
			std::vector<std::string> more = AbandonRows(o, trials, trial, o.transfers[i], outcome);
			rows.insert(rows.end(), more.begin(), more.end());
		}
	}
	RecordRows(o, rows);
	for (size_t i = 0; i < o.transfers.size(); i++)
		log.Add(o.transfers[i], outcome);
	std::cout << "cpp " << trial.problem << " " << trial.algorithm << " " << trial.controller
	          << " n=" << trial.n << ": " << reason << std::endl;
	return 0;
}

// The trial this binary was built for, or an explanation of the mismatch.
static std::string BinaryMismatch(const Trial& trial)
{
	if (trial.problem != PROBLEM_NAME)
		return "built for " PROBLEM_NAME ", trial is " + trial.problem;
	if (trial.n != NT)
		return "built for n = " + std::to_string(NT) + ", trial has n = " + std::to_string(trial.n);
	if (trial.algorithm != SolverAlgorithm)
		return std::string("built for ") + SolverAlgorithm + ", trial is " + trial.algorithm;
	if (trial.precision != BuiltPrecision)
		return std::string("built for ") + BuiltPrecision + ", trial is " + trial.precision;
	int states = trial.StatesParam();
	if (states >= 0 && states != SD)
		return "built for " + std::to_string(SD) + " states, trial has " + std::to_string(states);
	return "";
}

int main(int argc, char* argv[])
{
	Options o = ParseOptions(argc, argv);
	std::vector<Trial> trials = ReadTrials(o.trials);
	const Trial* found = NULL;
	for (size_t i = 0; i < trials.size(); i++)
		if (trials[i].kind == "solve" && trials[i].trial_id == o.trial_id)
			found = &trials[i];
	if (!found)
	{
		std::cerr << "no solve trial " << o.trial_id << " in " << o.trials << std::endl;
		return 2;
	}
	const Trial& trial = *found;
	WriteProgress(o);
	std::string mismatch = BinaryMismatch(trial);
	if (!mismatch.empty())
	{
		std::cerr << "Bench.exe " << mismatch << std::endl;
		return 2;
	}

	OutcomeLog log;
	log.path = o.outcome;
	WatchdogLabel = std::string(PROBLEM_NAME) + " " + SolverAlgorithm + " n=" + std::to_string(NT);
	std::thread(WatchdogMain).detach();

	if (trial.parameter != PROBLEM_PARAMETER)
		return FailAll(o, trials, trial, log, std::string("error: ValueError: ") + PROBLEM_NAME
			+ " sweeps " + PROBLEM_PARAMETER + ", not " + trial.parameter, Error);
	if (trial.controller != "fixed" && trial.controller != "default")
		return FailAll(o, trials, trial, log, "error: unknown controller " + trial.controller, Error);
	if (FixedSolver != (trial.controller == "fixed"))
		return FailAll(o, trials, trial, log, std::string("error: ValueError: ") + SolverAlgorithm
			+ " runs under controller " + (FixedSolver ? "fixed" : "default") + ", not "
			+ trial.controller, Error);
	if (FixedSolver && !(trial.dt > 0.0))
		return FailAll(o, trials, trial, log, "error: ValueError: a fixed step needs dt > 0", Error);

	std::vector<PRECISION> Values = Grid<PRECISION>(trial.grid_scale, trial.grid_min, trial.grid_max, NT);
	const PRECISION Duration = (PRECISION)trial.duration;

	int SelectedDevice = SelectDeviceByClosestRevision(3, 5);
	Solver* ScanPtr = NULL;
	try
	{
		ScanPtr = new Solver(SelectedDevice);
	}
	catch (const std::exception& e)
	{
		Outcome outcome;
		std::string reason = FailureReason("RuntimeError", e.what(), outcome);
		return FailAll(o, trials, trial, log, reason, outcome);
	}
	Solver& Scan = *ScanPtr;

	Scan.SolverOption(ThreadsPerBlock, 32);
	// The fixed step, or dt0 when the trial pins one; NaN leaves the package default.
	if (!std::isnan(trial.dt))
		Scan.SolverOption(InitialTimeStep, trial.dt);
	if (!std::isnan(trial.dt_min))
		Scan.SolverOption(MinimumTimeStep, trial.dt_min);
	if (!std::isnan(trial.dt_max))
		Scan.SolverOption(MaximumTimeStep, trial.dt_max);
	if (!FixedSolver)
		for (int c = 0; c < SD; c++)
		{
			Scan.SolverOption(RelativeTolerance, c, trial.rtol);
			Scan.SolverOption(AbsoluteTolerance, c, trial.atol);
		}

	// Device-side run budget, 1.25 over the host cap; see problems/stubs.cuh.
	int ClockKHz = 0;
	cudaDeviceGetAttribute(&ClockKHz, cudaDevAttrClockRate, SelectedDevice);
	if (ClockKHz <= 0) ClockKHz = 3000000;
	long long BudgetCycles = (long long)(WatchdogSeconds() * 1.25 * ClockKHz * 1000.0);
	Scan.SetHost(IntegerSharedParameters, 0, (int)(BudgetCycles >> WATCHDOG_CLOCK_SHIFT));

	std::string finals;
	for (size_t li = 0; li < o.transfers.size(); li++)
	{
		const std::string& transfers = o.transfers[li];
		LegResult leg = TimeLeg(Scan, Values, Duration, transfers == "both");
		RowValues v = BaseValues(o);
		v.min_ms = leg.min_ms;
		v.samples_ms = leg.samples;
		v.reason = leg.reason;
		std::vector<std::string> rows;
		if (leg.outcome == Ok)
		{
			// Untimed full d2h for the final states and times of every trajectory.
			Scan.SynchroniseFromDeviceToHost(All);
			Scan.SynchroniseDevice();
			v.errored_pct = ErroredPct(Scan, trial.duration);
			if (trial.finals && finals.empty())
				finals = RecordFinals(o, trial, Scan);
			v.finals = finals;
		}
		rows.push_back(RowText(trial, transfers, o.key, v));
		if (leg.outcome == Timeout || leg.outcome == Oom)
		{
			std::vector<std::string> more = AbandonRows(o, trials, trial, transfers, leg.outcome);
			rows.insert(rows.end(), more.begin(), more.end());
		}
		RecordRows(o, rows);
		log.Add(transfers, leg.outcome);
		std::cout << "cpp " << trial.problem << " " << trial.algorithm << " " << trial.controller
		          << " n=" << NT << " " << transfers << ": ";
		if (leg.outcome == Ok)
			std::cout << leg.min_ms << " ms over " << leg.samples.size() - 1 << " timed runs, errored "
			          << v.errored_pct << "%" << std::endl;
		else
			std::cout << leg.reason << std::endl;
	}
	delete ScanPtr;
	return 0;
}
