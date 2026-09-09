// test_trial <trials.jsonl> <trial_id> <key> <row.json> <spec.json>: reads the trial file with trial.cuh, prints the parsed fields of one trial as key=value lines, writes its store row and finals spec, and checks the reader on escaped and null values. Exit 0 on success, 1 on a failed check, 2 on a usage or read error.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../trial.cuh"

static int Failures = 0;

static void Check(bool condition, const std::string& what)
{
	if (!condition)
	{
		std::printf("FAIL %s\n", what.c_str());
		Failures++;
	}
}

// Escapes, unicode, null, arrays, nested objects and numbers all read back.
static void ReaderChecks()
{
	std::string line = "{\"a\": \"q\\\"\\\\\\u00e9\\n\", \"b\": null, \"c\": [\"x\", \"y\"], \"d\": true, "
	                   "\"e\": -1.5e-3, \"f\": {\"k\": [1, {\"z\": \"]\"}]}, \"g\": [], \"h\": [1, 2]}";
	JsonObject object = JsonReader(line).ReadObject();
	Check(object["a"].kind == JsonValue::String && object["a"].text == "q\"\\\xc3\xa9\n", "escaped string");
	Check(object["b"].kind == JsonValue::Null, "null");
	Check(std::isnan(NumberField(object, "b")), "null number is nan");
	Check(object["c"].items.size() == 2 && object["c"].items[0] == "x" && object["c"].items[1] == "y", "string array");
	Check(BoolField(object, "d"), "bool");
	Check(NumberField(object, "e") == -1.5e-3 && object["e"].text == "-1.5e-3", "number text kept");
	Check(object["f"].kind == JsonValue::Object && object["f"].text == "{\"k\": [1, {\"z\": \"]\"}]}", "nested object raw");
	Check(object["g"].kind == JsonValue::Array && object["g"].items.empty() && object["g"].text == "[]", "empty array");
	Check(object["h"].items.size() == 2 && object["h"].items[1] == "2", "number array");
	Check(JsonString("a\"b\\c\n") == "\"a\\\"b\\\\c\\n\"", "string escaping");
	Check(JsonNumber(std::nan("")) == "null" && JsonNumber(0.1) == "0.10000000000000001", "number emission");
	bool threw = false;
	try { JsonReader("{\"a\": 1").ReadObject(); } catch (const std::exception&) { threw = true; }
	Check(threw, "truncated object throws");
}

int main(int argc, char** argv)
{
	if (argc != 6)
	{
		std::cerr << "usage: test_trial <trials.jsonl> <trial_id> <key> <row.json> <spec.json>" << std::endl;
		return 2;
	}
	ReaderChecks();
	try
	{
		std::vector<Trial> trials = ReadTrials(argv[1]);
		const Trial* found = NULL;
		for (size_t i = 0; i < trials.size(); i++)
			if (trials[i].trial_id == argv[2]) found = &trials[i];
		if (!found)
		{
			std::cerr << "no trial " << argv[2] << std::endl;
			return 2;
		}
		const Trial& t = *found;
		std::printf("count=%d\n", (int)trials.size());
		std::printf("kind=%s\n", t.kind.c_str());
		std::printf("leg=%s\n", t.leg.c_str());
		std::printf("ordinal=%d\n", t.ordinal);
		std::printf("problem=%s\n", t.problem.c_str());
		std::printf("algorithm=%s\n", t.algorithm.c_str());
		std::printf("controller=%s\n", t.controller.c_str());
		std::printf("n=%lld\n", t.n);
		std::printf("states_param=%d\n", t.StatesParam());
		std::printf("dt=%.17g\n", t.dt);
		std::printf("atol_nan=%d\n", std::isnan(t.atol) ? 1 : 0);
		std::printf("finals=%d\n", t.finals ? 1 : 0);
		std::printf("cold=%d\n", t.cold ? 1 : 0);
		std::printf("transfers=%s\n", t.transfers.size() == 2 ? (t.transfers[0] + "," + t.transfers[1]).c_str()
		                                                     : (t.transfers.empty() ? "" : t.transfers[0].c_str()));
		std::printf("lists_none=%d\n", t.Lists("none") ? 1 : 0);
		RowValues v;
		v.states = 3;
		v.min_ms = 1.5;
		v.samples_ms.push_back(2.0);
		v.samples_ms.push_back(1.5);
		v.errored_pct = 0.0;
		v.build_s = std::nan("");
		v.reason = "";
		v.finals = "";
		v.package_version = "abcdef123456+nvcc13.3";
		v.suite_rev = "rev";
		std::ofstream row(argv[4], std::ios::binary);
		row << RowText(t, "both", argv[3], v) << "\n";
		std::ofstream spec(argv[5], std::ios::binary);
		spec << FinalsSpecText(t, argv[3]) << "\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 2;
	}
	if (Failures)
	{
		std::printf("%d checks failed\n", Failures);
		return 1;
	}
	std::printf("ok\n");
	return 0;
}
