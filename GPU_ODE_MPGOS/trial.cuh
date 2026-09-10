// Trial lines read into a Trial and the store row text Bench.cu records; host code only.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// The run spec columns a trial carries, in the store's table order (store.TRIAL_FIELDS).
static const char* const TRIAL_FIELDS[] = {
	"problem", "system_params", "duration", "precision",
	"parameter", "grid_scale", "grid_min", "grid_max", "n", "grid_dtype",
	"algorithm", "controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains",
	"newton_atol", "newton_rtol", "package"};
static const int TRIAL_FIELD_COUNT = sizeof(TRIAL_FIELDS) / sizeof(TRIAL_FIELDS[0]);

// One JSON value: scalars decoded, arrays and objects kept as raw text, scalar arrays also itemised.
struct JsonValue
{
	enum Kind { String, Number, Bool, Null, Array, Object };
	Kind kind;
	std::string text;                 // decoded string, raw number text, "true"/"false", "null", or the raw array/object text
	std::vector<std::string> items;   // scalar array elements, decoded
};

typedef std::map<std::string, JsonValue> JsonObject;

class JsonReader
{
public:
	explicit JsonReader(const std::string& text) : Text(text), Pos(0) {}

	JsonObject ReadObject()
	{
		JsonObject object;
		SkipSpace();
		Expect('{');
		SkipSpace();
		if (Peek() == '}') { Pos++; return object; }
		for (;;)
		{
			SkipSpace();
			std::string key = ReadString();
			SkipSpace();
			Expect(':');
			object[key] = ReadValue();
			SkipSpace();
			char c = Next();
			if (c == '}') return object;
			if (c != ',') Fail("expected , or }");
		}
	}

private:
	const std::string& Text;
	size_t Pos;

	void Fail(const std::string& what) const
	{
		throw std::runtime_error("trial JSON: " + what + " at offset " + std::to_string(Pos));
	}
	char Peek() const { return Pos < Text.size() ? Text[Pos] : '\0'; }
	char Next() { if (Pos >= Text.size()) Fail("unexpected end"); return Text[Pos++]; }
	void Expect(char c) { if (Next() != c) Fail(std::string("expected ") + c); }
	void SkipSpace() { while (Pos < Text.size() && std::isspace((unsigned char)Text[Pos])) Pos++; }

	static void AppendUtf8(std::string& out, unsigned code)
	{
		if (code < 0x80) out += (char)code;
		else if (code < 0x800) { out += (char)(0xC0 | (code >> 6)); out += (char)(0x80 | (code & 0x3F)); }
		else { out += (char)(0xE0 | (code >> 12)); out += (char)(0x80 | ((code >> 6) & 0x3F)); out += (char)(0x80 | (code & 0x3F)); }
	}

	std::string ReadString()
	{
		Expect('"');
		std::string out;
		for (;;)
		{
			char c = Next();
			if (c == '"') return out;
			if (c != '\\') { out += c; continue; }
			char e = Next();
			switch (e)
			{
				case '"': out += '"'; break;
				case '\\': out += '\\'; break;
				case '/': out += '/'; break;
				case 'b': out += '\b'; break;
				case 'f': out += '\f'; break;
				case 'n': out += '\n'; break;
				case 'r': out += '\r'; break;
				case 't': out += '\t'; break;
				case 'u':
				{
					unsigned code = 0;
					for (int i = 0; i < 4; i++)
					{
						char h = Next();
						code <<= 4;
						if (h >= '0' && h <= '9') code |= h - '0';
						else if (h >= 'a' && h <= 'f') code |= h - 'a' + 10;
						else if (h >= 'A' && h <= 'F') code |= h - 'A' + 10;
						else Fail("bad \\u escape");
					}
					AppendUtf8(out, code);
					break;
				}
				default: Fail("bad escape");
			}
		}
	}

	// The raw text of a nested array or object, brackets balanced outside strings.
	std::string ReadRaw()
	{
		size_t start = Pos;
		int depth = 0;
		bool inString = false;
		for (;;)
		{
			char c = Next();
			if (inString)
			{
				if (c == '\\') Next();
				else if (c == '"') inString = false;
				continue;
			}
			if (c == '"') inString = true;
			else if (c == '[' || c == '{') depth++;
			else if (c == ']' || c == '}')
			{
				depth--;
				if (depth == 0) return Text.substr(start, Pos - start);
			}
		}
	}

	JsonValue ReadValue()
	{
		SkipSpace();
		JsonValue value;
		char c = Peek();
		if (c == '"')
		{
			value.kind = JsonValue::String;
			value.text = ReadString();
		}
		else if (c == '[')
		{
			value.kind = JsonValue::Array;
			size_t start = Pos;
			Pos++;
			SkipSpace();
			if (Peek() == ']') { Pos++; value.text = "[]"; return value; }
			for (;;)
			{
				SkipSpace();
				if (Peek() == '[' || Peek() == '{')
				{
					Pos = start;
					value.items.clear();
					value.text = ReadRaw();
					return value;
				}
				JsonValue item = ReadValue();
				value.items.push_back(item.text);
				SkipSpace();
				char d = Next();
				if (d == ']') break;
				if (d != ',') Fail("expected , or ]");
			}
			value.text = Text.substr(start, Pos - start);
		}
		else if (c == '{')
		{
			value.kind = JsonValue::Object;
			value.text = ReadRaw();
		}
		else if (Text.compare(Pos, 4, "true") == 0) { value.kind = JsonValue::Bool; value.text = "true"; Pos += 4; }
		else if (Text.compare(Pos, 5, "false") == 0) { value.kind = JsonValue::Bool; value.text = "false"; Pos += 5; }
		else if (Text.compare(Pos, 4, "null") == 0) { value.kind = JsonValue::Null; value.text = "null"; Pos += 4; }
		else
		{
			size_t start = Pos;
			while (Pos < Text.size() && (std::isdigit((unsigned char)Text[Pos]) || Text[Pos] == '-'
				|| Text[Pos] == '+' || Text[Pos] == '.' || Text[Pos] == 'e' || Text[Pos] == 'E'))
				Pos++;
			if (Pos == start) Fail("bad value");
			value.kind = JsonValue::Number;
			value.text = Text.substr(start, Pos - start);
		}
		return value;
	}
};

// A trial line: typed fields plus the raw object for row emission.
struct Trial
{
	JsonObject raw;
	std::string trial_id, kind, leg, axis;
	std::string problem, system_params, precision, parameter, grid_scale, grid_dtype;
	std::string algorithm, controller, gains, package;
	double duration, grid_min, grid_max, dt, dt_min, dt_max, atol, rtol, newton_atol, newton_rtol;
	double watchdog_s;
	long long n;
	int ordinal;
	bool finals, cold;
	std::vector<std::string> transfers;

	// The states count system_params carries, or -1 when it names none.
	int StatesParam() const
	{
		size_t at = system_params.find("\"states\"");
		if (at == std::string::npos) return -1;
		at = system_params.find(':', at);
		if (at == std::string::npos) return -1;
		return (int)std::strtol(system_params.c_str() + at + 1, NULL, 10);
	}

	bool Lists(const std::string& transfer) const
	{
		for (size_t i = 0; i < transfers.size(); i++)
			if (transfers[i] == transfer) return true;
		return false;
	}
};

inline const JsonValue& Field(const JsonObject& object, const std::string& name)
{
	JsonObject::const_iterator it = object.find(name);
	if (it == object.end())
		throw std::runtime_error("trial lacks field " + name);
	return it->second;
}

inline std::string TextField(const JsonObject& object, const std::string& name)
{
	return Field(object, name).text;
}

// A number field; null is NaN.
inline double NumberField(const JsonObject& object, const std::string& name)
{
	const JsonValue& value = Field(object, name);
	if (value.kind == JsonValue::Null) return std::nan("");
	if (value.kind != JsonValue::Number)
		throw std::runtime_error("trial field " + name + " is not a number");
	return std::strtod(value.text.c_str(), NULL);
}

inline bool BoolField(const JsonObject& object, const std::string& name)
{
	return Field(object, name).text == "true";
}

inline Trial ParseTrial(const std::string& line)
{
	Trial t;
	t.raw = JsonReader(line).ReadObject();
	for (int i = 0; i < TRIAL_FIELD_COUNT; i++)
		Field(t.raw, TRIAL_FIELDS[i]);
	t.trial_id = TextField(t.raw, "trial_id");
	t.kind = TextField(t.raw, "kind");
	t.leg = TextField(t.raw, "leg");
	t.axis = TextField(t.raw, "axis");
	t.ordinal = (int)NumberField(t.raw, "ordinal");
	t.finals = BoolField(t.raw, "finals");
	t.cold = BoolField(t.raw, "cold");
	t.watchdog_s = NumberField(t.raw, "watchdog_s");
	t.transfers = Field(t.raw, "transfers").items;
	t.problem = TextField(t.raw, "problem");
	t.system_params = TextField(t.raw, "system_params");
	t.duration = NumberField(t.raw, "duration");
	t.precision = TextField(t.raw, "precision");
	t.parameter = TextField(t.raw, "parameter");
	t.grid_scale = TextField(t.raw, "grid_scale");
	t.grid_min = NumberField(t.raw, "grid_min");
	t.grid_max = NumberField(t.raw, "grid_max");
	t.n = (long long)NumberField(t.raw, "n");
	t.grid_dtype = TextField(t.raw, "grid_dtype");
	t.algorithm = TextField(t.raw, "algorithm");
	t.controller = TextField(t.raw, "controller");
	t.dt = NumberField(t.raw, "dt");
	t.dt_min = NumberField(t.raw, "dt_min");
	t.dt_max = NumberField(t.raw, "dt_max");
	t.atol = NumberField(t.raw, "atol");
	t.rtol = NumberField(t.raw, "rtol");
	t.gains = TextField(t.raw, "gains");
	t.newton_atol = NumberField(t.raw, "newton_atol");
	t.newton_rtol = NumberField(t.raw, "newton_rtol");
	t.package = TextField(t.raw, "package");
	return t;
}

// Every trial of a JSONL file, in file order.
inline std::vector<Trial> ReadTrials(const std::string& path)
{
	std::ifstream in(path.c_str());
	if (!in)
		throw std::runtime_error("cannot open " + path);
	std::vector<Trial> trials;
	std::string line;
	while (std::getline(in, line))
	{
		size_t first = line.find_first_not_of(" \t\r\n");
		if (first == std::string::npos) continue;
		trials.push_back(ParseTrial(line));
	}
	return trials;
}

// ---------------------------------------------------------------- emission

inline std::string JsonString(const std::string& text)
{
	std::string out = "\"";
	for (size_t i = 0; i < text.size(); i++)
	{
		unsigned char c = (unsigned char)text[i];
		if (c == '"') out += "\\\"";
		else if (c == '\\') out += "\\\\";
		else if (c == '\n') out += "\\n";
		else if (c == '\r') out += "\\r";
		else if (c == '\t') out += "\\t";
		else if (c < 0x20)
		{
			char buf[8];
			snprintf(buf, sizeof(buf), "\\u%04x", c);
			out += buf;
		}
		else out += (char)c;
	}
	return out + "\"";
}

// A JSON number with every double digit; NaN is null.
inline std::string JsonNumber(double value)
{
	if (std::isnan(value)) return "null";
	char buf[40];
	snprintf(buf, sizeof(buf), "%.17g", value);
	return buf;
}

// The raw JSON text of a trial field, as the trial file spelled it.
inline std::string RawField(const Trial& trial, const std::string& name)
{
	const JsonValue& value = Field(trial.raw, name);
	if (value.kind == JsonValue::String) return JsonString(value.text);
	return value.text;
}

// The spec fields as JSON members, transfers and key appended when given.
inline std::string SpecMembers(const Trial& trial, const std::string& transfers, const std::string& key)
{
	std::string out;
	for (int i = 0; i < TRIAL_FIELD_COUNT; i++)
	{
		if (i) out += ",";
		out += JsonString(TRIAL_FIELDS[i]) + ":" + RawField(trial, TRIAL_FIELDS[i]);
	}
	if (!transfers.empty()) out += ",\"transfers\":" + JsonString(transfers);
	if (!key.empty()) out += ",\"key\":" + JsonString(key);
	return out;
}

// The spec object store.py finals takes: the trial fields and the key.
inline std::string FinalsSpecText(const Trial& trial, const std::string& key)
{
	return "{" + SpecMembers(trial, "", key) + "}";
}

// The values of one store row beside its spec.
struct RowValues
{
	int states;
	double min_ms;
	std::vector<double> samples_ms;
	double errored_pct;
	double build_s;
	std::string reason;
	std::string finals;
	std::string package_version;
	std::string suite_rev;
};

// One store row as JSON.
inline std::string RowText(const Trial& trial, const std::string& transfers, const std::string& key,
                           const RowValues& values)
{
	std::string out = "{" + SpecMembers(trial, transfers, key);
	out += ",\"states\":" + std::to_string(values.states);
	out += ",\"min_ms\":" + JsonNumber(values.min_ms);
	out += ",\"samples_ms\":[";
	for (size_t i = 0; i < values.samples_ms.size(); i++)
		out += (i ? "," : "") + JsonNumber(values.samples_ms[i]);
	out += "]";
	out += ",\"errored_pct\":" + JsonNumber(values.errored_pct);
	out += ",\"build_s\":" + JsonNumber(values.build_s);
	out += ",\"reason\":" + JsonString(values.reason);
	out += ",\"finals\":" + JsonString(values.finals);
	out += ",\"package_version\":" + JsonString(values.package_version);
	out += ",\"suite_rev\":" + JsonString(values.suite_rev);
	return out + "}";
}
