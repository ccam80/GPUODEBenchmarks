// The ensemble grid of section 1.2, reproducing runner_scripts/grid.py bit for bit.
// Host code: every point is built in double and cast to float; double runs widen the float values back.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// v[0..n-1]: linear or log in double with the last point pinned to grid_max, as float.
inline std::vector<float> GridValues(const std::string& scale, double grid_min, double grid_max, int n)
{
	if (n < 2)
		throw std::invalid_argument("a grid needs n >= 2");
	if (!std::isfinite(grid_min) || !std::isfinite(grid_max))
		throw std::invalid_argument("grid_min and grid_max must be finite");
	std::vector<double> values(n);
	if (scale == "linear")
	{
		double step = (grid_max - grid_min) / (n - 1);
		for (int i = 0; i < n; i++)
			values[i] = grid_min + i * step;
	}
	else if (scale == "log")
	{
		if (grid_min <= 0.0 || grid_max <= 0.0)
			throw std::invalid_argument("a log grid needs grid_min > 0 and grid_max > 0");
		double a = log10(grid_min);
		double b = log10(grid_max);
		double step = (b - a) / (n - 1);
		for (int i = 0; i < n; i++)
			values[i] = pow(10.0, a + i * step);
	}
	else
		throw std::invalid_argument("grid_scale '" + scale + "' is not linear or log");
	values[n - 1] = grid_max;
	std::vector<float> out(n);
	for (int i = 0; i < n; i++)
		out[i] = (float)values[i];
	return out;
}

// v[index] in double, before the cast: the grid_max of a shorter grid that reproduces v[0..index] bit for bit.
inline double GridPoint(const std::string& scale, double grid_min, double grid_max, int n, int index)
{
	if (n < 2)
		throw std::invalid_argument("a grid needs n >= 2");
	if (index < 0 || index >= n)
		throw std::invalid_argument("index outside the grid");
	if (index == n - 1)
		return grid_max;
	if (scale == "linear")
		return grid_min + index * ((grid_max - grid_min) / (n - 1));
	if (scale != "log")
		throw std::invalid_argument("grid_scale '" + scale + "' is not linear or log");
	if (grid_min <= 0.0 || grid_max <= 0.0)
		throw std::invalid_argument("a log grid needs grid_min > 0 and grid_max > 0");
	double a = log10(grid_min);
	double b = log10(grid_max);
	return pow(10.0, a + index * ((b - a) / (n - 1)));
}

// The grid in the run precision T (float or double).
template <typename T>
std::vector<T> Grid(const std::string& scale, double grid_min, double grid_max, int n)
{
	std::vector<float> values = GridValues(scale, grid_min, grid_max, n);
	return std::vector<T>(values.begin(), values.end());
}
