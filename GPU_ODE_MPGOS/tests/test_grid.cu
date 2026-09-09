// grid.cuh against a numpy reference grid: test_grid <file.npy> <scale> <grid_min> <grid_max>.
// Exit 0 when every float matches bit for bit; prints the first mismatches and exits 1 otherwise.
// Built by runner_scripts/tests/test_grid.py with the Bench.cu host flags (-O3 -std=c++17).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../grid.cuh"

// The floats of a one-dimensional little-endian '<f4' C-order .npy file.
static std::vector<float> ReadNpy(const std::string& path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
		throw std::runtime_error("cannot open " + path);
	std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (bytes.size() < 12 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
		throw std::runtime_error(path + " is not an npy file");
	unsigned char major = (unsigned char)bytes[6];
	size_t header_len, offset;
	if (major == 1)
	{
		header_len = (unsigned char)bytes[8] | ((unsigned char)bytes[9] << 8);
		offset = 10;
	}
	else
	{
		header_len = (uint32_t)(unsigned char)bytes[8] | ((uint32_t)(unsigned char)bytes[9] << 8)
			| ((uint32_t)(unsigned char)bytes[10] << 16) | ((uint32_t)(unsigned char)bytes[11] << 24);
		offset = 12;
	}
	std::string header(bytes.begin() + offset, bytes.begin() + offset + header_len);
	if (header.find("'<f4'") == std::string::npos)
		throw std::runtime_error(path + " is not '<f4': " + header);
	if (header.find("'fortran_order': False") == std::string::npos)
		throw std::runtime_error(path + " is not C order");
	size_t start = offset + header_len;
	if ((bytes.size() - start) % 4 != 0)
		throw std::runtime_error(path + " has a partial float");
	std::vector<float> values((bytes.size() - start) / 4);
	std::memcpy(values.data(), bytes.data() + start, bytes.size() - start);
	return values;
}

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		std::cerr << "usage: test_grid <file.npy> <linear|log> <grid_min> <grid_max>" << std::endl;
		return 2;
	}
	try
	{
		std::vector<float> reference = ReadNpy(argv[1]);
		std::string scale = argv[2];
		double grid_min = std::strtod(argv[3], nullptr);
		double grid_max = std::strtod(argv[4], nullptr);
		int n = (int)reference.size();
		std::vector<float> ours = GridValues(scale, grid_min, grid_max, n);
		std::vector<double> wide = Grid<double>(scale, grid_min, grid_max, n);
		size_t mismatches = 0;
		for (int i = 0; i < n; i++)
		{
			uint32_t a, b;
			std::memcpy(&a, &ours[i], 4);
			std::memcpy(&b, &reference[i], 4);
			if (a != b || (double)ours[i] != wide[i])
			{
				if (mismatches < 5)
					std::printf("mismatch at %d: ours %.9g reference %.9g\n", i, ours[i], reference[i]);
				mismatches++;
			}
		}
		if (mismatches != 0)
		{
			std::printf("%zu of %d points differ\n", mismatches, n);
			return 1;
		}
		// The ne grid: 1024 points up to the double v[1023] reproduce the first 1024 reference points.
		if (n >= 1024)
		{
			double point = GridPoint(scale, grid_min, grid_max, n, 1023);
			std::vector<float> prefix = GridValues(scale, grid_min, point, 1024);
			if ((float)point != reference[1023])
			{
				std::printf("v[1023] %.9g is not the reference %.9g\n", (float)point, reference[1023]);
				return 1;
			}
			for (int i = 0; i < 1024; i++)
			{
				uint32_t a, b;
				std::memcpy(&a, &prefix[i], 4);
				std::memcpy(&b, &reference[i], 4);
				if (a != b)
				{
					std::printf("prefix mismatch at %d: %.9g reference %.9g\n", i, prefix[i], reference[i]);
					return 1;
				}
			}
		}
		std::printf("ok %s %d points\n", argv[1], n);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 2;
	}
}
