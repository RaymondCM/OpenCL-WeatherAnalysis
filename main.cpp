#include <iostream>
#include <vector>
#include <string>

#include "WeatherAnalysis.hpp"
#include "Parser.hpp"

int main(int argc, char **argv) {
	std::string root(".");

	#ifdef PROJECT_ROOT
		root = PROJECT_ROOT;
	#endif

	std::string file_path = root + "/data/temp_lincolnshire.txt";
	std::string kernels_path = root + "/opencl/kernels.cl";

	typedef float T;
	std::vector<T> data = Parse::File<T>(file_path);
    std::cout << "Size: " << data.size() << ", Last: " << data.back() << '\n' << std::endl;

    WeatherAnalysis<T> world(data);
	world.CmdParser(argc, argv);
	world.Initialise(kernels_path);

    world.Configure(32, 0);
    world.SetVerboseKernel(true);
    world.UsePreferredKernelOptions(false);
    world.PrintKernelProfilingData(true);

	world.PadData();
    world.PrintBaselineResults();
    world.WriteDataToDevice();

    world.Min();
    world.Max();
    world.Average();
    world.StdDeviation();

    world.PrintResults();

	char c;
	std::cin >> c;

	return 0;
}
