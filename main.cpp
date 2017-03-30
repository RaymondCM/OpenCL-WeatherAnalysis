//Raymond Kirk - 14474219@students.lincoln.ac.uk

#include <iostream>
#include <vector>
#include <string>
#include "SimpleTimer.hpp"

#include "WeatherAnalysis.hpp"
#include "Parser.hpp"

int main(int argc, char **argv) {
	//Enable a timer to measure overall host code execution time
    SimpleTimer t;
    t.Tic();

	//Configure path to be relative to where the root is initially
    std::string root(".");

	#ifdef PROJECT_ROOT
		root = PROJECT_ROOT;
	#endif

    std::string file_path = root + "/data/temp_lincolnshire_short.txt";
    std::string kernels_path = root + "/opencl/kernels.cl";

	//Parse the data file and set the typedef for the entire enviroment
    typedef int T;
    std::vector<T> data = Parse::File<T>(file_path);
    std::cout << "Size: " << data.size() << ", Last: " << data.back() << '\n' << std::endl;

	//Initialise the Analysis world variable with cmd args and path
    WeatherAnalysis<T> world(data);
    world.CmdParser(argc, argv);
    world.Initialise(kernels_path);

	//Configure the world to use a size of 512
    world.Configure(512, 0);

	//Optionally configure flags to determine kernel execution and verbose printing
    world.SetVerboseKernel(false);
    world.UsePreferredKernelOptions(false);
    world.PrintKernelProfilingData(false);
    world.SetKernelWorkGroupRecursion(false);

	//Mandatory functions to call initially
	world.PadData();
	world.WriteDataToDevice();

	//Print comparable statistics
    world.PrintBaselineResults();

	//Execute statistic kernels
    world.Min();
    world.Max();
    world.Sum();
    world.StdDeviation();
    world.Sort();

	//Print results and execution time
    world.PrintResults();
    std::cout << "Program terminated in: " << std::fixed << t.Toc() / 1000000 << "ms" << std::endl;

	//Wait for input before termination
    char c;
    std::cin >> c;
    return 0;
}
