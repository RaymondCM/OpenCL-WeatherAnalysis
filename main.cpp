#include <iostream>
#include <vector>
#include <string>
#include "SimpleTimer.hpp"

#include "WeatherAnalysis.hpp"
#include "Parser.hpp"

int main(int argc, char **argv) {
    SimpleTimer t;
    t.Tic();
    std::string root(".");

#ifdef PROJECT_ROOT
    root = PROJECT_ROOT;
#endif

    std::string file_path = root + "/data/temp_lincolnshire_short.txt";
    std::string kernels_path = root + "/opencl/kernels.cl";

    typedef int T;
    std::vector<T> data = Parse::File<T>(file_path);
    std::cout << "Size: " << data.size() << ", Last: " << data.back() << '\n' << std::endl;

    WeatherAnalysis<T> world(data);
    world.CmdParser(argc, argv);
    world.Initialise(kernels_path);

    world.Configure(16, 0);
    world.SetVerboseKernel(false);
    world.UsePreferredKernelOptions(false);
    world.PrintKernelProfilingData(false);
    world.SetKernelWorkGroupRecursion(false);

    world.PadData();
    world.PrintBaselineResults();
    world.WriteDataToDevice();

    world.Min();
    world.Max();
    world.Sum();
    world.Average();
    world.StdDeviation();
    world.Sort();

    world.PrintResults();


    std::cout << "Program terminated in: " << std::fixed << t.Toc() / 1000000 << "ms" << std::endl;
    char c;
    std::cin >> c;

    return 0;
}
