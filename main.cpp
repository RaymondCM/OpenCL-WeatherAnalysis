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

	typedef int data_type;
	std::vector<data_type> data = Parse::File<data_type>(file_path);

	WeatherAnalysis<data_type> world(data);
	world.CmdParser(argc, argv);
	world.Initialise(kernels_path);
	world.Configure(1024, 0);
	world.PadData();
	world.WriteDataToDevice();
	world.Min();
	world.Max();
	world.PrintResults();

	char c;
	std::cin >> c;
	return 0;
}
