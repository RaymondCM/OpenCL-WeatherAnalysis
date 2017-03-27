#ifndef _WEATHERANALYSIS_H_
#define _WEATHERANALYSIS_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

class WeatherAnalysis {
public:
    WeatherAnalysis(std::vector<int> &);
	void CmdParser(int&, char**&);
	void Initialise(const std::string);
	void Build();
	void Configure(int = 1024, int = 0);
	void PadData(int = 0);
	void WriteDataToDevice();
	void EnqueueKernel(cl::Kernel);
	void PrintResults();
	void PrintQueueOptions(const cl::Kernel&);
	void Min(); 
	void Max();

    void Sum();

    void Average();

    void StdDeviation();
private:
	int platform_ID = 0, device_ID = 0;
	int local_size = 1024, neutral_value = 0;
    unsigned int pad_right = 0;
    int minimum = 0, maximum = 0, sum = 0;
    float average = 0, std_deviation = 0;
	cl::NDRange local_range, global_range;
    std::vector<int> data;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::Program::Sources sources;
    cl::Buffer data_buffer, min_buffer, max_buffer, sum_buffer, std_buffer;
};

#endif