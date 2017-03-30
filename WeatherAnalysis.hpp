#ifndef _WEATHERANALYSIS_H_
#define _WEATHERANALYSIS_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <string>
#include "SimpleTimer.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

template<class T>
class WeatherAnalysis {
public:
    WeatherAnalysis(std::vector<T> &);
	void CmdParser(int&, char**&);
	void Initialise(const std::string);
	void Build();
	void Configure(int = 1024, T = 0);
	void PadData(T = 0, bool = true);
	void WriteDataToDevice();
	void PrintResults();
	void PrintQueueOptions(const cl::Kernel&);
	void SetVerboseKernel(bool = true);
    void UsePreferredKernelOptions(bool = true);
    void PrintKernelProfilingData(bool = true);
	void SetKernelWorkGroupRecursion(bool = true);
    void PrintBaselineResults();
	void Min(); 
	void Max();
    void Sum();
    void Average();
    void StdDeviation();
    void Sort();
private:
	int platform_ID = 0, device_ID = 0;
	int local_size = 1024;
	T neutral_value = 0, minimum = 0, maximum = 0, sum = 0, median = 0, first_quantile = 0, third_quantile = 0;
	float average = 0, std_deviation = 0;
    unsigned int pad_right = 0;
	bool verbose = false, use_preferred = false, print_profiling_data = false, kernel_work_group_recursion = false;
	cl::NDRange local_range, global_range;
    cl::Event prof_event;
    std::vector<T> data, sorted_data;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::Program::Sources sources;
    cl::Buffer data_buffer, min_buffer, max_buffer, sum_buffer, std_buffer, sort_buffer;
	std::string type = "";
    SimpleTimer timer;

	void TypeCheck();
    void PrintProfilingData(const std::string &kernel_ID);
	void EnqueueKernel(cl::Kernel &, const std::string &, cl::Buffer &, int = 0);
	void EnqueueKernel(cl::Kernel &k, const std::string &ID);
	void EnqueueNDRangeKernel(cl::Kernel &k, const std::string &kernel_ID);
	std::string GetKernelName(std::string s, bool can_reduce = true);
};

#endif