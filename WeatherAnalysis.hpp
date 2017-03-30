//Raymond Kirk - 14474219@students.lincoln.ac.uk

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

//OpenCL Parallel Weather Analysis Class
// User Friendly Analysis class for any int/float vectors. Fully templated to support multiple types
// and provides greater abstraction from low-level OpenCL features.
template<class T>
class WeatherAnalysis {
public:
    WeatherAnalysis(std::vector<T> &);
	//Configures options from command-line arguments such as device and platform.
	void CmdParser(int&, char**&);
	//Initialises context and queue.
	void Initialise(const std::string);
	//Builds the program and kernels from the opencl/kernels.cl.
	void Build();
	//Allows the user to customise frequently changed values such as Local Size and Neutral Pad Value.
	void Configure(int = 1024, T = 0);
	//Pads the data so that local size is a factor of the data size. Automatically re-configures appropriate options.
	void PadData(T = 0, bool = true);
	//Writes all of the buffers to the device at once for use throughout the class. Self-manages buffer sizes.
	void WriteDataToDevice();
	//Print class used to check current model of statistics.
	void PrintResults();
	//Function print kernel specific options such as preffered queue size.
	void PrintQueueOptions(const cl::Kernel&);
	//Sets a flag determining if PrintQueueOptions is called (Default: false)
	void SetVerboseKernel(bool = true);
	//Sets a flag determining if the kernels should automatically configure the local and global size.
    void UsePreferredKernelOptions(bool = true);
	//Sets a flag determining if the kernels should track and print profiling data such as device/host execution time.
    void PrintKernelProfilingData(bool = true);
	//Determines if the kernels should reduce workgroup or reduce on the kernel.
	void SetKernelWorkGroupRecursion(bool = true);
	//Prints an estimation of results that should be simular to parallel ones.
    void PrintBaselineResults();
	//Kernel Functions
	void Min(); 
	void Max();
    void Sum();
    void Average();
    void StdDeviation();
    void Sort();
private:
	//Context parameters
	int platform_ID = 0, device_ID = 0;
	int local_size = 1024;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::Program::Sources sources;
	cl::Buffer data_buffer, min_buffer, max_buffer, sum_buffer, std_buffer, sort_buffer;
	cl::NDRange local_range, global_range;
	cl::Event prof_event;

	//Statistic values
	T neutral_value = 0, minimum = 0, maximum = 0, sum = 0, median = 0, first_quantile = 0, third_quantile = 0;
	float average = 0, std_deviation = 0;

	//Class Flags
	bool verbose = false, use_preferred = false, print_profiling_data = false, kernel_work_group_recursion = false;

	//Data
    std::vector<T> data, sorted_data;
	unsigned int pad_right = 0;

	//Utility
	std::string type = "";
    SimpleTimer timer;

	void TypeCheck();
    void PrintProfilingData(const std::string &kernel_ID);
	//Wrapper to enqueue kernels using the correct implementation from kernels.cl, manages automatic configuration of properties
	void EnqueueKernel(cl::Kernel &, const std::string &, cl::Buffer &, int = 0);
	void EnqueueKernel(cl::Kernel &k, const std::string &ID);
	void EnqueueNDRangeKernel(cl::Kernel &k, const std::string &kernel_ID);
	//Utility function for kernel naming
	std::string GetKernelName(std::string s, bool can_reduce = true);
};

#endif