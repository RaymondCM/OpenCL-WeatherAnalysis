#include <iostream>
#include <cmath>

#include "WeatherAnalysis.hpp"
#include "Utils.hpp"

WeatherAnalysis::WeatherAnalysis(std::vector<int> &t_data) {
	this->data = t_data;
	this->local_range = cl::NDRange(this->local_size);
	this->global_range = cl::NDRange(t_data.size());
};

void WeatherAnalysis::CmdParser(int &argc, char **&argv) {
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { this->platform_ID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { this->device_ID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
};

void WeatherAnalysis::Initialise(const std::string cl_path) {
	try {
		//Select both platform and device from user options 
		this->context = GetContext(this->platform_ID, this->device_ID);

		//Print device and platform
		std::cout << "Runinng on " << GetPlatformName(this->platform_ID) << ", " << GetDeviceName(this->platform_ID, this->device_ID) << std::endl;

		//Create a queue for kernels (commands)
		this->queue = cl::CommandQueue(this->context);

		//Read file in and add to sources as pair (string*, length)
		AddSources(this->sources, cl_path);
		this->program = cl::Program(this->context, this->sources);
		this->Build();
	} 
	catch (const cl::Error& e) {
		std::cerr << "ERROR: " << e.what() << '\n';
		std::cerr << "\t" << getErrorString(e.err()) << std::endl;
		throw std::exception();
	}
};

void WeatherAnalysis::Build() {
	try {
		program.build();
	}
	catch (const cl::Error& e) {
        std::cerr << "Build Status: "
                  << this->program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(this->context.getInfo<CL_CONTEXT_DEVICES>()[0])
                  << '\n';
        std::cerr << "Build Options:\t" << this->program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(
                this->context.getInfo<CL_CONTEXT_DEVICES>()[0]) << '\n';
        std::cerr << "Build Log:\t "
                  << this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->context.getInfo<CL_CONTEXT_DEVICES>()[0])
                  << std::endl;
        throw std::exception();
	}
};

void WeatherAnalysis::Configure(int local_size, int neutral_value) {
	this->local_size = local_size;
	this->local_range = cl::NDRange(local_size);
	this->neutral_value = neutral_value;
};

void WeatherAnalysis::PadData(int neutral_value) {
    this->pad_right = this->data.size() % this->local_size;
    this->neutral_value = neutral_value;
    if (this->pad_right > 0) {
        std::vector<int> neutrals(this->local_size - this->pad_right, this->neutral_value);
		this->data.insert(this->data.end(), neutrals.begin(), neutrals.end());
		this->global_range = cl::NDRange(this->data.size());
        std::cout << "Padded data by " << this->pad_right << " elements of " << this->neutral_value << std::endl;
	}
	else {
		std::cout << "Data already padded" << std::endl;
	}
};

void WeatherAnalysis::WriteDataToDevice() {
    unsigned int data_size = this->data.size() * sizeof(int);

	//Allocate device buffers
    ///TODO: Change to CL_MEM_READ_WRITE
    this->data_buffer = cl::Buffer(this->context, CL_MEM_READ_ONLY, data_size);
    this->min_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, data_size);
    this->max_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, data_size);
    this->sum_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, data_size);
    this->std_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, data_size);

	//Copy data_buffer data to device
	this->queue.enqueueWriteBuffer(this->data_buffer, CL_TRUE, 0, data_size, &this->data[0]);

	//Copy output buffer to device with all zeros
	this->queue.enqueueFillBuffer(this->min_buffer, 0, 0, data_size);
    this->queue.enqueueFillBuffer(this->max_buffer, 0, 0, data_size);
    this->queue.enqueueFillBuffer(this->sum_buffer, 0, 0, data_size);
    this->queue.enqueueFillBuffer(this->std_buffer, 0, 0, data_size);
};

void WeatherAnalysis::EnqueueKernel(cl::Kernel k) {
	//Queue and execute kernel
	this->queue.enqueueNDRangeKernel(k, cl::NullRange, this->global_range, this->local_range);
};

void WeatherAnalysis::PrintResults() {
	std::cout << "OpenCL Weather Analysis:" << "\n\t";
	std::cout << "Min: " << this->minimum << "\n\t";
	std::cout << "Max: " << this->maximum << "\n\t";
    std::cout << "Sum: " << this->sum << "\n\t";
    std::cout << "Average: " << this->average << "\n\t";
	std::cout << "Std Deviation: " << this->std_deviation << std::endl;
};

void WeatherAnalysis::PrintQueueOptions(const cl::Kernel &k) {
//	cl::Device device = this->context.getInfo()<CL_CONTEXT_DEVICES>()[0];
//	unsigned int preffered = k.getWorkGroupInfo()<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
//	unsigned int maxw = k.getWorkGroupInfo()<CL_KERNEL_WORK_GROUP_SIZE>(device);
//
//	std::cout << "Preffered Kernel Options: " << "\n\t";
//	std::cout << "Preffered Work Group Size: " << preffered << "\n\t";
//	std::cout << "Max Work Group Size: " << maxw << "\n\t";
//	std::cout << "Global Size: " << this->data.size() << "\n\t";
//	std::cout << "Local Size: " << this->local_size << std::endl;
};

void WeatherAnalysis::Min() {
	//Configure kernels and queue them for execution
	cl::Kernel min_kernel = cl::Kernel(program, "minimum");
	min_kernel.setArg(0, this->data_buffer);
	min_kernel.setArg(1, this->min_buffer);

	//Allocate local memory with number of local elements * size
    min_kernel.setArg(2, cl::Local(this->local_size * sizeof(int)));

	this->EnqueueKernel(min_kernel);

	//Create vector to read final values to
    std::vector<int> output(this->data.size(), 0);

	//5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->min_buffer, CL_TRUE, 0, sizeof(int), &output[0]);

	this->minimum = output.at(0);
};

void WeatherAnalysis::Max() {
	//Configure kernels and queue them for execution
	cl::Kernel max_kernel = cl::Kernel(program, "maximum");
	max_kernel.setArg(0, this->data_buffer);
	max_kernel.setArg(1, this->max_buffer);

	//Allocate local memory with number of local elements * size
    max_kernel.setArg(2, cl::Local(this->local_size * sizeof(int)));

	this->EnqueueKernel(max_kernel);

	//Create vector to read final values to
    std::vector<int> output(this->data.size(), 0);

	//5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->max_buffer, CL_TRUE, 0, sizeof(int), &output[0]);

	this->maximum = output.at(0);
};

void WeatherAnalysis::Average() {
    this->Sum();
    this->average = (float) this->sum / (this->data.size() - this->pad_right);
};

void WeatherAnalysis::Sum() {
    //Configure kernels and queue them for execution
    cl::Kernel sum_kernel = cl::Kernel(program, "sum");
    sum_kernel.setArg(0, this->data_buffer);
    sum_kernel.setArg(1, this->sum_buffer);

    //Allocate local memory with number of local elements * size
    sum_kernel.setArg(2, cl::Local(this->local_size * sizeof(int)));

    this->EnqueueKernel(sum_kernel);

    //Create vector to read final values to
    std::vector<int> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->sum_buffer, CL_TRUE, 0, sizeof(int), &output[0]);

    this->sum = output.at(0);
};


void WeatherAnalysis::StdDeviation() {
    //Configure kernels and queue them for execution
    cl::Kernel std_kernel = cl::Kernel(this->program, "std");
    std_kernel.setArg(0, this->data_buffer);
    std_kernel.setArg(1, this->std_buffer);
    std_kernel.setArg(2, (int) this->average);

    //Allocate local memory with number of local elements * size
    std_kernel.setArg(3, cl::Local(this->local_size * sizeof(int)));

    this->EnqueueKernel(std_kernel);

    //Create vector to read final values to
    std::vector<int> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->std_buffer, CL_TRUE, 0, sizeof(int), &output[0]);

    this->std_deviation = sqrt((float) output.at(0) / (this->data.size() - this->pad_right));
};