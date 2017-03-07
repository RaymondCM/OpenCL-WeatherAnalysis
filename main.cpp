#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.hpp"

template<typename T>
bool ParseFile(std::string file_path, std::vector<T>& dest) {
	//Open input stream to file
	std::ifstream input_file(file_path);

	//Seek to the end of the file and get the position of the last char
	input_file.seekg(0, std::ios_base::end);
	std::size_t size = input_file.tellg();

	//Seek back to the start of the file
	input_file.seekg(0, std::ios_base::beg);

	//Read into char buffer with size of file
	char * file_contents = new char[size];
	input_file.read(&file_contents[0], size);

	//Close the file
	input_file.close();

	//Parse the file by keeping track of the last space before \n
	long space_pos = 0;

	//Final vector of values

	for (long i = 0; i < size; ++i) {
		char c = file_contents[i];

		if (c == ' ') {
			//+1 to i to itterate after last space
			space_pos = i + 1;
		}
		else if (c == '\n') {
			int len = i - space_pos, index = 0;

			//Allocate buffer for word between space_pos and \n
			char * word = new char[len];
			word[len] = '\0';

			//Get every char between last space and \n
			for (int j = space_pos; j < i; j++) {
				word[index++] = file_contents[j];
			}

			//Parse word to double (higher precison)
			dest.push_back(strtod(word, NULL));
		}
	};

	return true;
};

int main(int argc, char **argv) {
	//Get user options from command line
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	std::string root(".");

	#ifdef PROJECT_ROOT
		root = PROJECT_ROOT;
	#endif

	std::string file_path = root + "/data/temp_lincolnshire_short.txt";
	std::string kernels_path = root + "/opencl/kernels.cl";

	typedef int temptype;
	std::vector<temptype> temperature;
	ParseFile(file_path, temperature);

	try {
		//Select both platform and device from user options 
		cl::Context context = GetContext(platform_id, device_id);

		//Print device and platform
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//Create a queue for kernels (commands)
		cl::CommandQueue queue(context);

		//Build the kernel code and detect any issues
		cl::Program::Sources sources;

		//Read file in and add to sources as pair (string*, length)
		AddSources(sources, kernels_path);
		cl::Program program(context, sources);

		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Create local and global size variables.
		//Pad variables so that the input length is divisable by the workgroup size
		//Use 9999 for neutral values as this is a tempreature value not present in the dataset
		unsigned int local_size = 256;
		cl::NDRange local_range(local_size);

		unsigned int element_count = temperature.size();
		unsigned int padding = element_count % local_size;
		int neutral_value = 0;

		if (padding > 0) {
			std::vector<temptype> neutral_values(local_size - padding, neutral_value);
			temperature.insert(temperature.end(), neutral_values.begin(), neutral_values.end());
		}

		//Recalculate number of elements 
		element_count = temperature.size();

		unsigned int elements_size = element_count * sizeof(temptype);
		cl::NDRange global_range(element_count);
		unsigned int group_count = element_count / local_size;

		//Output variables
		int min, max, average, standard_deviation;

		//Allocate device buffers
		cl::Buffer temperature_buffer(context, CL_MEM_READ_ONLY, elements_size);
		cl::Buffer output_buffer(context, CL_MEM_READ_ONLY, elements_size);

		//Copy temperature_buffer data to device
		queue.enqueueWriteBuffer(temperature_buffer, CL_TRUE, 0, elements_size, &temperature[0]);

		//Copy output buffer to device with all zeros
		queue.enqueueFillBuffer(output_buffer, 0, 0, elements_size);

		//Configure kernels and queue them for execution
		cl::Kernel kernel_1 = cl::Kernel(program, "MinimumAndMax");
		kernel_1.setArg(0, temperature_buffer);
		kernel_1.setArg(1, output_buffer);
		//Allocate local memory with number of local elements * size
		kernel_1.setArg(2, cl::Local(local_size * sizeof(temptype)));

		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; //get device
		size_t preffered = kernel_1.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
		size_t maxW = kernel_1.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

		std::cout << "Preffered Work Group Size: " << preffered << std::endl;
		std::cout << "Max Work Group Size: " << maxW << std::endl;

		//call all kernels in a sequence
		std::cout << "Global: " << element_count << ", Local: " << local_size << std::endl;

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, global_range, local_range);

		//Create vector to read final values to
		std::vector<temptype> output(element_count, 0);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(temptype), &output[0]);

		std::cout << "Min: " << output.at(0) << ", Max: " << output.at(1) << std::endl;

		//std::cout << "Kernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		//std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		return 0;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
