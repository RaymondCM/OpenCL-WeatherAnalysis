#ifdef _WEATHERANALYSIS_H_

template<class T>
WeatherAnalysis<T>::WeatherAnalysis(std::vector<T>& t_data) {
	this->data = t_data;
	this->local_range = cl::NDRange(this->local_size);
	this->global_range = cl::NDRange(t_data.size());
};

template<class T>
void WeatherAnalysis<T>::CmdParser(int& argc, char**&argv) {
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { this->platform_ID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { this->device_ID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
};

template<class T>
void WeatherAnalysis<T>::Initialise(const std::string cl_path) {
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

template<class T>
void WeatherAnalysis<T>::Build() {
	try {
		program.build();
	}
	catch (const cl::Error& e) {
		std::cerr << "Build Status: " << this->program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(this->context.getInfo<CL_CONTEXT_DEVICES>()[0]) << '\n';
		std::cerr << "Build Options:\t" << this->program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(this->context.getInfo<CL_CONTEXT_DEVICES>()[0]) << '\n';
		std::cerr << "Build Log:\t " << this->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		throw std::exception();
	}
};

template<class T>
void WeatherAnalysis<T>::Configure(int local_size, int neutral_value) {
	this->local_size = local_size;
	this->local_range = cl::NDRange(local_size);
	this->neutral_value = neutral_value;
};

template<class T>
void WeatherAnalysis<T>::PadData(int neutral_value) {
	unsigned int pad_right = this->data.size() % this->local_size;
	if (pad_right > 0) {
		std::vector<T> neutrals(this->local_size - pad_right, this->neutral_value);
		this->data.insert(this->data.end(), neutrals.begin(), neutrals.end());
		this->global_range = cl::NDRange(this->data.size());
		std::cout << "Padded data by " << pad_right << " elements of " << this->neutral_value << std::endl;
	}
	else {
		std::cout << "Data already padded" << std::endl;
	}
};

template<class T>
void WeatherAnalysis<T>::WriteDataToDevice() {
	unsigned int data_size = this->data.size() * sizeof(T);

	//Allocate device buffers
	this->data_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, data_size);
	this->min_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, data_size);
	this->max_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, data_size);

	//Copy data_buffer data to device
	this->queue.enqueueWriteBuffer(this->data_buffer, CL_TRUE, 0, data_size, &this->data[0]);

	//Copy output buffer to device with all zeros
	this->queue.enqueueFillBuffer(this->min_buffer, 0, 0, data_size);
	this->queue.enqueueFillBuffer(this->max_buffer, 0, 0, data_size);
};

template<class T>
void WeatherAnalysis<T>::EnqueueKernel(cl::Kernel k) {
	//Queue and execute kernel
	this->queue.enqueueNDRangeKernel(k, cl::NullRange, this->global_range, this->local_range);
};

template<class T>
void WeatherAnalysis<T>::PrintResults() {
	std::cout << "OpenCL Weather Analysis:" << "\n\t";
	std::cout << "Min: " << this->minimum << "\n\t";
	std::cout << "Max: " << this->maximum << "\n\t";
	std::cout << "Average: " << this->average << "\n\t";
	std::cout << "Std Deviation: " << this->std_deviation << std::endl;
};

template<class T>
void WeatherAnalysis<T>::PrintQueueOptions(const cl::Kernel& k) {
	cl::device device = this->context.getinfo<cl_context_devices>()[0]; 
	size_t preffered = k.getworkgroupinfo<cl_kernel_preferred_work_group_size_multiple>(device);
	size_t maxw = k.getworkgroupinfo<cl_kernel_work_group_size>(device);

	std::cout << "Preffered Kernel Options: " << "\n\t";
	std::cout << "Preffered Work Group Size: " << preffered << "\n\t";
	std::cout << "Max Work Group Size: " << maxw << "\n\t";
	std::cout << "Global Size: " << this->data.size() << "\n\t";
	std::cout << "Local Size: " << this->local_size << std::endl;
};

template<class T>
void WeatherAnalysis<T>::Min() {
	//Configure kernels and queue them for execution
	cl::Kernel min_kernel = cl::Kernel(program, "minimum");
	min_kernel.setArg(0, this->data_buffer);
	min_kernel.setArg(1, this->min_buffer);

	//Allocate local memory with number of local elements * size
	min_kernel.setArg(2, cl::Local(this->local_size * sizeof(T)));

	this->EnqueueKernel(min_kernel);

	//Create vector to read final values to
	std::vector<T> output(this->data.size(), 0);

	//5.3 Copy the result from device to host
	this->queue.enqueueReadBuffer(this->min_buffer, CL_TRUE, 0, sizeof(T), &output[0]);

	this->minimum = output.at(0);
};

template<class T>
void WeatherAnalysis<T>::Max() {
	//Configure kernels and queue them for execution
	cl::Kernel max_kernel = cl::Kernel(program, "maximum");
	max_kernel.setArg(0, this->data_buffer);
	max_kernel.setArg(1, this->max_buffer);

	//Allocate local memory with number of local elements * size
	max_kernel.setArg(2, cl::Local(this->local_size * sizeof(T)));

	this->EnqueueKernel(max_kernel);

	//Create vector to read final values to
	std::vector<T> output(this->data.size(), 0);

	//5.3 Copy the result from device to host
	this->queue.enqueueReadBuffer(this->max_buffer, CL_TRUE, 0, sizeof(T), &output[0]);

	this->maximum = output.at(0);
};

#endif