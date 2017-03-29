#include <iostream>
#include <cmath>

#include "WeatherAnalysis.hpp"
#include "Utils.hpp"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"

template<class T>
WeatherAnalysis<T>::WeatherAnalysis(std::vector<T> &t_data) {
    this->TypeCheck();
    this->data = t_data;
    this->local_range = cl::NDRange(this->local_size);
    this->global_range = cl::NDRange(t_data.size());
};

template<class T>
void WeatherAnalysis<T>::CmdParser(int &argc, char **&argv) {
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
        std::cout << "Running on " << GetPlatformName(this->platform_ID) << ", "
                  << GetDeviceName(this->platform_ID, this->device_ID) << std::endl;

        //Create a queue for kernels (commands)
        this->queue = cl::CommandQueue(this->context, CL_QUEUE_PROFILING_ENABLE);

        //Read file in and add to sources as pair (string*, length)
        AddSources(this->sources, cl_path);
        this->program = cl::Program(this->context, this->sources);
        this->Build();
    }
    catch (const cl::Error &e) {
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
    catch (const cl::Error &e) {
        PrintBuildErrors(this->context, this->program);
        throw std::exception();
    }
};

template<class T>
void WeatherAnalysis<T>::Configure(int local_size, T neutral_value) {
    this->local_size = local_size;
    this->local_range = cl::NDRange(local_size);
    this->neutral_value = neutral_value;
};

template<class T>
void WeatherAnalysis<T>::PadData(T neutral_value, bool print) {
    unsigned int pad_count = this->data.size() % this->local_size;
    this->neutral_value = neutral_value;

    if (pad_count > 0) {
        this->pad_right = pad_count;
        std::vector<T> neutrals(this->local_size - this->pad_right, this->neutral_value);
        this->data.insert(this->data.end(), neutrals.begin(), neutrals.end());
        this->global_range = cl::NDRange(this->data.size());
        if (print)
            std::cout << "Padded by " << this->pad_right << " elements of " << this->neutral_value << '\n' << std::endl;
    } else {
        if (print)
            std::cout << "Data already a factor of local size\n" << std::endl;
    }
};

template<class T>
void WeatherAnalysis<T>::PrintResults() {
    std::stringstream results;
    results.precision(5);
    results << "OpenCL Weather Analysis:" << "\n\t";
    results << std::fixed << "Min: " << this->minimum << "\n\t";
    results << std::fixed << "Max: " << this->maximum << "\n\t";
    results << std::fixed << "Sum: " << this->sum << "\n\t";
    results << std::fixed << "Average: " << this->average << "\n\t";
    results << std::fixed << "Std Deviation: " << this->std_deviation << std::endl;
    std::cout << results.str();
};

template<class T>
void WeatherAnalysis<T>::PrintProfilingData(const std::string &kernel_ID) {
    std::cout << "Executed kernel '" << kernel_ID << "':\n\t";
    std::cout << "Elapsed time on host: " << this->timer.Toc() << "ns\n\t";
    this->queue.finish();
    ProfilingInfo(this->prof_event);
};

template<class T>
void WeatherAnalysis<T>::PrintKernelProfilingData(bool print_data) {
    this->print_profiling_data = print_data;
};

template<class T>
void WeatherAnalysis<T>::PrintQueueOptions(const cl::Kernel &k) {
    PrintPreferredWorkGroupSize(this->context, k, this->data.size(), this->local_size);
    this->SetVerboseKernel(false);
};

template<class T>
void WeatherAnalysis<T>::UsePreferredKernelOptions(bool use_pref) {
    this->use_preferred = use_pref;
};

template<class T>
void WeatherAnalysis<T>::SetVerboseKernel(bool verbose) {
    this->verbose = verbose;
};

template<class T>
void WeatherAnalysis<T>::SetKernelWorkGroupRecursion(bool should_recurse) {
    if (this->type == "FLOAT") {
        this->kernel_work_group_recursion = should_recurse;
    } else {
        std::cout << "No supported kernels for workgroup recursion for type " << this->type << std::endl;
    }
};

template<class T>
void WeatherAnalysis<T>::PrintBaselineResults() {
    T smin = this->data[0], smax = 0, ssum = 0;
    float savg = 0, sstd = 0;

    for (auto val : this->data) {
        if (val < smin)
            smin = val;
        else if (val > smax)
            smax = val;
        ssum += val;
    }

    savg = ssum / (float) (this->data.size() - this->pad_right);

    for (auto val: this->data) {
        sstd += (val - savg) * (val - savg);
    }

    sstd = sqrt(sstd / (float) (this->data.size() - this->pad_right));

    std::stringstream results;
    results.precision(5);
    results << std::fixed << "OpenCL Weather Analysis - Baseline Results:" << "\n\t";
    results << std::fixed << "Min: " << smin << "\n\t";
    results << std::fixed << "Max: " << smax << "\n\t";
    results << std::fixed << "Sum: " << ssum << "\n\t";
    results << std::fixed << "Average: " << savg << "\n\t";
    results << std::fixed << "Std Deviation: " << sstd << '\n' << std::endl;
    std::cout << results.str();
};

template<class T>
void WeatherAnalysis<T>::WriteDataToDevice() {
    unsigned int data_size = this->data.size() * sizeof(T);
    unsigned int work_group_size = (this->data.size() / this->local_size) * sizeof(T);

    //Allocate device buffers
    this->data_buffer = cl::Buffer(this->context, CL_MEM_READ_ONLY, data_size);
    //Copy data_buffer data to device
    this->queue.enqueueWriteBuffer(this->data_buffer, CL_TRUE, 0, data_size, &this->data[0]);

    //Allocate device buffers
    this->min_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, work_group_size);
    this->max_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, work_group_size);
    this->sum_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, work_group_size);
    this->std_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, work_group_size);
    this->sort_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, data_size);

    //Copy output buffer to device with all zeros
    this->queue.enqueueFillBuffer(this->min_buffer, 0, 0, work_group_size);
    this->queue.enqueueFillBuffer(this->max_buffer, 0, 0, work_group_size);
    this->queue.enqueueFillBuffer(this->sum_buffer, 0, 0, work_group_size);
    this->queue.enqueueFillBuffer(this->std_buffer, 0, 0, work_group_size);
    this->queue.enqueueFillBuffer(this->sort_buffer, 0, 0, data_size);
};

template<class T>
void WeatherAnalysis<T>::EnqueueKernel(cl::Kernel &k, const std::string &ID) {
    this->EnqueueNDRangeKernel(k, ID);
}

template<class T>
void WeatherAnalysis<T>::EnqueueKernel(cl::Kernel &k, const std::string &ID, cl::Buffer &out_buffer, int in_index) {
    if (this->kernel_work_group_recursion) {
        std::cout << "Calling '" << ID << "' recursively until work group output reduced to one element\n" << std::endl;
        int work_group_count = this->data.size() / this->local_size;
        bool swapped = false;
        while (work_group_count > this->local_size) {
            this->EnqueueNDRangeKernel(k, ID);

            if (swapped) {
                work_group_count /= this->local_size;
            } else {
                k.setArg(in_index, out_buffer);
                swapped = true;
            }
        }
    }

    this->EnqueueNDRangeKernel(k, ID);
}

template<class T>
void WeatherAnalysis<T>::EnqueueNDRangeKernel(cl::Kernel &k, const std::string &kernel_ID) {
    if (this->use_preferred) {
        this->Configure(GetPreferredWorkGroupSize(this->context, k), this->neutral_value);
        this->PadData(this->neutral_value, false);
    }

    if (this->verbose)
        this->PrintQueueOptions(k);

    this->timer.Tic();
    //Queue and execute kernel
    this->queue.enqueueNDRangeKernel(k, cl::NullRange, this->global_range, this->local_range, NULL, &this->prof_event);

    if (this->print_profiling_data)
        this->PrintProfilingData(kernel_ID);
};

template<class T>
std::string WeatherAnalysis<T>::GetKernelName(std::string s, bool can_reduce) {
    return s + (can_reduce && this->kernel_work_group_recursion ? "_WG_REDUCE_" : "_") + this->type;
}

template<class T>
void WeatherAnalysis<T>::Min() {
    std::string kernel_ID(this->GetKernelName("min"));

    //Configure kernels and queue them for execution
    //Allocate local memory with number of local elements * size
    cl::Kernel min_kernel = cl::Kernel(program, kernel_ID.c_str());
    min_kernel.setArg(0, this->data_buffer);
    min_kernel.setArg(1, this->min_buffer);
    min_kernel.setArg(2, cl::Local(this->local_size * sizeof(T)));

    this->EnqueueKernel(min_kernel, kernel_ID, this->min_buffer);

    //Create vector to read final values to
    std::vector<T> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->min_buffer, CL_TRUE, 0, sizeof(T), &output[0]);

    this->minimum = output.at(0);
};

template<class T>
void WeatherAnalysis<T>::Max() {
    std::string kernel_ID(this->GetKernelName("max"));

    //Configure kernels and queue them for execution
    //Allocate local memory with number of local elements * size
    cl::Kernel max_kernel = cl::Kernel(program, kernel_ID.c_str());
    max_kernel.setArg(0, this->data_buffer);
    max_kernel.setArg(1, this->max_buffer);
    max_kernel.setArg(2, cl::Local(this->local_size * sizeof(T)));

    this->EnqueueKernel(max_kernel, kernel_ID, this->max_buffer);

    //Create vector to read final values to
    std::vector<T> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->max_buffer, CL_TRUE, 0, sizeof(T), &output[0]);

    this->maximum = output.at(0);
};

template<class T>
void WeatherAnalysis<T>::Average() {
    if (this->sum == 0)
        this->Sum();
    this->average = (float) this->sum / (float) (this->data.size() - this->pad_right);
};

template<class T>
void WeatherAnalysis<T>::Sum() {
    std::string kernel_ID(this->GetKernelName("sum"));

    //Configure kernels and queue them for execution
    cl::Kernel sum_kernel = cl::Kernel(program, kernel_ID.c_str());
    sum_kernel.setArg(0, this->data_buffer);
    sum_kernel.setArg(1, this->sum_buffer);

    //Allocate local memory with number of local elements * size
    sum_kernel.setArg(2, cl::Local(this->local_size * sizeof(T)));

    this->EnqueueKernel(sum_kernel, kernel_ID, this->sum_buffer);

    //Create vector to read final values to
    std::vector<T> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->sum_buffer, CL_TRUE, 0, sizeof(T), &output[0]);

    this->sum = output.at(0);
};

template<class T>
void WeatherAnalysis<T>::StdDeviation() {
    std::string kernel_ID(this->GetKernelName("std", false));

    //Configure kernels and queue them for execution
    cl::Kernel std_kernel = cl::Kernel(this->program, kernel_ID.c_str());
    std_kernel.setArg(0, this->data_buffer);
    std_kernel.setArg(1, this->std_buffer);
    std_kernel.setArg(2, this->average);

    //Allocate local memory with number of local elements * size
    std_kernel.setArg(3, cl::Local(this->local_size * sizeof(T)));

    this->EnqueueKernel(std_kernel, kernel_ID);

    //Create vector to read final values
    std::vector<T> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->std_buffer, CL_TRUE, 0, sizeof(T), &output[0]);
    this->std_deviation = this->type == "INT" ? sqrt((float)output.at(0) / (float)(this->data.size() - this->pad_right)) : output.at(0);
};

template<class T>
void WeatherAnalysis<T>::Sort() {
    std::string kernel_ID(this->GetKernelName("sort"), false);

    //Configure kernels and queue them for execution
    cl::Kernel sort_kernel = cl::Kernel(this->program, kernel_ID.c_str());
    sort_kernel.setArg(0, this->data_buffer);
    sort_kernel.setArg(1, this->sort_buffer);

    //Allocate local memory with number of local elements * size
    sort_kernel.setArg(2, cl::Local(this->local_size * sizeof(T)));

    this->EnqueueKernel(sort_kernel, kernel_ID);

    //Create vector to read final values
    std::vector<T> output(this->data.size(), 0);

    //5.3 Copy the result from device to host
    this->queue.enqueueReadBuffer(this->sort_buffer, CL_TRUE, 0, this->data.size() * sizeof(T), &output[0]);

    this->sorted_data = output;
    this->data = this->sorted_data;
    PrintNonZeros(this->sorted_data);
    std::cout << this->sorted_data.at(0);
};

template<class T>
void WeatherAnalysis<T>::TypeCheck() {
    this->type = "UNKNOWN";
    throw std::runtime_error("ERROR: WeatherAnalysis instantiated with incorrect type.");
}

template<>
void WeatherAnalysis<float>::TypeCheck() {
    this->type = "FLOAT";
};

template<>
void WeatherAnalysis<int>::TypeCheck() {
    this->type = "INT";
};

template
class WeatherAnalysis<float>;

template
class WeatherAnalysis<int>;

#pragma clang diagnostic pop