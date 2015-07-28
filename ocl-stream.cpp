
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include "common.h"

std::string getDeviceName(const cl::Device& device);
unsigned getDeviceList(std::vector<cl::Device>& devices);

struct badfile : public std::exception
{
    virtual const char * what () const throw ()
    {
        return "Cannot open kernel file";
    }
};


// Print error and exit
void die(std::string msg, cl::Error& e)
{
    std::cerr
            << "Error: "
            << msg
            << ": " << e.what()
            << "(" << e.err() << ")"
            << std::endl;
    exit(e.err());
}

int main(int argc, char *argv[])
{

    // Print out run information
    std::cout
        << "GPU-STREAM" << std::endl
        << "Version: " << VERSION_STRING << std::endl
        << "Implementation: OpenCL" << std::endl;

    parseArguments(argc, argv);
    try
    {
        if (NTIMES < 2) throw badntimes();
    }
    catch (std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Precision: ";
    if (useFloat) std::cout << "float";
    else std::cout << "double";
    std::cout << std::endl << std::endl;

    if (ARRAY_SIZE % 1024 != 0)
    {
        unsigned int OLD_ARRAY_SIZE = ARRAY_SIZE;
        ARRAY_SIZE -= ARRAY_SIZE % 1024;
        std::cout
            << "Warning: array size must divide 1024" << std::endl
            << "Resizing array from " << OLD_ARRAY_SIZE
            << " to " << ARRAY_SIZE << std::endl;
    }

    // Get precision (used to reset later)
    std::streamsize ss = std::cout.precision();

    size_t DATATYPE_SIZE;

    if (useFloat)
    {
        DATATYPE_SIZE = sizeof(float);
    }
    else
    {
        DATATYPE_SIZE = sizeof(double);
    }

    // Display number of bytes in array
    std::cout << std::setprecision(1) << std::fixed
        << "Array size: " << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
        << " (=" << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
        << std::endl;
    std::cout << "Total size: " << 3.0*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
        << " (=" << 3.0*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
        << std::endl;
        
    // Reset precision
    std::cout.precision(ss);

    // Open the Kernel source
    std::string kernels;
    try
    {
        std::ifstream in("ocl-stream-kernels.cl");
        if (!in.is_open()) throw badfile();
        kernels = std::string (std::istreambuf_iterator<char>(in), (std::istreambuf_iterator<char>()));
    }
    catch (std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;
        exit(EXIT_FAILURE);
    }

    // Setup OpenCL

    // Get list of devices
    std::vector<cl::Device> devices;
    getDeviceList(devices);

    // Check device index is in range
    try
    {
        if (deviceIndex >= devices.size()) throw invaliddevice();
    }
    catch (std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;
        exit(EXIT_FAILURE);
    }

    cl::Device device = devices[deviceIndex];
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    try
    {
        context = cl::Context(device);
    }
    catch (cl::Error& e)
    {
        die("Creating context", e);
    }

    try
    {
        queue = cl::CommandQueue(context);
    }
    catch (cl::Error &e)
    {
        die("Creating queue", e);
    }

    try
    {
        program = cl::Program(context, kernels);
    }
    catch (cl::Error &e)
    {
        die("Creating program", e);
    }

    // Print out device name
    std::string name = getDeviceName(device);
    std::cout << "Using OpenCL device " << name << std::endl;

    try
    {
        std::string options = "";
        if (useFloat)
            options = "-DFLOAT";
        program.build(options.c_str());
    }
    catch (cl::Error& e)
    {
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        std::cerr
            << "Build error:"
            << buildlog
            << std::endl;
        exit(e.err());
    }

    cl::make_kernel<cl::Buffer&, cl::Buffer&> copy(program, "copy");
    cl::make_kernel<cl::Buffer&, cl::Buffer&> mul(program, "mul");
    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&> add(program, "add");
    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&> triad(program, "triad");

    // Create host vectors
    void *h_a = malloc(ARRAY_SIZE * DATATYPE_SIZE);
    void *h_b = malloc(ARRAY_SIZE * DATATYPE_SIZE);
    void *h_c = malloc(ARRAY_SIZE * DATATYPE_SIZE);

    // Initilise arrays
    for (unsigned int i = 0; i < ARRAY_SIZE; i++)
    {
        if (useFloat)
        {
            ((float*)h_a)[i] = 1.0;
            ((float*)h_b)[i] = 2.0;
            ((float*)h_c)[i] = 0.0;
        }
        else
        {
            ((double*)h_a)[i] = 1.0;
            ((double*)h_b)[i] = 2.0;
            ((double*)h_c)[i] = 0.0;
        }
    }

    // Create device buffers
    cl::Buffer d_a, d_b, d_c;
    try
    {
        d_a = cl::Buffer(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
        d_b = cl::Buffer(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
        d_c = cl::Buffer(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
    }
    catch (cl::Error &e)
    {
        die("Creating buffers", e);
    }

    // Copy host memory to device
    try
    {
        queue.enqueueWriteBuffer(d_a, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_a);
        queue.enqueueWriteBuffer(d_b, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_b);
        queue.enqueueWriteBuffer(d_c, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_c);
    }
    catch (cl::Error &e)
    {
        die("Copying buffers to device", e);
    }

    // Make sure the copies are finished
    try
    {
        queue.finish();
    }
    catch (cl::Error &e)
    {
        die("Queue finish", e);
    }

    // List of times
    std::vector< std::vector<double> > timings;

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    // Main loop
    for (unsigned int k = 0; k < NTIMES; k++)
    {
        std::vector<double> times;
        t1 = std::chrono::high_resolution_clock::now();
        try
        {
            copy(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_c);
            queue.finish();
        }
        catch (cl::Error &e)
        {
            die("Executing copy", e);
        }
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        try
        {
            mul(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_b, d_c);
            queue.finish();
        }
        catch (cl::Error &e)
        {
            die("Executing mul", e);
        }
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        try
        {
            add(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_b, d_c);
            queue.finish();
        }
        catch (cl::Error &e)
        {
            die("Executing add", e);
        }
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        try
        {
            triad(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_b, d_c);
            queue.finish();
        }
        catch (cl::Error &e)
        {
            die("Executing triad", e);
        }
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

        timings.push_back(times);

    }

    // Check solutions
    try
    {
        queue.enqueueReadBuffer(d_a, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_a);
        queue.enqueueReadBuffer(d_b, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_b);
        queue.enqueueReadBuffer(d_c, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_c);
        queue.finish();
    }
    catch (cl::Error &e)
    {
        die("Copying back buffers", e);
    }
        
    if (useFloat)
    {
        check_solution<float>(h_a, h_b, h_c);
    }
    else
    {
        check_solution<double>(h_a, h_b, h_c);
    }

    // Crunch results
    size_t sizes[4] = {
        2 * DATATYPE_SIZE * ARRAY_SIZE,
        2 * DATATYPE_SIZE * ARRAY_SIZE,
        3 * DATATYPE_SIZE * ARRAY_SIZE,
        3 * DATATYPE_SIZE * ARRAY_SIZE
    };
    double min[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
    double max[4] = {0.0, 0.0, 0.0, 0.0};
    double avg[4] = {0.0, 0.0, 0.0, 0.0};
    // Ignore first result
    for (unsigned int i = 1; i < NTIMES; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            avg[j] += timings[i][j];
            min[j] = std::min(min[j], timings[i][j]);
            max[j] = std::max(max[j], timings[i][j]);
        }
    }
    for (int j = 0; j < 4; j++)
        avg[j] /= (double)(NTIMES-1);

    // Display results
    std::string labels[] = {"Copy", "Mul", "Add", "Triad"};
    std::cout
        << std::left << std::setw(12) << "Function"
        << std::left << std::setw(12) << "MBytes/sec"
        << std::left << std::setw(12) << "Min (sec)"
        << std::left << std::setw(12) << "Max"
        << std::left << std::setw(12) << "Average"
        << std::endl;
    for (int j = 0; j < 4; j++)
    {
        std::cout
            << std::left << std::setw(12) << labels[j]
            << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * sizes[j]/min[j]
            << std::left << std::setw(12) << std::setprecision(5) << min[j]
            << std::left << std::setw(12) << std::setprecision(5) << max[j]
            << std::left << std::setw(12) << std::setprecision(5) << avg[j]
            << std::endl;
    }

}




unsigned getDeviceList(std::vector<cl::Device>& devices)
{
    // Get list of platforms
    std::vector<cl::Platform> platforms;
    try
    {
        cl::Platform::get(&platforms);
    }
    catch (cl::Error &e)
    {
        die("Getting platforms", e);
    }

    // Enumerate devices
    for (unsigned int i = 0; i < platforms.size(); i++)
    {
        std::vector<cl::Device> plat_devices;
        try
        {
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
        }
        catch (cl::Error &e)
        {
            die("Getting devices", e);
        }
        devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
    }

    return devices.size();
}


std::string getDeviceName(const cl::Device& device)
{
    std::string name;
    cl_device_info info = CL_DEVICE_NAME;

    try
    {

        // Special case for AMD
#ifdef CL_DEVICE_BOARD_NAME_AMD
        device.getInfo(CL_DEVICE_VENDOR, &name);
        if (strstr(name.c_str(), "Advanced Micro Devices"))
            info = CL_DEVICE_BOARD_NAME_AMD;
#endif

        device.getInfo(info, &name);
    }
    catch (cl::Error &e)
    {
        die("Getting device name", e);
    }

    return name;
}

void listDevices(void)
{
    // Get list of devices
    std::vector<cl::Device> devices;
    getDeviceList(devices);

    // Print device names
    if (devices.size() == 0)
    {
        std::cout << "No devices found." << std::endl;
    }
    else
    {
        std::cout << std::endl;
        std::cout << "Devices:" << std::endl;
        for (unsigned i = 0; i < devices.size(); i++)
        {
            std::cout << i << ": " << getDeviceName(devices[i]) << std::endl;
        }
        std::cout << std::endl;
    }
}

