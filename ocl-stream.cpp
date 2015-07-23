
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

unsigned int ARRAY_SIZE = 50000000;
unsigned int NTIMES = 10;

size_t DATATYPE_SIZE = sizeof(double);
bool useFloat = false;

#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

#define VERSION_STRING "0.0"

void parseArguments(int argc, char *argv[]);
std::string getDeviceName(const cl::Device& device);
unsigned getDeviceList(std::vector<cl::Device>& devices);

struct badfile : public std::exception
{
    virtual const char * what () const throw ()
    {
        return "Cannot open kernel file";
    }
};

struct invaliddevice : public std::exception
{
    virtual const char * what () const throw ()
    {
        return "Chosen device index is invalid";
    }
};

struct badntimes : public std::exception
{
    virtual const char * what () const throw ()
    {
        return "Chosen number of times is invalid, must be >= 2";
    }
};

size_t sizes[4] = {
    2 * DATATYPE_SIZE * ARRAY_SIZE,
    2 * DATATYPE_SIZE * ARRAY_SIZE,
    3 * DATATYPE_SIZE * ARRAY_SIZE,
    3 * DATATYPE_SIZE * ARRAY_SIZE
};

void check_solution(void* a, void* b, void* c)
{
    // Generate correct solution
    double golda = 1.0;
    double goldb = 2.0;
    double goldc = 0.0;
    float goldaf = 1.0;
    float goldbf = 2.0;
    float goldcf = 0.0;

    const double scalar = 3.0;
    const float scalarf = 3.0;

    for (unsigned int i = 0; i < NTIMES; i++)
    {
        // Double
        goldc = golda;
        goldb = scalar * goldc;
        goldc = golda + goldb;
        golda = goldb + scalar * goldc;
        // Float
        goldcf = goldaf;
        goldbf = scalarf * goldcf;
        goldcf = goldaf + goldbf;
        goldaf = goldbf + scalarf * goldcf;
    }

    // Calculate average error
    double erra = 0.0;
    double errb = 0.0;
    double errc = 0.0;
    for (unsigned int i = 0; i < ARRAY_SIZE; i++)
    {
        if (useFloat)
        {
            erra += fabsf(((float*)a)[i] - goldaf);
            errb += fabsf(((float*)b)[i] - goldbf);
            errc += fabsf(((float*)c)[i] - goldcf);
        }
        else
        {
            erra += fabs(((double*)a)[i] - (double)golda);
            errb += fabs(((double*)b)[i] - (double)goldb);
            errc += fabs(((double*)c)[i] - (double)goldc);
        }
    }
    erra /= (double)ARRAY_SIZE;
    errb /= (double)ARRAY_SIZE;
    errc /= (double)ARRAY_SIZE;

    double epsi;
    if (useFloat) epsi = 1.0E-6;
    else epsi = 1.0E-13;

    if (erra > epsi)
        std::cout
            << "Validation failed on a[]. Average error " << erra
            << std::endl;
    if (errb > epsi)
        std::cout
            << "Validation failed on b[]. Average error " << errb
            << std::endl;
    if (errc > epsi)
        std::cout
            << "Validation failed on c[]. Average error " << errc
            << std::endl;
}

cl_uint deviceIndex = 0;

int main(int argc, char *argv[])
{

    // Print out run information
    std::cout
        << "GPU-STREAM" << std::endl
        << "Version: " << VERSION_STRING << std::endl
        << "Implementation: OpenCL" << std::endl << std::endl;

    try
    {
        parseArguments(argc, argv);

        if (NTIMES < 2) throw badntimes();

        // Open the Kernel source
        std::ifstream in("ocl-stream-kernels.cl");
        if (!in.is_open()) throw badfile();
        std::string kernels(std::istreambuf_iterator<char>(in), (std::istreambuf_iterator<char>()));

        // Setup OpenCL

        // Get list of devices
        std::vector<cl::Device> devices;
        getDeviceList(devices);

        // Check device index is in range
        if (deviceIndex >= devices.size()) throw invaliddevice();

        cl::Device device = devices[deviceIndex];

        cl::Context context(device);
        cl::CommandQueue queue(context);
        cl::Program program(context, kernels);

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
            throw e;
        }

        cl::make_kernel<cl::Buffer, cl::Buffer> copy(program, "copy");
        cl::make_kernel<cl::Buffer, cl::Buffer> mul(program, "mul");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> add(program, "add");
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> triad(program, "triad");

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
        cl::Buffer d_a(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
        cl::Buffer d_b(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
        cl::Buffer d_c(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);

        // Copy host memory to device
        queue.enqueueWriteBuffer(d_a, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_a);
        queue.enqueueWriteBuffer(d_b, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_b);
        queue.enqueueWriteBuffer(d_c, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_c);

        // Make sure the copies are finished
        queue.finish();

        // List of times
        std::vector< std::vector<double> > timings;

        // Declare timers
        std::chrono::high_resolution_clock::time_point t1, t2;

        // Main loop
        for (unsigned int k = 0; k < NTIMES; k++)
        {
            std::vector<double> times;
            t1 = std::chrono::high_resolution_clock::now();
            copy(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            mul(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_b, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            add(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_b, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            triad(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_b, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

            timings.push_back(times);

        }

        // Check solutions
        queue.enqueueReadBuffer(d_a, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_a);
        queue.enqueueReadBuffer(d_b, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_b);
        queue.enqueueReadBuffer(d_c, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_c);
        queue.finish();
        check_solution(h_a, h_b, h_c);

        // Crunch results
        double min[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
        double max[4] = {0.0, 0.0, 0.0, 0.0};
        double avg[4] = {0.0, 0.0, 0.0, 0.0};
        // Ignore first result
        for (unsigned int i = 1; i < NTIMES; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                avg[j] += timings[i][j];
                min[j] = MIN(min[j], timings[i][j]);
                max[j] = MAX(max[j], timings[i][j]);
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
                << std::left << std::setw(12) << 1.0E-06 * sizes[j]/min[j]
                << std::left << std::setw(12) << min[j]
                << std::left << std::setw(12) << max[j]
                << std::left << std::setw(12) << avg[j]
                << std::endl;
        }

    }
    // Catch OpenCL Errors and display information
    catch (cl::Error& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << "(" << e.err() << ")"
            << std::endl;
    }
    catch (std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;
    }
}

unsigned getDeviceList(std::vector<cl::Device>& devices)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned int i = 0; i < platforms.size(); i++)
  {
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }

  return devices.size();
}

std::string getDeviceName(const cl::Device& device)
{
    std::string name;
    cl_device_info info = CL_DEVICE_NAME;

    // Special case for AMD
#ifdef CL_DEVICE_BOARD_NAME_AMD
    device.getInfo(CL_DEVICE_VENDOR, &name);
    if (strstr(name.c_str(), "Advanced Micro Devices"))
        info = CL_DEVICE_BOARD_NAME_AMD;
#endif

    device.getInfo(info, &name);
    return name;
}


int parseUInt(const char *str, cl_uint *output)
{
    char *next;
    *output = strtoul(str, &next, 10);
    return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--list"))
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
            exit(0);
        }
        else if (!strcmp(argv[i], "--device"))
        {
            if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
            {
                std::cout << "Invalid device index" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--arraysize") || !strcmp(argv[i], "-s"))
        {
            if (++i >= argc || !parseUInt(argv[i], &ARRAY_SIZE))
            {
                std::cout << "Invalid array size" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--numtimes") || !strcmp(argv[i], "-n"))
        {
            if (++i >= argc || !parseUInt(argv[i], &NTIMES))
            {
                std::cout << "Invalid number of times" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--float"))
        {
            useFloat = true;
            DATATYPE_SIZE = sizeof(float);
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            std::cout << std::endl;
            std::cout << "Usage: ./gpu-stream-ocl [OPTIONS]" << std::endl << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h  --help               Print the message" << std::endl;
            std::cout << "      --list               List available devices" << std::endl;
            std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
            std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
            std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
            std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
            std::cout << std::endl;
            exit(0);
        }
        else
        {
            std::cout << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
            exit(1);
        }
    }
}
