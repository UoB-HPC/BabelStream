
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>

#include <cuda.h>

#define DATATYPE double
unsigned int ARRAY_SIZE = 50000000;
unsigned int NTIMES = 10;

#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

#define VERSION_STRING "0.0"

void parseArguments(int argc, char *argv[]);
std::string getDeviceName();

struct badtype : public std::exception
{
    virtual const char * what () const throw ()
    {
        return "Datatype is not 4 or 8";
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
    2 * sizeof(DATATYPE) * ARRAY_SIZE,
    2 * sizeof(DATATYPE) * ARRAY_SIZE,
    3 * sizeof(DATATYPE) * ARRAY_SIZE,
    3 * sizeof(DATATYPE) * ARRAY_SIZE
};

void check_solution(DATATYPE * a, DATATYPE * b, DATATYPE * c)
{
    // Generate correct solution
    DATATYPE golda = 1.0;
    DATATYPE goldb = 2.0;
    DATATYPE goldc = 0.0;

    const DATATYPE scalar = 3.0;

    for (unsigned int i = 0; i < NTIMES; i++)
    {
        goldc = golda;
        goldb = scalar * goldc;
        goldc = golda + goldb;
        golda = goldb + scalar * goldc;
    }

    // Calculate average error
    double erra = 0.0;
    double errb = 0.0;
    double errc = 0.0;
    for (unsigned int i = 0; i < ARRAY_SIZE; i++)
    {
        erra += fabs(a[i] - golda);
        errb += fabs(b[i] - goldb);
        errc += fabs(c[i] - goldc);
    }
    erra /= (double)ARRAY_SIZE;
    errb /= (double)ARRAY_SIZE;
    errc /= (double)ARRAY_SIZE;

    double epsi;
    if (sizeof(DATATYPE) == 4) epsi = 1.0E-6;
    else if (sizeof(DATATYPE) == 8) epsi = 1.0E-13;
    else throw badtype();

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

const DATATYPE scalar = 3.0;

__global__ void copy(const DATATYPE * a, DATATYPE * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i];
}

__global__ void mul(DATATYPE * b, const DATATYPE * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    b[i] = scalar * c[i];
}

__global__ void add(const DATATYPE * a, const DATATYPE * b, DATATYPE * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void triad(DATATYPE * a, const DATATYPE * b, const DATATYPE * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    a[i] = b[i] + scalar * c[i];
}

int deviceIndex = 0;

int main(int argc, char *argv[])
{

    // Print out run information
    std::cout
        << "GPU-STREAM" << std::endl
        << "Version: " << VERSION_STRING << std::endl
        << "Implementation: CUDA" << std::endl << std::endl;

    if (ARRAY_SIZE % 1024 != 0)
    {
        unsigned int OLD_ARRAY_SIZE = ARRAY_SIZE;
        ARRAY_SIZE -= ARRAY_SIZE % 1024;
        std::cout
            << "Warning: array size must divide 1024" << std::endl
            << "Resizing array from " << OLD_ARRAY_SIZE
            << " to " << ARRAY_SIZE << std::endl;
    }

    try
    {
        parseArguments(argc, argv);

        if (NTIMES < 2) throw badntimes();

        // Check device index is in range
        int count;
        cudaGetDeviceCount(&count);
        if (deviceIndex >= count) throw invaliddevice();
        cudaSetDevice(deviceIndex);

        // Print out device name
        std::cout << "Using CUDA device " << getDeviceName() << std::endl;


        // Create host vectors
        DATATYPE * h_a = (DATATYPE *) malloc(ARRAY_SIZE*sizeof(DATATYPE));
        DATATYPE * h_b = (DATATYPE *) malloc(ARRAY_SIZE*sizeof(DATATYPE));
        DATATYPE * h_c = (DATATYPE *) malloc(ARRAY_SIZE*sizeof(DATATYPE));

        // Initilise host vectors
        for (unsigned int i = 0; i < ARRAY_SIZE; i++)
        {
            h_a[i] = 1.0;
            h_b[i] = 2.0;
            h_c[i] = 0.0;
        }

        // Create device buffers
        DATATYPE * d_a, * d_b, *d_c;
        cudaMalloc(&d_a, ARRAY_SIZE*sizeof(DATATYPE));
        cudaMalloc(&d_b, ARRAY_SIZE*sizeof(DATATYPE));
        cudaMalloc(&d_c, ARRAY_SIZE*sizeof(DATATYPE));

        // Copy host memory to device
        cudaMemcpy(d_a, h_a, ARRAY_SIZE*sizeof(DATATYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, ARRAY_SIZE*sizeof(DATATYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c, ARRAY_SIZE*sizeof(DATATYPE), cudaMemcpyHostToDevice);

        // Make sure the copies are finished
        cudaDeviceSynchronize();

        // List of times
        std::vector< std::vector<double> > timings;

        // Declare timers
        std::chrono::high_resolution_clock::time_point t1, t2;

        // Main loop
        for (unsigned int k = 0; k < NTIMES; k++)
        {
            std::vector<double> times;
            t1 = std::chrono::high_resolution_clock::now();
            copy<<<ARRAY_SIZE/1024, 1024>>>(d_a, d_c);
            cudaDeviceSynchronize();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            mul<<<ARRAY_SIZE/1024, 1024>>>(d_b, d_c);
            cudaDeviceSynchronize();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            add<<<ARRAY_SIZE/1024, 1024>>>(d_a, d_b, d_c);
            cudaDeviceSynchronize();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            triad<<<ARRAY_SIZE/1024, 1024>>>(d_a, d_b, d_c);
            cudaDeviceSynchronize();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

            timings.push_back(times);

        }

        // Check solutions
        cudaMemcpy(h_a, d_a, ARRAY_SIZE*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b, d_b, ARRAY_SIZE*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c, d_c, ARRAY_SIZE*sizeof(DATATYPE), cudaMemcpyDeviceToHost);
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
    catch (std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;
    }
}

std::string getDeviceName()
{
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return std::string(prop.name);
}


int parseUInt(const char *str, unsigned int *output)
{
    char *next;
    *output = strtoul(str, &next, 10);
    return !strlen(next);
}

int parseInt(const char *str, int *output)
{
    char *next;
    *output = strtol(str, &next, 10);
    return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--list"))
        {
            // Get number of devices
            int count;
            cudaGetDeviceCount(&count);

            // Print device names
            if (count == 0)
            {
                std::cout << "No devices found." << std::endl;
            }
            else
            {
                std::cout << std::endl;
                std::cout << "Devices:" << std::endl;
                for (int i = 0; i < count; i++)
                {
                    cudaSetDevice(i);
                    std::cout << i << ": " << getDeviceName() << std::endl;
                }
                std::cout << std::endl;
            }
            exit(0);
        }
        else if (!strcmp(argv[i], "--device"))
        {
            if (++i >= argc || !parseInt(argv[i], &deviceIndex))
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
