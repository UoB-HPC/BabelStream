
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

size_t DATATYPE_SIZE = sizeof(double);
bool useFloat = false;

#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

#define VERSION_STRING "0.0"

void parseArguments(int argc, char *argv[]);
std::string getDeviceName(int device);

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

// Code to check CUDA errors
void check_cuda_error(void)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr
            << "Error: "
            << cudaGetErrorString(err)
            << std::endl;
            exit(err);
    }
}

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


template <typename T>
__global__ void copy(const T * a, T * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i];
}

template <typename T>
__global__ void mul(T * b, const T * c)
{
    const T scalar = 3.0;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    b[i] = scalar * c[i];
}

template <typename T>
__global__ void add(const T * a, const T * b, T * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

template <typename T>
__global__ void triad(T * a, const T * b, const T * c)
{
    const T scalar = 3.0;
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
        << "Implementation: CUDA" << std::endl;

    try
    {
        parseArguments(argc, argv);

        if (NTIMES < 2) throw badntimes();

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

        // Display number of bytes in array
        std::cout << std::setprecision(1) << std::fixed
            << "Array size: " << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
            << " (=" << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
            << std::endl;
        std::cout << "Total size: " << 3*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
            << " (=" << 3*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
            << std::endl;

        // Reset precision
        std::cout.precision(ss);

        // Check device index is in range
        int count;
        cudaGetDeviceCount(&count);
        check_cuda_error();
        if (deviceIndex >= count) throw invaliddevice();
        cudaSetDevice(deviceIndex);
        check_cuda_error();

        // Print out device name
        std::cout << "Using CUDA device " << getDeviceName(deviceIndex) << std::endl;


        // Create host vectors
        void * h_a = malloc(ARRAY_SIZE*DATATYPE_SIZE);
        void * h_b = malloc(ARRAY_SIZE*DATATYPE_SIZE);
        void * h_c = malloc(ARRAY_SIZE*DATATYPE_SIZE);

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
        void * d_a, * d_b, *d_c;
        cudaMalloc(&d_a, ARRAY_SIZE*DATATYPE_SIZE);
        check_cuda_error();
        cudaMalloc(&d_b, ARRAY_SIZE*DATATYPE_SIZE);
        check_cuda_error();
        cudaMalloc(&d_c, ARRAY_SIZE*DATATYPE_SIZE);
        check_cuda_error();

        // Copy host memory to device
        cudaMemcpy(d_a, h_a, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyHostToDevice);
        check_cuda_error();
        cudaMemcpy(d_b, h_b, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyHostToDevice);
        check_cuda_error();
        cudaMemcpy(d_c, h_c, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyHostToDevice);
        check_cuda_error();

        // Make sure the copies are finished
        cudaDeviceSynchronize();
        check_cuda_error();

        // List of times
        std::vector< std::vector<double> > timings;

        // Declare timers
        std::chrono::high_resolution_clock::time_point t1, t2;

        // Main loop
        for (unsigned int k = 0; k < NTIMES; k++)
        {
            std::vector<double> times;
            t1 = std::chrono::high_resolution_clock::now();
            if (useFloat)
                copy<<<ARRAY_SIZE/1024, 1024>>>((float*)d_a, (float*)d_c);
            else
                copy<<<ARRAY_SIZE/1024, 1024>>>((double*)d_a, (double*)d_c);
            check_cuda_error();
            cudaDeviceSynchronize();
            check_cuda_error();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            if (useFloat)
                mul<<<ARRAY_SIZE/1024, 1024>>>((float*)d_b, (float*)d_c);
            else
                mul<<<ARRAY_SIZE/1024, 1024>>>((double*)d_b, (double*)d_c);
            check_cuda_error();
            cudaDeviceSynchronize();
            check_cuda_error();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            if (useFloat)
                add<<<ARRAY_SIZE/1024, 1024>>>((float*)d_a, (float*)d_b, (float*)d_c);
            else
                add<<<ARRAY_SIZE/1024, 1024>>>((double*)d_a, (double*)d_b, (double*)d_c);
            check_cuda_error();
            cudaDeviceSynchronize();
            check_cuda_error();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            t1 = std::chrono::high_resolution_clock::now();
            if (useFloat)
                triad<<<ARRAY_SIZE/1024, 1024>>>((float*)d_a, (float*)d_b, (float*)d_c);
            else
                triad<<<ARRAY_SIZE/1024, 1024>>>((double*)d_a, (double*)d_b, (double*)d_c);
            check_cuda_error();
            cudaDeviceSynchronize();
            check_cuda_error();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

            timings.push_back(times);

        }

        // Check solutions
        cudaMemcpy(h_a, d_a, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyDeviceToHost);
        check_cuda_error();
        cudaMemcpy(h_b, d_b, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyDeviceToHost);
        check_cuda_error();
        cudaMemcpy(h_c, d_c, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyDeviceToHost);
        check_cuda_error();
        check_solution(h_a, h_b, h_c);

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
                << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * sizes[j]/min[j]
                << std::left << std::setw(12) << std::setprecision(5) << min[j]
                << std::left << std::setw(12) << std::setprecision(5) << max[j]
                << std::left << std::setw(12) << std::setprecision(5) << avg[j]
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

std::string getDeviceName(int device)
{
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    check_cuda_error();
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
            check_cuda_error();

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
                    std::cout << i << ": " << getDeviceName(i) << std::endl;
                    check_cuda_error();
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
        else if (!strcmp(argv[i], "--float"))
        {
            useFloat = true;
            DATATYPE_SIZE = sizeof(float);
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            std::cout << std::endl;
            std::cout << "Usage: ./gpu-stream-cuda [OPTIONS]" << std::endl << std::endl;
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
