
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#define DATATYPE double
#define ARRAY_SIZE 50000000
#define NTIMES 2

#define MIN(a,b) ((a) < (b)) ? (a) : (b)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)

struct badfile : public std::exception
{
  virtual const char * what () const throw ()
  {
    return "Cannot open kernel file";
  }
};

size_t sizes[4] = {
	2 * sizeof(DATATYPE) * ARRAY_SIZE,
	2 * sizeof(DATATYPE) * ARRAY_SIZE,
	3 * sizeof(DATATYPE) * ARRAY_SIZE,
	3 * sizeof(DATATYPE) * ARRAY_SIZE
};


int main(void)
{
	try
	{
		// Open the Kernel source
		std::ifstream in("ocl-stream-kernels.cl");
		if (!in.is_open()) throw badfile();
		std::string kernels(std::istreambuf_iterator<char>(in), (std::istreambuf_iterator<char>()));

		// Setup OpenCL
		cl::Context context(CL_DEVICE_TYPE_CPU);
		cl::CommandQueue queue(context);
		cl::Program program(context, kernels);

		try
		{
			program.build();
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
		std::vector<DATATYPE> h_a(ARRAY_SIZE, 1.0);
		std::vector<DATATYPE> h_b(ARRAY_SIZE, 2.0);
		std::vector<DATATYPE> h_c(ARRAY_SIZE, 0.0);

		// Create device buffers
		cl::Buffer d_a(context, CL_MEM_READ_WRITE, sizeof(DATATYPE) * ARRAY_SIZE);
		cl::Buffer d_b(context, CL_MEM_READ_WRITE, sizeof(DATATYPE) * ARRAY_SIZE);
		cl::Buffer d_c(context, CL_MEM_READ_WRITE, sizeof(DATATYPE) * ARRAY_SIZE);

		// Copy host memory to device
		cl::copy(queue, h_a.begin(), h_a.end(), d_a);
		cl::copy(queue, h_b.begin(), h_b.end(), d_b);
		cl::copy(queue, h_c.begin(), h_c.end(), d_c);

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
			avg[j] /= (double)NTIMES;

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
