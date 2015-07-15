
#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#define DATATYPE double
#define ARRAY_SIZE 1000000

struct badfile : public std::exception
{
  virtual const char * what () const throw ()
  {
    return "Cannot open kernel file";
  }
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
		cl::Context context(CL_DEVICE_TYPE_GPU);
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
		cl::make_kernel<cl::Buffer, cl::Buffer> add(program, "add");
		cl::make_kernel<cl::Buffer, cl::Buffer> triad(program, "triad");

		// Create host vectors
		std::vector<DATATYPE> h_a(ARRAY_SIZE, 1.0);
		std::vector<DATATYPE> h_b(ARRAY_SIZE, 2.0);
		std::vector<DATATYPE> h_c(ARRAY_SIZE, 0.0);
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
