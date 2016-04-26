
#include <iostream>
#include <vector>

#include "common.h"
#include "Stream.h"
#include "CUDAStream.h"


const unsigned int ARRAY_SIZE = 52428800;

#define IMPLEMENTATION_STRING "CUDA"

int main(int argc, char *argv[])
{
  std::cout
    << "GPU-STREAM" << std::endl
    << "Version:" << VERSION_STRING << std::endl
    << "Implementation: " << IMPLEMENTATION_STRING << std::endl;


  // Create host vectors
  std::vector<double> a(ARRAY_SIZE, 1.0);
  std::vector<double> b(ARRAY_SIZE, 2.0);
  std::vector<double> c(ARRAY_SIZE, 0.0);

  Stream<double> *stream;

  // Use the CUDA implementation
  stream = new CUDAStream<double>(ARRAY_SIZE);

  stream->copy();

  delete[] stream;

}
