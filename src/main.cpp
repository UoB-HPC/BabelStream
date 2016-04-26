
#include <iostream>

#include "common.h"
#include "Stream.h"
#include "CUDAStream.h"



int main(int argc, char *argv[])
{
  std::cout
    << "GPU-STREAM" << std::endl
    << "Version:" << VERSION_STRING << std::endl
    << "Implementation: " << std::endl;

  Stream<double> *stream;
  stream = new CUDAStream<double>();
  stream->copy();

  delete[] stream;

}
