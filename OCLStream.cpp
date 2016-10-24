
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"

// Cache list of devices
bool cached = false;
std::vector<cl::Device> devices;
void getDeviceList(void);

std::string kernels{R"CLC(

  constant TYPE scalar = startScalar;

  kernel void copy(
    global const TYPE * restrict a,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i];
  }

  kernel void mul(
    global TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    b[i] = scalar * c[i];
  }

  kernel void add(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i] + b[i];
  }

  kernel void triad(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    a[i] = b[i] + scalar * c[i];
  }

)CLC"};


template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  // Setup default OpenCL GPU
  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device = devices[device_index];

  // Print out device information
  std::cout << "Using OpenCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  context = cl::Context(device);
  queue = cl::CommandQueue(context);

  // Create program
  cl::Program program(context, kernels);
  std::ostringstream args;
  args << "-DstartScalar=" << startScalar << " ";
  if (sizeof(T) == sizeof(double))
  {
    args << "-DTYPE=double";
    // Check device can do double
    if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
      throw std::runtime_error("Device does not support double precision, please use --float");
    try
    {
      program.build(args.str().c_str());
    }
    catch (cl::Error& err)
    {
      if (err.err() == CL_BUILD_PROGRAM_FAILURE)
      {
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>()[0].second << std::endl;
        throw err;
      }
    }
  }
  else if (sizeof(T) == sizeof(float))
  {
    args << "-DTYPE=float";
    program.build(args.str().c_str());
  }

  // Create kernels
  copy_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "copy");
  mul_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "mul");
  add_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "add");
  triad_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "triad");

  array_size = ARRAY_SIZE;

  // Check buffers fit on the device
  cl_ulong totalmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  cl_ulong maxbuffer = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  if (maxbuffer < sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device cannot allocate a buffer big enough");
  if (totalmem < 3*sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create buffers
  d_a = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);

}

template <class T>
OCLStream<T>::~OCLStream()
{
  delete copy_kernel;
  delete mul_kernel;
  delete add_kernel;
  delete triad_kernel;
}

template <class T>
void OCLStream<T>::copy()
{
  (*copy_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::mul()
{
  (*mul_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::add()
{
  (*add_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::triad()
{
  (*triad_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  cl::copy(queue, a.begin(), a.end(), d_a);
  cl::copy(queue, b.begin(), b.end(), d_b);
  cl::copy(queue, c.begin(), c.end(), d_c);
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  cl::copy(queue, d_a, a.begin(), a.end());
  cl::copy(queue, d_b, b.begin(), b.end());
  cl::copy(queue, d_c, c.begin(), c.end());
}

void getDeviceList(void)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned i = 0; i < platforms.size(); i++)
  {
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }


}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;
  cl_device_info info = CL_DEVICE_NAME;

  if (device < devices.size())
  {
    devices[device].getInfo(info, &name);
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;

}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    devices[device].getInfo(CL_DRIVER_VERSION, &driver);
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}


template class OCLStream<float>;
template class OCLStream<double>;
