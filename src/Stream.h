
#pragma once

#include <vector>

template <class T>
class Stream
{
  public:
    // Kernels
    // These must be blocking calls
    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;

    // Copy memory between host and device
    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) = 0;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) = 0;

    // Implementation specific device functions
    static std::vector<int> getDeviceList();
    static std::vector<int> getDeviceName();
    static std::vector<int> getDeviceDriver();

};

