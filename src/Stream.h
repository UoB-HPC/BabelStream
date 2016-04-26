
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



    // Implementation specific device functions
    static std::vector<int> getDeviceList();
    static std::vector<int> getDeviceName();
    static std::vector<int> getDeviceDriver();

};
