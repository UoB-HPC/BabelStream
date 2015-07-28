#include <iomanip>
#include <iostream>
#include <cstring>
#include <limits>

extern void parseArguments(int argc, char *argv[]);

extern void listDevices(void);

extern int ARRAY_SIZE;
extern int NTIMES;

extern bool useFloat;

extern int deviceIndex;

template < typename T >
void check_solution(void* a_in, void* b_in, void* c_in)
{
    // Generate correct solution
    T golda = 1.0;
    T goldb = 2.0;
    T goldc = 0.0;

    T * a = static_cast<T*>(a_in);
    T * b = static_cast<T*>(b_in);
    T * c = static_cast<T*>(c_in);

    const T scalar = 3.0;

    for (unsigned int i = 0; i < NTIMES; i++)
    {
        // Double
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

    erra /= ARRAY_SIZE;
    errb /= ARRAY_SIZE;
    errc /= ARRAY_SIZE;

    double epsi = std::numeric_limits<T>::epsilon() * 100;

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

#define VERSION_STRING "0.0"

