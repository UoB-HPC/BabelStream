/*=============================================================================
*------------------------------------------------------------------------------
* Copyright 2015: Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC
* Based on John D. McCalpin’s original STREAM benchmark for CPUs
*------------------------------------------------------------------------------
* License:
*  1. You are free to use this program and/or to redistribute
*     this program.
*  2. You are free to modify this program for your own use,
*     including commercial use, subject to the publication
*     restrictions in item 3.
*  3. You are free to publish results obtained from running this
*     program, or from works that you derive from this program,
*     with the following limitations:
*     3a. In order to be referred to as "GPU-STREAM benchmark results",
*         published results must be in conformance to the GPU-STREAM
*         Run Rules published at
*         http://github.com/UoB-HPC/GPU-STREAM/wiki/Run-Rules
*         and incorporated herein by reference.
*         The copyright holders retain the
*         right to determine conformity with the Run Rules.
*     3b. Results based on modified source code or on runs not in
*         accordance with the GPU-STREAM Run Rules must be clearly
*         labelled whenever they are published.  Examples of
*         proper labelling include:
*         "tuned GPU-STREAM benchmark results" 
*         "based on a variant of the GPU-STREAM benchmark code"
*         Other comparable, clear and reasonable labelling is
*         acceptable.
*     3c. Submission of results to the GPU-STREAM benchmark web site
*         is encouraged, but not required.
*  4. Use of this program or creation of derived works based on this
*     program constitutes acceptance of these licensing restrictions.
*  5. Absolutely no warranty is expressed or implied.
*———————————————————————————————————-----------------------------------------*/

#include <iomanip>
#include <iostream>
#include <cstring>
#include <limits>

#define VERSION_STRING "0.9"

extern void parseArguments(int argc, char *argv[]);

extern void listDevices(void);

extern int ARRAY_SIZE;
extern int NTIMES;

extern bool useFloat;

extern int deviceIndex;

// Exceptions
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

struct badarraysize : public std::exception
{
    virtual const char * what () const throw ()
    {
        return "Array size must be >= 1024";
    }
};

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

