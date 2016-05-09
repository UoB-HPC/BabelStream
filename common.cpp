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

#include "common.h"

// Default array size 50 * 2^20 (50*8 Mebibytes double precision)
// Use binary powers of two so divides 1024
unsigned int ARRAY_SIZE = 52428800;
size_t   ARRAY_PAD_BYTES  = 0;

unsigned int NTIMES = 10;

bool useFloat = false;
unsigned int  groups   = 0;
unsigned int  groupSize   = 1024;

unsigned int deviceIndex = 0;

int parseUInt(const char *str, unsigned int *output)
{
    char *next;
    *output = strtoul(str, &next, 10);
    return !strlen(next);
}

int parseSize(const char *str, size_t *output)
{
    char *next;
    *output = strtoull(str, &next, 0);
	int l = strlen(str);
	if (l) {
		char c = str[l-1]; // last char.
		if ((c == 'k') || (c == 'K')) {
			*output *= 1024;
		}
		if ((c == 'm') || (c == 'M')) {
			*output *= (1024*1024);
		}

	}
    return !strlen(next);
}


void parseArguments(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--list"))
        {
            listDevices();
            exit(0);
        }
        else if (!strcmp(argv[i], "--device"))
        {
            if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
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
        else if (!strcmp(argv[i], "--groups"))
        {
            if (++i >= argc || !parseUInt(argv[i], &groups))
            {
                std::cout << "Invalid group number" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--groupSize"))
        {
            if (++i >= argc || !parseUInt(argv[i], &groupSize))
            {
                std::cout << "Invalid group size" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--pad"))
        {
            if (++i >= argc || !parseSize(argv[i], &ARRAY_PAD_BYTES))
            {
                std::cout << "Invalid size" << std::endl;
                exit(1);
            }

        }
        else if (!strcmp(argv[i], "--float"))
        {
            useFloat = true;
            std::cout << "Warning: If number of iterations set >= 8, expect rounding errors with single precision, not apply to AMD device" << std::endl;
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
            std::cout << "      --groups             Set number of groups to launch -  each work-item proceses multiple array items" << std::endl;
            std::cout << "      --groupSize          Set size of each group (default 1024)" << std::endl;
            std::cout << "      --pad                Add additional array padding. Can use trailing K (KB) or M (MB)" << std::endl;
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
