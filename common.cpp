#include "common.h"

int ARRAY_SIZE = 50000000;
int NTIMES = 10;

bool useFloat = false;

int deviceIndex = 0;

int parseInt(const char *str, int *output)
{
    char *next;
    *output = strtol(str, &next, 10);
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
            if (++i >= argc || !parseInt(argv[i], &deviceIndex))
            {
                std::cout << "Invalid device index" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--arraysize") || !strcmp(argv[i], "-s"))
        {
            if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE))
            {
                std::cout << "Invalid array size" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--numtimes") || !strcmp(argv[i], "-n"))
        {
            if (++i >= argc || !parseInt(argv[i], &NTIMES))
            {
                std::cout << "Invalid number of times" << std::endl;
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--float"))
        {
            useFloat = true;
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
