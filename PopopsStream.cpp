
// Copyright (c) 2015-16 Thorben Louw, Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "PoplarStream.h"
#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>
#include <popops/codelets.hpp>
#include <map>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <optional>
#include <popops/Reduce.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/Util.hpp>
#include <poputil/TileMapping.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops;

enum Programs
{
    INIT_PROGRAM,
    COPY_PROGRAM,
    MUL_PROGRAM,
    ADD_PROGRAM,
    TRIAD_PROGRAM,
    DOT_PROGRAM,
    STREAM_BACK_TO_HOST_PROGRAM
};

// const OptionFlags POPLAR_ENGINE_OPTIONS{
//         {"target.saveArchive",                "archive.a"},
//         {"debug.instrument",                  "true"},
//         {"debug.instrumentCompute",           "true"},
//         {"debug.loweredVarDumpFile",          "vars.capnp"},
//         {"debug.instrumentControlFlow",       "true"},
//         {"debug.computeInstrumentationLevel", "tile"}
// };

const OptionFlags POPLAR_ENGINE_OPTIONS{
    {"debug.instrument", "false"}};

// This is due to https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion,
// and is an ultra-hacky way to convert the initial array values to half without having a full half library
// on the host
uint16_t toHalf(float val)
{
    uint32_t x = static_cast<uint32_t>(val);
    return ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
}

void captureProfileInfo(Engine &engine)
{
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);

    poplar::serializeToJSON(graphOfs, engine.getGraphProfile(), false);
    poplar::serializeToJSON(executionOfs, engine.getExecutionProfile(), false);

    graphOfs.close();
    executionOfs.close();
}

std::optional<Device> getIpuDevice(const unsigned deviceType = 0)
{
    const auto validOptions = std::array<unsigned, 6>{0, 1, 2, 4, 8, 16};
    const bool isValid = std::find(validOptions.begin(), validOptions.end(), deviceType) != validOptions.end();
    if (isValid)
    {
        if (deviceType == 0)
        { // Use the CPUDevice
            // Note that as of Poplar v1.1.11, this ony returns a useless device with 1 tile and 256Kb of memory!
            return std::optional<Device>(Device::createCPUDevice());
        }
        else
        {
            // Target an IPUDevice
            DeviceManager manager = DeviceManager::createDeviceManager();
            Device device;
            for (auto &hwDevice : manager.getDevices(poplar::TargetType::IPU, deviceType))
            {
                device = std::move(hwDevice);
                if (device.attach())
                {
                    std::cout << "Attached to IPU " << device.getId() << std::endl;
                    return std::optional<Device>(std::move(device));
                }
            }
        }
    }
    return std::nullopt;
}

void listDevices()
{

    DeviceManager manager = DeviceManager::createDeviceManager();

    std::cout << 0 << ": "
              << "CPUDevice" << std::endl;

    // Attempt to attach to a single IPU:
    Device device;
    if (auto devices = manager.getDevices(poplar::TargetType::IPU, 1); !devices.empty())
    {
        std::cout << 1 << ": "
                  << "IPUDevice" << std::endl;
    }

    // Attempt to attach to 2 IPUs:
    if (auto devices = manager.getDevices(poplar::TargetType::IPU, 2); !devices.empty())
    {
        std::cout << 2 << ": "
                  << "2x IPUDevices" << std::endl;
    }

    // Attempt to attach to 4 IPUs:
    if (auto devices = manager.getDevices(poplar::TargetType::IPU, 4); !devices.empty())
    {
        std::cout << 4 << ": "
                  << "4x IPUDevices" << std::endl;
    }

    // Attempt to attach to 8 IPUs:
    if (auto devices = manager.getDevices(poplar::TargetType::IPU, 8); !devices.empty())
    {
        std::cout << 8 << ": "
                  << "8x IPUDevices" << std::endl;
    }

    // Attempt to attach to 16 IPUs:
    if (auto devices = manager.getDevices(poplar::TargetType::IPU, 16); !devices.empty())
    {
        std::cout << 16 << ": "
                  << "16x IPUDevices" << std::endl;
    }
}

class PoplarStreamUtil
{
private:
    std::map<std::string, Tensor> tensors = {};
    std::vector<Program> programs = {};
    std::map<std::string, DataStream> dataStreams = {};

    // This could be done faster with a Stream copy, but that creates a very large FIFO
    // which limits the array size even further
    Program initProgram(Graph &graph, size_t arraySize)
    {
        Sequence(s);
        tensors["a"] = poputil::duplicate(graph, tensors["initA"].reshape({1}).broadcast(arraySize, 0), s, "a");
        tensors["b"] = poputil::duplicate(graph, tensors["initB"].reshape({1}).broadcast(arraySize, 0), s, "b");
        tensors["c"] = poputil::duplicate(graph, tensors["initC"].reshape({1}).broadcast(arraySize, 0), s, "c");
        poputil::mapTensorLinearly(graph, tensors["a"]);
        poputil::mapTensorLinearly(graph, tensors["b"]);
        poputil::mapTensorLinearly(graph, tensors["c"]);

        // s.add(PrintTensor("a", tensors["a"]));
        // s.add(PrintTensor("b", tensors["b"]));
        // s.add(PrintTensor("c", tensors["c"]));

        return s;
    }

    // c = a
    Program copyProgram(Graph &graph)
    {
        auto s = Sequence();
        s.add(Copy(tensors["a"], tensors["c"]));
    //    s.add(PrintTensor("c (=a)", tensors["c"]));
        return s;
    }

    // b[i] = x * c[i];
    Program mulProgram(Graph &graph)
    {
        auto s = Sequence();
        s.add(Copy(popops::mul(graph, tensors["c"], tensors["alpha"], s, "Mul"), tensors["b"]));
     //   s.add(PrintTensor("b (=xc)", tensors["b"]));
        return s;
    }

    // c = a + b
    Program addProgram(Graph &graph)
    {
        auto s = Sequence();
        s.add(Copy(popops::add(graph, tensors["a"], tensors["b"], s, "Add"), tensors["c"]));
     //   s.add(PrintTensor("c (=a+b)", tensors["c"]));
        return s;
    }

    // a = b + xc
    Program triadProgram(Graph &graph)
    {
        auto s = Sequence();
        s.add(Copy(tensors["b"], tensors["a"]));
        popops::scaledAddTo(graph, tensors["a"], tensors["c"], tensors["alpha"], s, "Triad", {{"optimizeForSpeed", "true"}});
     //   s.add(PrintTensor("a (=b + xc)", tensors["a"]));
        return s;
    }

    // sum = reduce+(a * b)
    Program dotProdProgram(Graph &graph)
    {
        Sequence s;

        popops::reduceWithOutput(graph,
                                 popops::mul(graph, tensors["a"], tensors["b"], s, "a*b"), tensors["sum"], {0},
                                 {popops::Operation::ADD},
                                 s,
                                 "reduce+");
     //   s.add(PrintTensor("reduce+(a * b)", tensors["sum"]));

        return s;
    }

    void createAndLayOutTensors(poplar::Type type, Graph &graph)
    {
        tensors["initA"] = graph.addVariable(type, {}, "initA");
        tensors["initB"] = graph.addVariable(type, {}, "initB");
        tensors["initC"] = graph.addVariable(type, {}, "initC");

        tensors["alpha"] = graph.addConstant(type, {}, startScalar, "alpha");
        tensors["sum"] = graph.addVariable(FLOAT, {}, "sum");

        graph.createHostRead("sum", tensors["sum"]);
        graph.createHostWrite("initA", tensors["initA"]);
        graph.createHostWrite("initB", tensors["initB"]);
        graph.createHostWrite("initC", tensors["initC"]);

        graph.setTileMapping(tensors["sum"], 4);
        graph.setTileMapping(tensors["initA"], 0);
        graph.setTileMapping(tensors["initB"], 1);
        graph.setTileMapping(tensors["initC"], 2);
        graph.setTileMapping(tensors["alpha"], 3);
    }

    void createDataStreams(poplar::Type type, Graph &graph, size_t arraySize)
    {
        dataStreams["out_a"] = graph.addDeviceToHostFIFO("out_a", type, arraySize);
        dataStreams["out_b"] = graph.addDeviceToHostFIFO("out_b", type, arraySize);
        dataStreams["out_c"] = graph.addDeviceToHostFIFO("out_c", type, arraySize);
    }

public:
    std::unique_ptr<Engine> prepareEngine(const Device &device,
                                          const Graph &graph,
                                          void *a,
                                          void *b,
                                          void *c)
    {
        assert(!programs.empty());
        auto engine = std::make_unique<Engine>(graph, programs, POPLAR_ENGINE_OPTIONS);

        engine->connectStream("out_a", a);
        engine->connectStream("out_b", b);
        engine->connectStream("out_c", c);

        engine->load(device);

        return std::move(engine);
    }

    void buildComputeGraph(poplar::Type type, Graph &graph, size_t arraySize)
    {

        createDataStreams(type, graph, arraySize);

        createAndLayOutTensors(type, graph);

        auto InitProg = initProgram(graph, arraySize);

        auto CopyProg = copyProgram(graph);
        auto MulProg = mulProgram(graph);
        auto AddProg = addProgram(graph);
        auto TriadProg = triadProgram(graph);
        auto DotProg = dotProdProgram(graph);

        auto StreamToHostProg = Sequence(Copy(tensors["a"], dataStreams["out_a"]),
                                         Copy(tensors["b"], dataStreams["out_b"]),
                                         Copy(tensors["c"], dataStreams["out_c"]));

        programs = {InitProg, CopyProg, MulProg, AddProg, TriadProg, DotProg,
                    StreamToHostProg};
    }
};

template <class T>
PoplarStream<T>::PoplarStream(const unsigned int arraySize, const int device_num, const bool halfPrecision) : arraySize(arraySize),
                                                                                                              halfPrecision(halfPrecision),
                                                                                                              a(std::unique_ptr<T[]>(new T[arraySize]())),
                                                                                                              b(std::unique_ptr<T[]>(new T[arraySize]())),
                                                                                                              c(std::unique_ptr<T[]>(new T[arraySize]()))
{

    auto device = getIpuDevice(device_num);

    if (!device.has_value())
    {
        throw std::runtime_error("Could not allocate IPU device");
    }

    Graph graph(device.value());
    const auto numTiles = graph.getTarget().getNumTiles();
    const auto numWorkers = graph.getTarget().getNumWorkerContexts();
    const auto maxBytesPerTile = graph.getTarget().getBytesPerTile();
    const auto clockFrequency = graph.getTarget().getTileClockFrequency();

    const auto maxArraySize = ((double)numTiles) * maxBytesPerTile / 1024.0 / 1024.0 / 3;

    std::cout << "Using IPU with " << numTiles << " tiles, each with " << numWorkers
              << " workers and " << maxBytesPerTile / 1024 << "KB of memory per tile, and clock frequency "
              << (int)clockFrequency / 1000 / 1000 << " MHz. Maximum array size will be slightly less than "
              << std::fixed << std::setprecision(2) << std::floor(maxArraySize) << " MB"
              << std::endl;

    auto util = PoplarStreamUtil();

    graph.addCodelets("PoplarKernels.cpp");
    popops::addCodelets(graph);

    if (sizeof(T) > sizeof(float))
    {
        throw std::runtime_error("Device does not support double precision, please use --float");
    }

    // Check buffers fit on the device
    size_t sizeT = (sizeof(T) == sizeof(float) && halfPrecision) ? sizeof(T) / 2 : sizeof(T);
    unsigned long maxbuffer = ((unsigned long)numTiles) * maxBytesPerTile;
    unsigned long totalmem = ((unsigned long)numTiles) * maxBytesPerTile;
    if (maxbuffer < sizeT * ((unsigned long)arraySize))
        throw std::runtime_error("Device cannot allocate a buffer big enough");
    if (totalmem < 3L * sizeT * arraySize)
        throw std::runtime_error("Device does not have enough memory for all 3 buffers");

    util.buildComputeGraph(halfPrecision ? HALF : FLOAT, graph, arraySize);
    engine = util.prepareEngine(device.value(), graph, a.get(), b.get(), c.get());
}

template <class T>
PoplarStream<T>::~PoplarStream() = default;

template <class T>
void PoplarStream<T>::copy()
{
    engine->run(COPY_PROGRAM);
}

template <class T>
void PoplarStream<T>::mul()
{
    engine->run(MUL_PROGRAM);
}

template <class T>
void PoplarStream<T>::add()
{
    engine->run(ADD_PROGRAM);
}

template <class T>
void PoplarStream<T>::triad()
{
    engine->run(TRIAD_PROGRAM);
}

template <class T>
T PoplarStream<T>::dot()
{
    engine->run(DOT_PROGRAM);
    engine->readTensor("sum", &sum);
    return sum;
}

template <class T>
void PoplarStream<T>::init_arrays(T initA, T initB, T initC)
{

    if (halfPrecision)
    {
        const uint32_t fakeA = toHalf(initA);
        const uint32_t fakeB = toHalf(initB);
        const uint32_t fakeC = toHalf(initC);

        engine->writeTensor("initA", &fakeA);
        engine->writeTensor("initB", &fakeB);
        engine->writeTensor("initC", &fakeC);
    }
    else
    {
        engine->writeTensor("initA", &initA);
        engine->writeTensor("initB", &initB);
        engine->writeTensor("initC", &initC);
    }
    engine->run(INIT_PROGRAM);
}

template <class T>
void PoplarStream<T>::read_arrays(std::vector<T> &h_a, std::vector<T> &h_b, std::vector<T> &h_c)
{

    engine->run(STREAM_BACK_TO_HOST_PROGRAM);

    for (unsigned i = 0; i < arraySize; i++)
    {
        h_a[i] = a[i];
        h_b[i] = b[i];
        h_c[i] = c[i];
    }

    // captureProfileInfo(*engine);
    // engine->printProfileSummary(std::cout,
    //                          OptionFlags{{"showExecutionSteps", "true"}});
}

template class PoplarStream<float>;

template class PoplarStream<double>; // Not usable, but needs to exist for stream.cpp
