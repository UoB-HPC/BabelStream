
// Copyright (c) 2015-16 Thorben Louw, Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "PoplarStream.h"
#include <poplar/Engine.hpp>
#include <poplar/Target.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplin/codelets.hpp>
#include <map>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <optional>

using namespace poplar;
using namespace poplar::program;

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

#ifdef DEBUG
const OptionFlags POPLAR_ENGINE_OPTIONS{
    {"target.saveArchive", "archive.a"},
    {"debug.instrument", "true"},
    {"debug.instrumentCompute", "true"},
    {"debug.loweredVarDumpFile", "vars.capnp"},
    {"debug.instrumentControlFlow", "true"},
    {"debug.computeInstrumentationLevel", "tile"}};
#else
const OptionFlags POPLAR_ENGINE_OPTIONS{
    {"debug.instrument", "false"}};
#endif


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
    auto multiIpu = std::array<unsigned, 4>{2, 4, 8, 16};
    for (auto i : multiIpu)
    {
        if (auto devices = manager.getDevices(poplar::TargetType::IPU, i); !devices.empty())
        {
            std::cout << i << ": "
                      << "IPUDevice" << std::endl;
        }
    }
}

class PoplarStreamUtil
{
private:
    const unsigned numTiles;
    const unsigned numWorkersPerTile;
    const unsigned arraySize;

    const unsigned totalTilesThatWillBeUsed;
    const unsigned totalWorkersThatWillBeUsed;

    std::map<std::string, Tensor> tensors = {};
    std::vector<Program> programs = {};
    std::map<std::string, DataStream> dataStreams = {};

    [[nodiscard]] unsigned numItemsForTileAndWorker(const unsigned tileNum, const unsigned workerNum) const
    {
        if (arraySize <= numTiles)
        { // Just use 1 item per worker
            return 1;
        }
        else
        { // Balance as fairly as possible
            auto extra = tileNum < arraySize % numTiles;
            auto totalForTile = unsigned(arraySize / numTiles) + extra;
            auto extraForThread = workerNum < totalForTile % numWorkersPerTile;
            return unsigned(totalForTile / numWorkersPerTile) + extraForThread;
        }
    }

    [[nodiscard]] unsigned numItemsForTile(const unsigned tileNum) const
    {
        if (arraySize <= numTiles)
        { // We'll just use one item per worker
            return numWorkersUsedOnTile(tileNum);
        }
        else
        { // Now we must balance as fairly as possible
            auto extra = tileNum < arraySize % numTiles;
            return unsigned(arraySize / numTiles) + extra;
        }
    }

    [[nodiscard]] unsigned numWorkersUsedOnTile(const unsigned tileNum) const
    {
        return std::min(numWorkersPerTile, totalWorkersThatWillBeUsed - tileNum * numWorkersPerTile);
    }

    // This could be done faster with a Stream copy, but that creates a very large FIFO
    // which limits the array size even further
    Program initProgram(const std::string &templateStr, Graph &graph)
    {
        ComputeSet cs = graph.addComputeSet("init");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++)
        {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "InitKernel" + templateStr,
                {
                    {"a", tensors["a"].slice({idx}, {idx + numItems}).flatten()},
                    {"b", tensors["b"].slice({idx}, {idx + numItems}).flatten()},
                    {"c", tensors["c"].slice({idx}, {idx + numItems}).flatten()},
                    {"initA", tensors["initA"]},
                    {"initB", tensors["initB"]},
                    {"initC", tensors["initC"]},
                });
            graph.setInitialValue(v["size"], unsigned(numItems));
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }

    Program copyProgram(const std::string &templateStr, Graph &graph)
    {
        ComputeSet cs = graph.addComputeSet("copy");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++)
        {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(cs,
                                           "CopyKernel" + templateStr,
                                           {{"a", tensors["a"].slice({idx}, {idx + numItems}).flatten()},
                                            {"c", tensors["c"].slice({idx}, {idx + numItems}).flatten()}});
            graph.setInitialValue(v["size"], unsigned(numItems));

            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }

    Program mulProgram(const std::string &templateStr, Graph &graph)
    {
        ComputeSet cs = graph.addComputeSet("mul");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++)
        {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "MulKernel" + templateStr,
                {
                    {"b", tensors["b"].slice({idx}, {idx + numItems})},
                    {"c", tensors["c"].slice({idx}, {idx + numItems})},
                });
            graph.setInitialValue(v["size"], unsigned(numItems));
            graph.setInitialValue(v["alpha"], float(startScalar));
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }

    Program addProgram(const std::string &templateStr, Graph &graph)
    {
        ComputeSet cs = graph.addComputeSet("add");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++)
        {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "AddKernel" + templateStr,
                {
                    {"b", tensors["b"].slice({idx}, {idx + numItems})},
                    {"a", tensors["a"].slice({idx}, {idx + numItems})},
                    {"c", tensors["c"].slice({idx}, {idx + numItems})},
                });
            graph.setInitialValue(v["size"], unsigned(numItems));
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }

    Program triadProgram(const std::string &templateStr, Graph &graph)
    {
        ComputeSet cs = graph.addComputeSet("triad");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++)
        {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;

            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "TriadKernel" + templateStr,
                {
                    {"b", tensors["b"].slice({idx}, {idx + numItems})},
                    {"a", tensors["a"].slice({idx}, {idx + numItems})},
                    {"c", tensors["c"].slice({idx}, {idx + numItems})},
                });
            graph.setInitialValue(v["size"], unsigned(numItems));
            graph.setInitialValue(v["alpha"], float(startScalar));
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }

    Program dotProdProgram(const std::string &templateStr, Graph &graph)
    {
        ComputeSet dotCs = graph.addComputeSet("dot");
        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++)
        {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                dotCs,
                "DotProdKernel" + templateStr,
                {{"b", tensors["b"].slice({idx}, {idx + numItems})},
                 {"a", tensors["a"].slice({idx}, {idx + numItems})},
                 {"sum", tensors["partialSumsPerWorker"][tile * numWorkersPerTile +
                                                         worker]}

                });
            graph.setInitialValue(v["size"], unsigned(numItems));
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        ComputeSet partialReduxCs = graph.addComputeSet("reductionPerTile");
        idx = 0u;
        for (unsigned tile = 0; tile < totalTilesThatWillBeUsed; tile++)
        {
            auto workersUsedOnThisTile = numWorkersUsedOnTile(tile);
            const auto v = graph.addVertex(
                partialReduxCs,
                "ReduceSum",
                {
                    {"partialSums", tensors["partialSumsPerWorker"].slice(
                                        {idx}, {idx + workersUsedOnThisTile})},
                    {"sum", tensors["partialSumsPerTile"][tile]},
                });
            graph.setInitialValue(v["size"], unsigned(workersUsedOnThisTile));

            graph.setCycleEstimate(v, workersUsedOnThisTile);
            graph.setTileMapping(v, tile);
            idx += workersUsedOnThisTile;
        }

        ComputeSet finalReduxCs = graph.addComputeSet("finalReduction");
        const auto v = graph.addVertex(
            finalReduxCs,
            "ReduceSum",
            {{"partialSums", tensors["partialSumsPerTile"]},
             {"sum", tensors["sum"]}});
        graph.setInitialValue(v["size"], unsigned(totalTilesThatWillBeUsed));

        graph.setCycleEstimate(v, numTiles);
        graph.setTileMapping(v, numTiles - 1);

        return Sequence(
            Execute(dotCs),
            Execute(partialReduxCs),
            Execute(finalReduxCs));
    }

    void createAndLayOutTensors(poplar::Type type, Graph &graph)
    {
        tensors["initA"] = graph.addVariable(FLOAT, {}, "initA");
        tensors["initB"] = graph.addVariable(FLOAT, {}, "initB");
        tensors["initC"] = graph.addVariable(FLOAT, {}, "initC");

        tensors["a"] = graph.addVariable(type, {arraySize}, "a");
        tensors["b"] = graph.addVariable(type, {arraySize}, "b");
        tensors["c"] = graph.addVariable(type, {arraySize}, "c");
        tensors["sum"] = graph.addVariable(FLOAT, {}, "sum");
        tensors["partialSumsPerWorker"] = graph.addVariable(FLOAT, {totalWorkersThatWillBeUsed},
                                                            "partialSumsPerWorker");
        tensors["partialSumsPerTile"] = graph.addVariable(FLOAT, {totalTilesThatWillBeUsed}, "partialSumsPerTile");
        graph.createHostRead("sum", tensors["sum"]);
        graph.createHostWrite("initA", tensors["initA"]);
        graph.createHostWrite("initB", tensors["initB"]);
        graph.createHostWrite("initC", tensors["initC"]);

        auto idx = 0u;
        for (auto tile = 0u; tile < totalTilesThatWillBeUsed; tile++)
        {
            auto mapMem = [=]() -> unsigned {
#ifdef MEM_ON_NEXT_TILE
                return (tile + 1) % totalTilesThatWillBeUsed;
#else
                return tile;
#endif
            };

            if (auto numItems = numItemsForTile(tile); numItems > 0)
            {
                graph.setTileMapping(tensors["a"].slice(idx, idx + numItems), mapMem());
                graph.setTileMapping(tensors["b"].slice(idx, idx + numItems), mapMem());
                graph.setTileMapping(tensors["c"].slice(idx, idx + numItems), mapMem());
                idx += numItems;
            }

            graph.setTileMapping(tensors["partialSumsPerWorker"].slice(
                                     {tile * numWorkersPerTile}, {tile * numWorkersPerTile + numWorkersUsedOnTile(tile)}),
                                 mapMem());
            graph.setTileMapping(tensors["partialSumsPerTile"].slice(tile, tile + 1), mapMem());
        }
        graph.setTileMapping(tensors["sum"], numTiles - 1);
        graph.setTileMapping(tensors["initA"], 0);
        graph.setTileMapping(tensors["initB"], 0);
        graph.setTileMapping(tensors["initC"], 0);
    }

    void createDataStreams(poplar::Type type, Graph &graph)
    {
        // dataStreams["in_a"] = graph.addHostToDeviceFIFO("in_a", type, arraySize);
        // dataStreams["in_b"] = graph.addHostToDeviceFIFO("in_b", type, arraySize);
        // dataStreams["in_c"] = graph.addHostToDeviceFIFO("in_c", type, arraySize);

        dataStreams["out_a"] = graph.addDeviceToHostFIFO("out_a", type, arraySize);
        dataStreams["out_b"] = graph.addDeviceToHostFIFO("out_b", type, arraySize);
        dataStreams["out_c"] = graph.addDeviceToHostFIFO("out_c", type, arraySize);
    }

    [[nodiscard]] const unsigned numWorkersNeeded(unsigned arraySize, unsigned numTiles, unsigned numWorkersPerTile) const
    {
        return std::min(
            arraySize,
            std::min(
                numTiles * numWorkersPerTile,
                unsigned(std::ceil(arraySize / (numWorkersPerTile * 1.0))) * 6));
    }

    [[nodiscard]] const unsigned numTilesNeeded(unsigned arraySize, unsigned numTiles, unsigned numWorkersPerTile) const
    {
        return std::min(numTiles, unsigned(std::ceil(arraySize / (numWorkersPerTile * 1.0))));
    }

public:
    PoplarStreamUtil(unsigned numTiles, unsigned numWorkersPerTile, unsigned arraySize) : numTiles(numTiles),
                                                                                          numWorkersPerTile(numWorkersPerTile),
                                                                                          arraySize(arraySize),
                                                                                          totalTilesThatWillBeUsed(numTilesNeeded(arraySize, numTiles, numWorkersPerTile)),
                                                                                          totalWorkersThatWillBeUsed(numWorkersNeeded(arraySize, numTiles, numWorkersPerTile))
    {
    }

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

    void buildComputeGraph(poplar::Type type, Graph &graph)
    {
        // Set up data streams to copy data in and out of graph

        auto typeStr1 = type == FLOAT ? "<float," : "<half,";
#ifdef VECTORISED
        auto typeStr2 = type == FLOAT ? "float2>" : "half4>";
#else
        auto typeStr2 = type == FLOAT ? "float>" : "half>";
#endif

        auto typeStr = std::string(typeStr1) + std::string(typeStr2);

        createDataStreams(type, graph);

        createAndLayOutTensors(type, graph);

        auto InitProg = initProgram(typeStr, graph);

        auto CopyProg = copyProgram(typeStr, graph);
        auto MulProg = mulProgram(typeStr, graph);
        auto AddProg = addProgram(typeStr, graph);
        auto TriadProg = triadProgram(typeStr, graph);
        auto DotProg = dotProdProgram(typeStr, graph);

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
    target = device->getTarget();

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

    auto util = PoplarStreamUtil(numTiles, numWorkers, arraySize);

    graph.addCodelets("PoplarKernels.cpp", CodeletFileType::Auto, "-O3");

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

    util.buildComputeGraph(halfPrecision ? HALF : FLOAT, graph);
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



template<> void PoplarStream<double>::copyArrays( const double *src, double *dst) {
    std::memcpy(dst, src, arraySize * sizeof(double));
}

template<> void PoplarStream<float>::copyArrays(const float *src, float *dst) {
    copyDeviceHalfToFloat(target, src, dst, arraySize);
}



template <class T>
void PoplarStream<T>::read_arrays(std::vector<T> &h_a, std::vector<T> &h_b, std::vector<T> &h_c)
{

    engine->run(STREAM_BACK_TO_HOST_PROGRAM);
    copyArrays(a.get(), h_a.data());
    copyArrays(b.get(), h_b.data());
    copyArrays(c.get(), h_c.data());


#ifdef DEBUG
    captureProfileInfo(*engine);
    engine->printProfileSummary(std::cout,
                                OptionFlags{{"showExecutionSteps", "true"}});
#endif
}

template class PoplarStream<float>;

template class PoplarStream<double>; // Not usable, but needs to exist for stream.cpp
