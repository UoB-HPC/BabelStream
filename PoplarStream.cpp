
// Copyright (c) 2015-16 Thorben Louw, Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "PoplarStream.h"
#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplin/codelets.hpp>
#include <math.h>

using namespace poplar;
using namespace poplar::program;


enum Programs {
    STREAM_TO_DEVICE_PROGRAM,
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
        {"debug.instrument", "false"}
};


std::optional <Device> getIpuDevice(const unsigned deviceType = 0) {
    if (deviceType == 0) { // Target an IPUDevice
        DeviceManager manager = DeviceManager::createDeviceManager();
        Device device;
        bool success = false;
        for (auto &hwDevice : manager.getDevices(poplar::TargetType::IPU, 1)) {
            device = std::move(hwDevice);
            if ((success = device.attach())) {
                std::cout << "Attached to IPU " << device.getId() << std::endl;
                return std::optional<Device>(std::move(device));
            }
        }
    } else if (deviceType == 1) { // Use the CPUDevice
        // Note that as of Poplar v1.1.11, this ony returns a useless device with 1 tile and 256Kb of memory!
        return std::optional<Device>(Device::createCPUDevice());

    }
    return std::nullopt;
}


void listDevices(void) {    
    std::cout << 0 << ": " << "CPUDevice" << std::endl;

    // For now we just support the IPU device rather than the CPU device
    // Create the DeviceManager which is used to discover devices
    DeviceManager manager = DeviceManager::createDeviceManager();

    // Attempt to attach to a single IPU:
    Device device;
    if (auto devices = manager.getDevices(poplar::TargetType::IPU, 1); !devices.empty()) {
        std::cout << 1 << ": " << "CPUDevice" << std::endl;
    }
}

class PoplarStreamUtil {
private:
    const unsigned numTiles;
    const unsigned numWorkersPerTile;
    const unsigned arraySize;

    const unsigned totalTilesThatWillBeUsed;
    const unsigned totalWorkersThatWillBeUsed;

    std::map <std::string, Tensor> tensors = {};
    std::vector <Program> programs = {};
    std::map <std::string, DataStream> dataStreams = {};

    unsigned numItemsForTileAndWorker(const unsigned tileNum, const unsigned workerNum) const {
        if (arraySize <= numTiles) { // Just use 1 item per worker
            return 1;
        } else { // Balance as fairly as possible
            auto extra =  tileNum < arraySize % numTiles;
            auto totalForTile = unsigned(arraySize / numTiles) + extra;
            auto extraForThread = workerNum < totalForTile % numWorkersPerTile;
            return unsigned(totalForTile / numWorkersPerTile) + extraForThread;
        }
    }

    unsigned numItemsForTile(const unsigned tileNum) const {
        if (arraySize <= numTiles) { // We'll just use one item per worker
            return numWorkersUsedOnTile(tileNum);
        } else { // Now we must balance as fairly as possible
            auto extra = tileNum < arraySize % numTiles;
            return unsigned(arraySize / numTiles) + extra;
        }
    }

    unsigned numWorkersUsedOnTile(const unsigned tileNum) const {
        return std::min(numWorkersPerTile, totalWorkersThatWillBeUsed - tileNum * numWorkersPerTile); 
    }

    Program copyProgram(Graph &graph) {
        ComputeSet cs = graph.addComputeSet("copy");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++) {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            //std::cout <<  tile  << " " << worker << " " << numItems << std::endl;
            const auto v = graph.addVertex(cs,
                                           "CopyKernel",
                                           {
                                                   {"a", tensors["a"].slice({idx}, {idx + numItems}).flatten()},
                                                   {"c", tensors["c"].slice({idx}, {idx + numItems}).flatten()}
                                           });
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }

    Program mulProgram(Graph &graph) {
        ComputeSet cs = graph.addComputeSet("mul");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++) {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "MulKernel",
                {
                        {"b",     tensors["b"].slice({idx}, {idx + numItems})},
                        {"c",     tensors["c"].slice({idx}, {idx + numItems})},
                        {"alpha", tensors["alpha"]},
                }
            );
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }


    Program addProgram(Graph &graph) {
        ComputeSet cs = graph.addComputeSet("add");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++) {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "AddKernel",
                {
                        {"b", tensors["b"].slice({idx}, {idx + numItems})},
                        {"a", tensors["a"].slice({idx}, {idx + numItems})},
                        {"c", tensors["c"].slice({idx}, {idx + numItems})},

                }
            );
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        return Execute(cs);
    }


    Program triadProgram(Graph &graph) {
        ComputeSet cs = graph.addComputeSet("triad");

        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++) {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;

            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                cs,
                "TriadKernel",
                {
                        {"b",     tensors["b"].slice({idx}, {idx + numItems})},
                        {"a",     tensors["a"].slice({idx}, {idx + numItems})},
                        {"c",     tensors["c"].slice({idx}, {idx + numItems})},
                        {"alpha", tensors["alpha"]},

                }
            );
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;

        }
        
        return Execute(cs);
    }


    Program dotProdProgram(Graph &graph) {
        ComputeSet dotCs = graph.addComputeSet("dot");
        unsigned idx = 0u;
        for (unsigned w = 0; w < totalWorkersThatWillBeUsed; w++) {
            auto tile = w / numWorkersPerTile;
            auto worker = w % numWorkersPerTile;
            auto numItems = numItemsForTileAndWorker(tile, worker);
            const auto v = graph.addVertex(
                dotCs,
                "DotProdKernel",
                {
                        {"b",   tensors["b"].slice({idx}, {idx + numItems})},
                        {"a",   tensors["a"].slice({idx}, {idx + numItems})},
                        {"sum", tensors["partialSumsPerWorker"][tile * numWorkersPerTile +
                                                                worker]},
                }
            );
            graph.setCycleEstimate(v, numItems);
            graph.setTileMapping(v, tile);
            idx += numItems;
        }

        ComputeSet partialReduxCs = graph.addComputeSet("reductionPerTile");
        idx = 0u;
        for (unsigned tile = 0; tile < totalTilesThatWillBeUsed; tile++) {
            auto workersUsedOnThisTile = numWorkersUsedOnTile(tile);
            const auto v = graph.addVertex(
                partialReduxCs,
                "ReduceSum",
                {
                        {"partialSums", tensors["partialSumsPerWorker"].slice(
                            {idx}, {idx + workersUsedOnThisTile})},
                        {"sum",         tensors["partialSumsPerTile"][tile]},
                }
            );
            graph.setCycleEstimate(v, workersUsedOnThisTile);
            graph.setTileMapping(v, tile);
            idx += workersUsedOnThisTile;
        }

        ComputeSet finalReduxCs = graph.addComputeSet("finalReduction");
        const auto v = graph.addVertex(
            finalReduxCs,
            "ReduceSum",
            {
                    {"partialSums", tensors["partialSumsPerTile"]},
                    {"sum",         tensors["sum"]},
            });
        graph.setCycleEstimate(v, numTiles);
        graph.setTileMapping(v, numTiles - 1);

        return Sequence(
            Execute(dotCs), 
            Execute(partialReduxCs),
            Execute(finalReduxCs)
        );
    }

    void createAndLayOutTensors(poplar::Type type, Graph &graph) {
        tensors["a"] = graph.addVariable(type, {arraySize}, "a");
        tensors["b"] = graph.addVariable(type, {arraySize}, "b");
        tensors["c"] = graph.addVariable(type, {arraySize}, "c");
        tensors["sum"] = graph.addVariable(type, {}, "sum");
        tensors["partialSumsPerWorker"] = graph.addVariable(type, {totalWorkersThatWillBeUsed},
                                                            "partialSumsPerWorker");
        tensors["partialSumsPerTile"] = graph.addVariable(type, {totalTilesThatWillBeUsed}, "partialSumsPerTile");
        tensors["alpha"] = graph.addConstant(type, {}, startScalar, "alpha");
        graph.createHostRead("sum", tensors["sum"]);

        
            auto idx = 0u;
            for (auto tile = 0u; tile < totalTilesThatWillBeUsed; tile++) {
                if (auto numItems = numItemsForTile(tile); numItems > 0) {
                    graph.setTileMapping(tensors["a"].slice(idx, idx + numItems), tile);
                    graph.setTileMapping(tensors["b"].slice(idx, idx + numItems), tile);
                    graph.setTileMapping(tensors["c"].slice(idx, idx + numItems), tile);
                    idx += numItems;

                }

                graph.setTileMapping(tensors["partialSumsPerWorker"].slice(
                        {tile * numWorkersPerTile}, {tile * numWorkersPerTile + numWorkersUsedOnTile(tile)}),
                                     tile);
                graph.setTileMapping(tensors["partialSumsPerTile"].slice(tile, tile + 1), tile);

            }
            graph.setTileMapping(tensors["sum"], numTiles - 1);
            graph.setTileMapping(tensors["alpha"], 0);
    }

    void createDataStreams(poplar::Type type, Graph &graph) {
        dataStreams["in_a"] = graph.addHostToDeviceFIFO("in_a", type, arraySize);
        dataStreams["in_b"] = graph.addHostToDeviceFIFO("in_b", type, arraySize);
        dataStreams["in_c"] = graph.addHostToDeviceFIFO("in_c", type, arraySize);

        dataStreams["out_a"] = graph.addDeviceToHostFIFO("out_a", type, arraySize);
        dataStreams["out_b"] = graph.addDeviceToHostFIFO("out_b", type, arraySize);
        dataStreams["out_c"] = graph.addDeviceToHostFIFO("out_c", type, arraySize);
    }

    constexpr unsigned numWorkersNeeded(unsigned arraySize, unsigned numTiles, unsigned numWorkersPerTile) {
        return std::min(
            arraySize, 
            std::min(
                numTiles * numWorkersPerTile, 
                unsigned(std::ceil(arraySize / (numWorkersPerTile * 1.0))) * 6
            )
        );
    }

    constexpr unsigned numTilesNeeded(unsigned arraySize, unsigned numTiles, unsigned numWorkersPerTile) {
        return std::min(numTiles, unsigned(std::ceil(arraySize / (numWorkersPerTile * 1.0))));
    }

public:
    PoplarStreamUtil(unsigned numTiles, unsigned numWorkersPerTile, unsigned arraySize) :
            numTiles(numTiles),
            numWorkersPerTile(numWorkersPerTile),
            arraySize(arraySize),
            totalTilesThatWillBeUsed(numTilesNeeded(arraySize, numTiles, numWorkersPerTile)),
            totalWorkersThatWillBeUsed(numWorkersNeeded(arraySize, numTiles, numWorkersPerTile)) {
    }

    template<typename T>
    std::unique_ptr <Engine> prepareEngine(const Device &device,
                                           const Graph &graph,
                                           void *a,
                                           void *b,
                                           void *c);


    void buildComputeGraph(poplar::Type type, Graph &graph) {
        // Set up data streams to copy data in and out of graph
        createDataStreams(type, graph);


        createAndLayOutTensors(type, graph);


        auto StreamToDeviceProg = Sequence(Copy(dataStreams["in_a"], tensors["a"]),
                                           Copy(dataStreams["in_b"], tensors["b"]),
                                           Copy(dataStreams["in_c"], tensors["c"]));

        auto CopyProg = copyProgram(graph);
        auto MulProg = mulProgram(graph);
        auto AddProg = addProgram(graph);
        auto TriadProg = triadProgram(graph);
        auto DotProg = dotProdProgram(graph);

        auto StreamToHostProg = Sequence(Copy(tensors["a"], dataStreams["out_a"]),
                                         Copy(tensors["b"], dataStreams["out_b"]),
                                         Copy(tensors["c"], dataStreams["out_c"]));

        programs = {StreamToDeviceProg, CopyProg, MulProg, AddProg, TriadProg, DotProg,
                    StreamToHostProg};


    }


};

template<typename T>
std::unique_ptr <Engine> PoplarStreamUtil::prepareEngine(const Device &device,
                                                         const Graph &graph,
                                                         void *a,
                                                         void *b,
                                                         void *c) {
    assert(programs.size() > 0);
    auto engine = std::make_unique<Engine>(graph, programs, POPLAR_ENGINE_OPTIONS);


    engine->connectStream("in_a", a);
    engine->connectStream("in_b", b);
    engine->connectStream("in_c", c);

    engine->connectStream("out_a", a);
    engine->connectStream("out_b", b);
    engine->connectStream("out_c", c);


    engine->load(device);

    return std::move(engine);
}


template<class T>
PoplarStream<T>::PoplarStream(const unsigned int arraySize, const int device_num) {
    this->arraySize = arraySize;
    this->a = std::unique_ptr<T[]>(new T[arraySize]());
    this->b = std::unique_ptr<T[]>(new T[arraySize]());
    this->c = std::unique_ptr<T[]>(new T[arraySize]());

    auto device = getIpuDevice(device_num);

    if (!device.has_value()) {
        throw std::runtime_error("Could not allocate IPU device");
    }

    Graph graph(device.value());
    const auto numTiles = graph.getTarget().getNumTiles();
    const auto numWorkers = graph.getTarget().getNumWorkerContexts();
    const auto maxBytesPerTile = graph.getTarget().getBytesPerTile();
    const auto clockFrequency = graph.getTarget().getTileClockFrequency();

    std::cout << "Using IPU with " << numTiles << " tiles, each with " << numWorkers <<
              " workers and " << maxBytesPerTile / 1024 << "KB of memory per tile, and clock frequency " <<
              (int) clockFrequency / 1000 / 1000 << " MHz" << std::endl;


    auto util = PoplarStreamUtil(numTiles, numWorkers, arraySize);

    graph.addCodelets("PoplarKernels.cpp");

    if (sizeof(T) > sizeof(float)) {
        throw std::runtime_error("Device does not support double precision, please use --float");
    }

    // Check buffers fit on the device
    unsigned long maxbuffer = numTiles * maxBytesPerTile;
    unsigned long totalmem = numTiles * maxBytesPerTile;
    if (maxbuffer < sizeof(T) * arraySize)
        throw std::runtime_error("Device cannot allocate a buffer big enough");
    if (totalmem < 3 * sizeof(T) * arraySize)
        throw std::runtime_error("Device does not have enough memory for all 3 buffers");


    util.buildComputeGraph(FLOAT, graph);
    engine = util.prepareEngine<float>(device.value(), graph, a.get(), b.get(), c.get());
}


template<class T>
PoplarStream<T>::~PoplarStream() {
}

template<class T>
void PoplarStream<T>::copy() {
    engine->run(COPY_PROGRAM);
}

template<class T>
void PoplarStream<T>::mul() {
    engine->run(MUL_PROGRAM);
}

template<class T>
void PoplarStream<T>::add() {
    engine->run(ADD_PROGRAM);
}

template<class T>
void PoplarStream<T>::triad() {
    engine->run(TRIAD_PROGRAM);
}

template<class T>
T PoplarStream<T>::dot() {
    engine->run(DOT_PROGRAM);
    engine->readTensor("sum", &sum);
    return sum;
}

template<class T>
void PoplarStream<T>::init_arrays(T initA, T initB, T initC) {
    for (unsigned i = 0; i < this->arraySize; i++) {
        a[i] = initA;
        b[i] = initB;
        c[i] = initC;
    }

    engine->run(STREAM_TO_DEVICE_PROGRAM);
}


template<class T>
void PoplarStream<T>::read_arrays(std::vector <T> &h_a, std::vector <T> &h_b, std::vector <T> &h_c) {

    engine->run(STREAM_BACK_TO_HOST_PROGRAM);


    for (unsigned i = 0; i < arraySize; i++) {
        h_a[i] = a[i];
        h_b[i] = b[i];
        h_c[i] = c[i];

    }

    //engine->printProfileSummary(std::cout,
    //                          OptionFlags{{"showExecutionSteps", "false"}});
}


template
class PoplarStream<float>;

template
class PoplarStream<double>;


