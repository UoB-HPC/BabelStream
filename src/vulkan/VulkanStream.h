// Copyright (c) 2026 Stream HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

#include "Stream.h"

#define IMPLEMENTATION_STRING "Vulkan"

// State that we need to create to list devices
struct VulkanDevices {
    vk::raii::Context context;
    vk::raii::Instance instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT debugMessenger{nullptr};
    std::vector<vk::raii::PhysicalDevice> physicalDevices;

    VulkanDevices();
};

// Wrapper for Vulkan buffer objects and underlying memory
struct Buffer {
    vk::raii::Buffer buffer{nullptr};
    vk::raii::DeviceMemory memory{nullptr};
    vk::DeviceSize size{0};
    void* mapped{nullptr};
    vk::AccessFlagBits state{0};
};

// Wrapper for the Vulkan compute pipeline for each operation
struct Pipelines {
    vk::raii::Pipeline init{nullptr};
    vk::raii::Pipeline copy{nullptr};
    vk::raii::Pipeline mul{nullptr};
    vk::raii::Pipeline add{nullptr};
    vk::raii::Pipeline triad{nullptr};
    vk::raii::Pipeline nstream{nullptr};
    vk::raii::Pipeline dot{nullptr};
};

template <class T>
class VulkanStream : public Stream<T>
{
  public:
    // Interface constructor and destructor
    VulkanStream(BenchId bs, intptr_t array_size, int device_id, T initA, T initB,
        T initC);

    // Interface functions for running operations on device
    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    // Interface function for initializing arrays on device
    void init_arrays(T initA, T initB, T initC) override;
    // Interface function for reading results back from device
    void get_arrays(T const*& a, T const*& b, T const*& c) override;

  private:
    // Members initialized from constructor
    int _deviceIndex;
    intptr_t _arraySize;

    // Members tracking the execution grid to use for compute kernel dispatch
    uint32_t _workgroupSize{128};
    uint32_t _numDispatchGroups{1};
    uint32_t _dotNumDispatchGroups{1};

    // Constant for the 4 descriptors we have for in the shader module
    static constexpr unsigned _descriptorCount = 4;

    // Order of member declarations is important as we need to free the objects
    // created from the logical device before the logical device itself. C++
    // default destruction destroys elements in the reverse order they were
    // created.

    // Holds Vulkan instance and enumerated physical devices
    VulkanDevices _devices;

    // RAII Vulkan objects owned by the stream
    vk::raii::Device _device{nullptr};
    vk::raii::Queue _queue{nullptr};
    vk::raii::CommandPool _commandPool{nullptr};
    vk::raii::CommandBuffer _commandBuffer{nullptr};
    vk::raii::DescriptorSetLayout _descriptorSetLayout{nullptr};
    vk::raii::PipelineLayout _pipelineLayout{nullptr};
    vk::raii::DescriptorPool _descriptorPool{nullptr};
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    // Wrapper for all the RAII pipeline objects
    Pipelines _pipelines;

    // Wrappers for all the RAII buffer and memory objects
    Buffer _bufferA;
    Buffer _bufferB;
    Buffer _bufferC;
    Buffer _bufferPartial;
    Buffer _bufferStaging;
    Buffer _bufferPartialStaging;

    // Host side vectors needed to return a valid pointer from `get_arrays()`
    std::vector<T> _hostA;
    std::vector<T> _hostB;
    std::vector<T> _hostC;

    // Helper function for creating vulkan compute pipelines for each device
    // operation using the appropriate SPIR-V shader
    void createPipelines();
    vk::raii::Pipeline createPipeline(const char* entry);

    // Helper functions for create Vulkan buffers backend by memory
    void createBuffers();
    Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        bool map = false) const;
    uint32_t findMemoryType(uint32_t typeBits,
        vk::MemoryPropertyFlags properties) const;

    // Helper function for writing descriptor sets
    void writeDescriptorSets();

    // Helper function for resetting and beginning recording of the shared command
    // buffer
    vk::raii::CommandBuffer& beginCommands();

    // Helper function for launching a compute pipeline and waiting for it to
    // complete
    void dispatch(const vk::Pipeline& pipeline,
        const std::vector<vk::BufferMemoryBarrier>& barriers,
        const std::array<T, 3>* pushConstants = nullptr);

    // Helper function for copying device memory back to host
    void copyDeviceToHost(const Buffer& src, std::vector<T>& dst);
};

// Interface free standing function for querying the available devices
void listDevices(void);
