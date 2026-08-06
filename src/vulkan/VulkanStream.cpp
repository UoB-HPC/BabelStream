// Copyright (c) 2026 Stream HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "VulkanStream.h"

#include "double/addShader.spv.h"
#include "double/copyShader.spv.h"
#include "double/dotShader.spv.h"
#include "double/initShader.spv.h"
#include "double/mulShader.spv.h"
#include "double/nstreamShader.spv.h"
#include "double/triadShader.spv.h"
#include "float/addShader.spv.h"
#include "float/copyShader.spv.h"
#include "float/dotShader.spv.h"
#include "float/initShader.spv.h"
#include "float/mulShader.spv.h"
#include "float/nstreamShader.spv.h"
#include "float/triadShader.spv.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
std::string getDeviceDriver(const vk::raii::PhysicalDevice &physicalDevice) {
  auto properties2 =
      physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                    vk::PhysicalDeviceDriverProperties>();
  vk::PhysicalDeviceDriverProperties const &driverProperties =
      properties2.get<vk::PhysicalDeviceDriverProperties>();
  std::string driver(std::string(driverProperties.driverName));
  driver.append(", ");
  driver.append(std::string(driverProperties.driverInfo));
  return driver;
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  std::string callback_msg(pCallbackData->pMessage);
  if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
      severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "Babelstream validation layer: type" << to_string(type)
              << " message: " << callback_msg << std::endl;
  }

  return vk::False;
}

uint32_t selectQueueFamily(const vk::PhysicalDevice &device) {
  const auto families = device.getQueueFamilyProperties();

  // Prefer dedicated compute queues if possible
  for (uint32_t i = 0; i < families.size(); ++i) {
    const auto flags = families[i].queueFlags;
    if ((flags & vk::QueueFlagBits::eCompute) &&
        !(flags & vk::QueueFlagBits::eGraphics)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < families.size(); ++i) {
    const auto flags = families[i].queueFlags;
    if (flags & vk::QueueFlagBits::eCompute) {
      return i;
    }
  }

  throw std::runtime_error("No compute-capable Vulkan queue family found");
}

void enforceMemDep(Buffer &buffer,
                   std::vector<vk::BufferMemoryBarrier> &barriers,
                   vk::AccessFlagBits dstFlags) {
  // Enforces a memory dependency on a buffer when a previous command has
  // written to it, i.e write-after-write or read-after-write
  if ((buffer.state & vk::AccessFlagBits::eShaderWrite) !=
      vk::AccessFlagBits::eNone) {
    barriers.push_back(vk::BufferMemoryBarrier(
        buffer.state, dstFlags, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, buffer.buffer, 0, VK_WHOLE_SIZE));
  }
  buffer.state = dstFlags;
}

} // anonymous namespace

VulkanDevices::VulkanDevices() {
  vk::ApplicationInfo appInfo;
  appInfo.pApplicationName = "BabelStream";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "BabelStream";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = vk::ApiVersion14;

  // Enable validation layer in debug builds
  std::vector<char const *> requiredLayers;
  std::vector<const char *> enabledExtensions;
  auto extProperties = context.enumerateInstanceExtensionProperties();
#ifndef NDEBUG
  requiredLayers.push_back("VK_LAYER_KHRONOS_validation");
  enabledExtensions.push_back(vk::EXTDebugUtilsExtensionName);

  for (auto const &ext : enabledExtensions) {
    bool extUnsupported =
        std::none_of(std::begin(extProperties), std::end(extProperties),
                     [ext](auto const &extProperty) {
                       return strcmp(extProperty.extensionName, ext) == 0;
                     });
    if (extUnsupported) {
      throw std::runtime_error("Required Vulkan extension not supported");
    }
  }

  auto layerProperties = context.enumerateInstanceLayerProperties();
  for (auto const &requiredLayer : requiredLayers) {
    bool layerUnsupported = std::none_of(
        std::begin(layerProperties), std::end(layerProperties),
        [requiredLayer](auto const &layerProperty) {
          return strcmp(layerProperty.layerName, requiredLayer) == 0;
        });
    if (layerUnsupported) {
      throw std::runtime_error("Required Vulkan layer not supported");
    }
  }
#endif

  // To layer ontop of MoltenVK we need to opt-in to the Vulkan loader showing
  // non-conformant Vulkan implementations
  bool portabilityExtSupported = std::any_of(
      std::begin(extProperties), std::end(extProperties),
      [](auto const &extProperty) {
        return strcmp(extProperty.extensionName,
                      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
      });

  vk::InstanceCreateFlags flags{};
  if (portabilityExtSupported) {
    enabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  }

  vk::InstanceCreateInfo createInfo{
      flags,
      &appInfo,
      static_cast<uint32_t>(requiredLayers.size()),
      requiredLayers.data(),
      static_cast<uint32_t>(enabledExtensions.size()),
      enabledExtensions.data(),
  };

#ifndef NDEBUG
  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{
      {}, severityFlags, messageTypeFlags, &debugCallback};

  createInfo.pNext = &debugUtilsMessengerCreateInfo;
  instance = vk::raii::Instance(context, createInfo);
  debugMessenger =
      instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfo);
#else
  instance = vk::raii::Instance(context, createInfo);
#endif

  physicalDevices = instance.enumeratePhysicalDevices();
}

VulkanDevices::~VulkanDevices() {
  physicalDevices.clear();
  debugMessenger = nullptr;
  instance = nullptr;
}

template <class T>
VulkanStream<T>::VulkanStream(BenchId, const intptr_t array_size,
                              const int device_index, T initA, T initB, T initC)
    : _deviceIndex(device_index), _arraySize(array_size), _initA(initA),
      _initB(initB), _initC(initC) {
  const auto &physicalDevices = _devices.physicalDevices;
  if (physicalDevices.empty()) {
    throw std::runtime_error("No Vulkan physical devices found");
  }
  if (device_index < 0 ||
      static_cast<size_t>(device_index) >= physicalDevices.size()) {
    throw std::runtime_error("Invalid device index");
  }

  const vk::raii::PhysicalDevice &physicalDevice =
      physicalDevices[device_index];
  auto supportedFeatures =
      physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2,
                                  vk::PhysicalDeviceMaintenance4Features>();

  auto const &features =
      supportedFeatures.get<vk::PhysicalDeviceFeatures2>().features;
  if (std::is_same_v<T, double> && !features.shaderFloat64) {
    throw std::runtime_error(
        "Device does not support double precision, please use --float");
  }

  auto const &maint4Features =
      supportedFeatures.get<vk::PhysicalDeviceMaintenance4Features>();
  if (!maint4Features.maintenance4) {
    throw std::runtime_error(
        "Device does not support LocalSizeId execution mode in SPIR-V");
  }

  const auto limits = physicalDevice.getProperties().limits;
  const uint64_t maxStorageBufferElems =
      limits.maxStorageBufferRange / sizeof(T);
  if (_arraySize > maxStorageBufferElems) {
    throw std::runtime_error(
        "Vulkan device cannot expose a large enough storage buffer");
  }

  _workgroupSize = std::min(limits.maxComputeWorkGroupInvocations,
                            limits.maxComputeWorkGroupSize[0]);

  _numDispatchGroups = (_arraySize + _workgroupSize - 1) / _workgroupSize;
  if (_numDispatchGroups > limits.maxComputeWorkGroupCount[0]) {
    // We could workaround this threshold in future if we did multiple
    // submissions  of the execution grid.
    throw std::runtime_error(
        "Vulkan device cannot expose a large enough execution grid.");
  }

  // Vulkan does not provide a way to query the parallel unit count like
  // `CL_DEVICE_MAX_COMPUTE_UNITS` int OpenCL or `multiProcessorCount` in CUDA.
  // We therefore hardcode a workgroup count for DOT that enables enough work to
  // be done per thread to take advantage of the strided access loops.
  _dotNumDispatchGroups = std::min(64u, _numDispatchGroups);

  std::cout << "Using Vulkan device "
            << physicalDevice.getProperties().deviceName << std::endl;
  std::cout << "Driver: " << getDeviceDriver(physicalDevice) << std::endl;
  std::cout << "Kernel config: " << _numDispatchGroups << " groups of size "
            << _workgroupSize << std::endl;
  std::cout << "Dot config: " << _dotNumDispatchGroups << " groups of size "
            << _workgroupSize << std::endl;

  uint32_t queueFamilyIndex = selectQueueFamily(physicalDevice);
  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo queueInfo{{}, queueFamilyIndex, 1, &queuePriority};

  vk::DeviceCreateInfo deviceInfo({}, queueInfo);

  vk::PhysicalDeviceFeatures enabledFeatures{};
  enabledFeatures.shaderFloat64 = std::is_same_v<T, double>;

  vk::PhysicalDeviceMaintenance4Features enabledMaint4Features{};
  enabledMaint4Features.maintenance4 = true;

  auto physicalDeviceExtensions =
      physicalDevice.enumerateDeviceExtensionProperties();
  bool portabilityExtSupported = std::any_of(
      std::begin(physicalDeviceExtensions), std::end(physicalDeviceExtensions),
      [](auto const &extProperty) {
        return strcmp(extProperty.extensionName,
                      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0;
      });
  if (portabilityExtSupported) {
    const char *enabledExtension = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
    deviceInfo.setEnabledExtensionCount(1);
    deviceInfo.setPpEnabledExtensionNames(&enabledExtension);
  }

  vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceMaintenance4Features>
      dev_create_info_chain(deviceInfo, enabledFeatures, enabledMaint4Features);

  _device = vk::raii::Device(physicalDevices[device_index],
                             dev_create_info_chain.get<vk::DeviceCreateInfo>());
  _queue = vk::raii::Queue(_device, queueFamilyIndex, 0);

  vk::CommandPoolCreateInfo poolInfo{
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndex};
  _commandPool = vk::raii::CommandPool(_device, poolInfo);

  std::array<vk::DescriptorSetLayoutBinding, _descriptorCount> bindings{};
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
  }

  vk::DescriptorSetLayoutCreateInfo setLayoutInfo;
  setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  setLayoutInfo.pBindings = bindings.data();

  _descriptorSetLayout = vk::raii::DescriptorSetLayout(_device, setLayoutInfo);

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, *_descriptorSetLayout);
  _pipelineLayout = vk::raii::PipelineLayout(_device, pipelineLayoutInfo);

  vk::DescriptorPoolSize poolSize;
  poolSize.type = vk::DescriptorType::eStorageBuffer;
  poolSize.descriptorCount = _descriptorCount;

  vk::DescriptorPoolCreateInfo poolCreateInfo;
  poolCreateInfo.maxSets = 1;
  poolCreateInfo.poolSizeCount = 1;
  poolCreateInfo.pPoolSizes = &poolSize;
  poolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  _descriptorPool = vk::raii::DescriptorPool(_device, poolCreateInfo);

  vk::DescriptorSetAllocateInfo allocInfo(_descriptorPool,
                                          *_descriptorSetLayout);
  descriptorSets = _device.allocateDescriptorSets(allocInfo);

  createBuffers();
  writeDescriptorSets();
  createPipelines();
}

template <class T> VulkanStream<T>::~VulkanStream() {
  // Order of destruction is important we need to free the objects
  // created from the logical device before the logical device itself
  _bufferA.clear();
  _bufferB.clear();
  _bufferC.clear();
  _bufferPartial.clear();
  _bufferStaging.clear();
  _bufferPartialStaging.clear();
  _pipelines.clear();
  _queue = nullptr;
  _commandPool = nullptr;
  _descriptorSetLayout = nullptr;
  descriptorSets.clear();
  _pipelineLayout = nullptr;
  _descriptorPool = nullptr;
  _device = nullptr;
}

template <class T> void VulkanStream<T>::createBuffers() {
  const vk::DeviceSize arrayBytes =
      sizeof(T) * static_cast<vk::DeviceSize>(_arraySize);
  const vk::DeviceSize partialBytes =
      sizeof(T) * static_cast<vk::DeviceSize>(_dotNumDispatchGroups);

  const auto deviceUsage = vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eTransferSrc |
                           vk::BufferUsageFlagBits::eTransferDst;

  _bufferA = createBuffer(arrayBytes, deviceUsage,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
  _bufferB = createBuffer(arrayBytes, deviceUsage,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
  _bufferC = createBuffer(arrayBytes, deviceUsage,
                          vk::MemoryPropertyFlagBits::eDeviceLocal);
  _bufferPartial = createBuffer(partialBytes,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                vk::MemoryPropertyFlagBits::eDeviceLocal);

  _bufferStaging =
      createBuffer(arrayBytes, vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eHostVisible |
                       vk::MemoryPropertyFlagBits::eHostCoherent,
                   true);

  _bufferPartialStaging =
      createBuffer(partialBytes, vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eHostVisible |
                       vk::MemoryPropertyFlagBits::eHostCoherent,
                   true);
}

template <class T>
uint32_t
VulkanStream<T>::findMemoryType(uint32_t typeBits,
                                vk::MemoryPropertyFlags properties) const {
  const auto &physicalDevice = _devices.physicalDevices[_deviceIndex];
  const auto memProps = physicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    const bool supported = (typeBits & (1u << i)) != 0;
    const bool matches =
        (memProps.memoryTypes[i].propertyFlags & properties) == properties;
    if (supported && matches) {
      return i;
    }
  }

  throw std::runtime_error(
      "Unable to find a Vulkan memory type matching requirements");

  return 0;
}

template <class T>
Buffer VulkanStream<T>::createBuffer(vk::DeviceSize size,
                                     vk::BufferUsageFlags usage,
                                     vk::MemoryPropertyFlags properties,
                                     bool map) const {
  Buffer out;
  out.size = size;

  vk::BufferCreateInfo bufferInfo;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;
  out.buffer = vk::raii::Buffer(_device, bufferInfo);

  vk::MemoryRequirements requirements = out.buffer.getMemoryRequirements();

  vk::MemoryAllocateInfo allocInfo;
  allocInfo.allocationSize = requirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(requirements.memoryTypeBits, properties);
  out.memory = vk::raii::DeviceMemory(_device, allocInfo);

  out.buffer.bindMemory(out.memory, 0);

  if (map) {
    out.mapped = out.memory.mapMemory(0, size);
  }

  return out;
}

template <class T>
std::pair<size_t, const uint32_t *>
VulkanStream<T>::readShader(const ShaderKernel entry) const {
  constexpr bool useDouble = sizeof(T) == sizeof(double);
  switch (entry) {
  case ShaderKernel::INIT:
    return useDouble
               ? std::make_pair(initShader_double_sizeInBytes,
                                initShader_double)
               : std::make_pair(initShader_float_sizeInBytes, initShader_float);
  case ShaderKernel::COPY:
    return useDouble
               ? std::make_pair(copyShader_double_sizeInBytes,
                                copyShader_double)
               : std::make_pair(copyShader_float_sizeInBytes, copyShader_float);
  case ShaderKernel::MUL:
    return useDouble
               ? std::make_pair(mulShader_double_sizeInBytes, mulShader_double)
               : std::make_pair(mulShader_float_sizeInBytes, mulShader_float);
  case ShaderKernel::ADD:
    return useDouble
               ? std::make_pair(addShader_double_sizeInBytes, addShader_double)
               : std::make_pair(addShader_float_sizeInBytes, addShader_float);
  case ShaderKernel::TRIAD:
    return useDouble ? std::make_pair(triadShader_double_sizeInBytes,
                                      triadShader_double)
                     : std::make_pair(triadShader_float_sizeInBytes,
                                      triadShader_float);
  case ShaderKernel::NSTREAM:
    return useDouble ? std::make_pair(nstreamShader_double_sizeInBytes,
                                      nstreamShader_double)
                     : std::make_pair(nstreamShader_float_sizeInBytes,
                                      nstreamShader_float);
  case ShaderKernel::DOT:
    return useDouble
               ? std::make_pair(dotShader_double_sizeInBytes, dotShader_double)
               : std::make_pair(dotShader_float_sizeInBytes, dotShader_float);
  default:
    throw std::runtime_error("Unexpected shader requested to create pipeline");
  }
}

template <class T>
vk::raii::Pipeline VulkanStream<T>::createPipeline(const ShaderKernel entry) {
  auto [spirvSize, spirvBinary] = readShader(entry);

  vk::ShaderModuleCreateInfo moduleInfo;
  moduleInfo.codeSize = spirvSize;
  moduleInfo.pCode = spirvBinary;
  vk::raii::ShaderModule shaderModule = _device.createShaderModule(moduleInfo);

  // Specialization constants that must match ids in shader module
  struct __attribute__((__packed__)) SpecializationBuffer {
    uint32_t workgroupSize; // spec constant id 0
    uint32_t elementCount;  // spec constant id 1
    uint32_t dotGridSize;   // spec constant id 2
    T initA;                // spec constant id 3
    T initB;                // spec constant id 4
    T initC;                // spec constant id 5
    T scalar;               // spec constant id 6
  };

  uint32_t offset = 0;
  uint32_t index = 0;
  std::vector<vk::SpecializationMapEntry> specializationMap;
  specializationMap.emplace_back(index++, offset, sizeof(uint32_t));
  offset += sizeof(uint32_t);
  specializationMap.emplace_back(index++, offset, sizeof(uint32_t));
  offset += sizeof(uint32_t);
  specializationMap.emplace_back(index++, offset, sizeof(uint32_t));
  offset += sizeof(uint32_t);
  specializationMap.emplace_back(index++, offset, sizeof(T));
  offset += sizeof(T);
  specializationMap.emplace_back(index++, offset, sizeof(T));
  offset += sizeof(T);
  specializationMap.emplace_back(index++, offset, sizeof(T));
  offset += sizeof(T);
  specializationMap.emplace_back(index++, offset, sizeof(T));

  const uint32_t dotDispatchElems = _dotNumDispatchGroups * _workgroupSize;
  SpecializationBuffer specializationData{
      _workgroupSize,   static_cast<uint32_t>(_arraySize),
      dotDispatchElems, _initA,
      _initB,           _initC,
      T(startScalar)};

  vk::SpecializationInfo specializationInfo(
      specializationMap.size(), specializationMap.data(),
      sizeof(SpecializationBuffer), &specializationData);

  vk::PipelineShaderStageCreateInfo stageInfo(
      {}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main");
  stageInfo.pSpecializationInfo = &specializationInfo;

  vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, _pipelineLayout);

  return vk::raii::Pipeline(_device, nullptr, pipelineInfo);
}

template <class T> void VulkanStream<T>::createPipelines() {
  _pipelines.init = createPipeline(ShaderKernel::INIT);
  _pipelines.copy = createPipeline(ShaderKernel::COPY);
  _pipelines.mul = createPipeline(ShaderKernel::MUL);
  _pipelines.add = createPipeline(ShaderKernel::ADD);
  _pipelines.triad = createPipeline(ShaderKernel::TRIAD);
  _pipelines.nstream = createPipeline(ShaderKernel::NSTREAM);
  _pipelines.dot = createPipeline(ShaderKernel::DOT);
}

template <class T> void VulkanStream<T>::writeDescriptorSets() {
  // The same buffers are used by all the compute pipelines, so
  // we only need to update the descriptor sets once at the start
  // of the benchmark, rather than updating it per operation.
  const vk::DescriptorBufferInfo aInfo(*_bufferA.buffer, 0, VK_WHOLE_SIZE);
  const vk::DescriptorBufferInfo bInfo(*_bufferB.buffer, 0, VK_WHOLE_SIZE);
  const vk::DescriptorBufferInfo cInfo(*_bufferC.buffer, 0, VK_WHOLE_SIZE);
  const vk::DescriptorBufferInfo partialInfo(*_bufferPartial.buffer, 0,
                                             VK_WHOLE_SIZE);

  std::array<vk::WriteDescriptorSet, _descriptorCount> writes{};

  writes[0].dstSet = descriptorSets.front();
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
  writes[0].pBufferInfo = &aInfo;

  writes[1].dstSet = descriptorSets.front();
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
  writes[1].pBufferInfo = &bInfo;

  writes[2].dstSet = descriptorSets.front();
  writes[2].dstBinding = 2;
  writes[2].descriptorCount = 1;
  writes[2].descriptorType = vk::DescriptorType::eStorageBuffer;
  writes[2].pBufferInfo = &cInfo;

  writes[3].dstSet = descriptorSets.front();
  writes[3].dstBinding = 3;
  writes[3].descriptorCount = 1;
  writes[3].descriptorType = vk::DescriptorType::eStorageBuffer;
  writes[3].pBufferInfo = &partialInfo;

  _device.updateDescriptorSets(writes, {});
}

template <class T>
void VulkanStream<T>::dispatch(
    const vk::Pipeline &pipeline,
    const std::vector<vk::BufferMemoryBarrier> &barriers) {
  // Dispatches a command-buffer with a single compute pipeline command
  vk::CommandBufferAllocateInfo allocInfo{*_commandPool,
                                          vk::CommandBufferLevel::ePrimary, 1};

  auto commands = vk::raii::CommandBuffers(_device, allocInfo);
  auto &cmd = commands.front();

  vk::CommandBufferBeginInfo beginInfo;
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd.begin(beginInfo);

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *_pipelineLayout, 0,
                         {*descriptorSets.front()}, {});

  if (!barriers.empty()) {
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {},
                        barriers, {});
  }

  cmd.dispatch(_numDispatchGroups, 1, 1);

  cmd.end();

  const vk::CommandBuffer raw = *cmd;
  vk::SubmitInfo submitInfo;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &raw;

  _queue.submit(submitInfo, nullptr);
  _queue.waitIdle();
}

template <class T> void VulkanStream<T>::copy() {
  // The Copy kernel reads from the A buffer and writes to the C buffer
  std::vector<vk::BufferMemoryBarrier> barriers;
  enforceMemDep(_bufferA, barriers, vk::AccessFlagBits::eShaderRead);
  enforceMemDep(_bufferC, barriers, vk::AccessFlagBits::eShaderWrite);

  dispatch(*_pipelines.copy, barriers);
}

template <class T> void VulkanStream<T>::mul() {
  // The Mul kernel reads from the C buffer and writes to the B buffer
  std::vector<vk::BufferMemoryBarrier> barriers;
  enforceMemDep(_bufferB, barriers, vk::AccessFlagBits::eShaderWrite);
  enforceMemDep(_bufferC, barriers, vk::AccessFlagBits::eShaderRead);

  dispatch(*_pipelines.mul, barriers);
}

template <class T> void VulkanStream<T>::add() {
  // The Add kernel reads from the A & B buffers and writes to the C buffer
  std::vector<vk::BufferMemoryBarrier> barriers;
  enforceMemDep(_bufferA, barriers, vk::AccessFlagBits::eShaderRead);
  enforceMemDep(_bufferB, barriers, vk::AccessFlagBits::eShaderRead);
  enforceMemDep(_bufferC, barriers, vk::AccessFlagBits::eShaderWrite);

  dispatch(*_pipelines.add, barriers);
}

template <class T> void VulkanStream<T>::triad() {
  // The Triad kernel reads from the B & C buffers and writes to the A buffer
  std::vector<vk::BufferMemoryBarrier> barriers;
  enforceMemDep(_bufferA, barriers, vk::AccessFlagBits::eShaderWrite);
  enforceMemDep(_bufferB, barriers, vk::AccessFlagBits::eShaderRead);
  enforceMemDep(_bufferC, barriers, vk::AccessFlagBits::eShaderRead);

  dispatch(*_pipelines.triad, barriers);
}

template <class T> void VulkanStream<T>::nstream() {
  // The Nstream kernel reads from the A, B, C buffers and writes to the A
  // buffer
  std::vector<vk::BufferMemoryBarrier> barriers;
  enforceMemDep(_bufferA, barriers,
                vk::AccessFlagBits(VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT));
  enforceMemDep(_bufferB, barriers, vk::AccessFlagBits::eShaderRead);
  enforceMemDep(_bufferC, barriers, vk::AccessFlagBits::eShaderRead);

  dispatch(*_pipelines.nstream, barriers);
}

template <class T> T VulkanStream<T>::dot() {
  // Unlike the other device operations this interface returns the final value,
  // rather than void. The dot operation therefor contains two commands in the
  // command-buffer. A compute kernel dispatch that writes to the device side
  // buffer which an element for each workgroup in the dispatch. The second
  // command in the command-buffer then copies from the device side buffer
  // to a host visible and coherent staging buffer.
  vk::CommandBufferAllocateInfo allocInfo{*_commandPool,
                                          vk::CommandBufferLevel::ePrimary, 1};

  auto commands = vk::raii::CommandBuffers(_device, allocInfo);
  auto &cmd = commands.front();

  vk::CommandBufferBeginInfo beginInfo;
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd.begin(beginInfo);

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *_pipelines.dot);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *_pipelineLayout, 0,
                         {*descriptorSets.front()}, {});

  cmd.dispatch(_dotNumDispatchGroups, 1, 1);

  vk::BufferMemoryBarrier computeToTransfer(
      vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, _bufferPartial.buffer,
      0, VK_WHOLE_SIZE);
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eTransfer, {}, {},
                      {computeToTransfer}, {});

  vk::BufferCopy copyRegion(0, 0, _bufferPartial.size);
  cmd.copyBuffer(*_bufferPartial.buffer, *_bufferPartialStaging.buffer,
                 copyRegion);

  vk::BufferMemoryBarrier transferToHost(
      vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      _bufferPartialStaging.buffer, 0, VK_WHOLE_SIZE);

  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                      vk::PipelineStageFlagBits::eHost, {}, {},
                      {transferToHost}, {});

  cmd.end();

  const vk::CommandBuffer raw = *cmd;
  vk::SubmitInfo submitInfo;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &raw;

  _queue.submit(submitInfo, nullptr);
  _queue.waitIdle();

  // Accumulate across all the partial values calculated by each workgroup
  // using the host mapped pointer from the staging buffer
  const T *partials = static_cast<const T *>(_bufferPartialStaging.mapped);
  T total = std::accumulate(partials, partials + _dotNumDispatchGroups, T{});

  return total;
}

template <class T>
void VulkanStream<T>::init_arrays(T initA, T initB, T initC) {
  // These values match those set on VulkanStream object construction where they
  // are embedded in the device code as specialization constants
  (void)initA;
  (void)initB;
  (void)initC;

  // The init kernel is the first kernel executed and writes to buffers A,B,C
  // based on device side values.
  std::vector<vk::BufferMemoryBarrier> barriers;
  dispatch(*_pipelines.init, barriers);

  _bufferA.state = vk::AccessFlagBits::eShaderWrite;
  _bufferB.state = vk::AccessFlagBits::eShaderWrite;
  _bufferC.state = vk::AccessFlagBits::eShaderWrite;
}

template <class T>
void VulkanStream<T>::copyDeviceToHost(const Buffer &src, std::vector<T> &dst) {
  dst.resize(_arraySize);

  // Create and submit a command-buffer with a copy command that will copy our
  // device buffer into the staging buffer which is host visible & coherent.
  vk::CommandBufferAllocateInfo allocInfo{*_commandPool,
                                          vk::CommandBufferLevel::ePrimary, 1};

  auto commands = vk::raii::CommandBuffers(_device, allocInfo);
  auto &cmd = commands.front();

  vk::CommandBufferBeginInfo beginInfo;
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd.begin(beginInfo);

  vk::BufferMemoryBarrier computeToTransfer(
      src.state, vk::AccessFlagBits::eTransferRead, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, src.buffer, 0, VK_WHOLE_SIZE);

  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eTransfer, {}, {},
                      {computeToTransfer}, {});

  vk::BufferCopy region(0, 0, _arraySize * sizeof(T));
  cmd.copyBuffer(*src.buffer, *_bufferStaging.buffer, region);

  vk::BufferMemoryBarrier transferToHost(
      vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, _bufferStaging.buffer,
      0, VK_WHOLE_SIZE);

  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                      vk::PipelineStageFlagBits::eHost, {}, {},
                      {transferToHost}, {});

  cmd.end();

  const vk::CommandBuffer raw = *cmd;
  vk::SubmitInfo submitInfo;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &raw;

  _queue.submit(submitInfo, nullptr);
  _queue.waitIdle();

  // Our staging buffer is mapped to a host accessible pointer, copy contents of
  // the staging buffer to the host vector
  std::memcpy(dst.data(), _bufferStaging.mapped, _arraySize * sizeof(T));
}

template <class T>
void VulkanStream<T>::get_arrays(T const *&a, T const *&b, T const *&c) {
  copyDeviceToHost(_bufferA, _hostA);
  copyDeviceToHost(_bufferB, _hostB);
  copyDeviceToHost(_bufferC, _hostC);

  a = _hostA.data();
  b = _hostB.data();
  c = _hostC.data();
}

void listDevices(void) {
  VulkanDevices devices;

  const auto &physicalDevices = devices.physicalDevices;
  if (physicalDevices.empty()) {
    std::cerr << "No devices found." << std::endl;
    return;
  }

  std::cout << "\nDevices:" << std::endl;
  for (size_t i = 0; i < physicalDevices.size(); ++i) {
    std::cout << i << ": " << physicalDevices[i].getProperties().deviceName;
    std::cout << ". Driver - " << getDeviceDriver(physicalDevices[i])
              << std::endl;
  }
  std::cout << std::endl;
}

template class VulkanStream<float>;
template class VulkanStream<double>;
