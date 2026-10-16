using KernelAbstractions, CUDA, AMDGPU
include("Stream.jl")

const CuData = StreamData{T,CUDA.CuArray{T}} where {T}
const ROCData = StreamData{T,AMDGPU.ROCArray{T}} where {T}

const TBSize = 1024::Int
const DotBlocks = 256::Int
const DotOffsets = ntuple(k -> TBSize >> k, trailing_zeros(TBSize))

@enum Backend cuda rocm cpu

struct Context
  backend::Backend
  device::KernelAbstractions.Backend
end

function list_rocm_devices()::Vector{DeviceWithRepr}
  return !AMDGPU.functional() ? [] : map(d -> (d, repr(d), rocm), AMDGPU.devices())
end

function list_cuda_devices()::Vector{DeviceWithRepr}
  return !CUDA.functional(false) ? String[] :
         map(d -> (d, "$(CUDA.name(d)) ($(repr(d)))", cuda), CUDA.devices())
end

function devices()::Vector{DeviceWithRepr}
  cudas = list_cuda_devices()
  rocms = list_rocm_devices()
  cpus = [(undef, "$(Sys.cpu_info()[1].model) ($(Threads.nthreads())T)", cpu)]
  vcat(cpus, cudas, rocms)
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::DeviceWithRepr,
  silent::Bool,
) where {T}

  if arraysize % TBSize != 0
    error("arraysize ($(arraysize)) must be divisible by $(TBSize)!")
  end

  (selected, _, backend) = device
  if backend == cpu
    if !silent
      println("Using CPU with max $(Threads.nthreads()) threads")
    end
    partialsum = Vector{T}(undef, DotBlocks)
    data = VectorData{T}(
      Vector{T}(undef, arraysize),
      Vector{T}(undef, arraysize),
      Vector{T}(undef, arraysize),
      scalar,
      arraysize,
    )
    backend_impl = CPU()
  elseif backend == cuda
    CUDA.device!(selected)
    if CUDA.device() != selected
      error("Cannot select CUDA device, expecting $selected, but got $(CUDA.device())")
    end
    if !CUDA.functional(true)
      error("Non-functional CUDA configuration")
    end
    if !silent
      println("Using CUDA device: $(CUDA.name(selected)) ($(repr(selected)))")
    end
    partialsum = CuArray{T}(undef, DotBlocks)
    data = CuData{T}(
      CuArray{T}(undef, arraysize),
      CuArray{T}(undef, arraysize),
      CuArray{T}(undef, arraysize),
      scalar,
      arraysize,
    )
    backend_impl = CUDABackend()
  elseif backend == rocm
    AMDGPU.device!(selected)
    if AMDGPU.device() != selected
      error("Cannot select HIP device, expecting $selected, but got $(AMDGPU.device())")
    end
    if !silent
      println("Using HIP device: $(repr(selected))")
    end
    partialsum = ROCArray{T}(undef, DotBlocks)
    data = ROCData{T}(
      ROCArray{T}(undef, arraysize),
      ROCArray{T}(undef, arraysize),
      ROCArray{T}(undef, arraysize),
      scalar,
      arraysize,
    )
    backend_impl = ROCBackend()
  else
    error("unsupported backend $(backend)")
  end

  if !silent
    println("Kernel parameters   : <<<$(data.size),$(TBSize)>>>")
  end
  return (data, Context(backend, backend_impl))
end

function init_arrays!(
  data::StreamData{T,C},
  context::Context,
  init::Tuple{T,T,T},
) where {T,C}
  if context.backend == cpu
    Threads.@threads :static for i = 1:data.size
      @inbounds data.a[i] = init[1]
      @inbounds data.b[i] = init[2]
      @inbounds data.c[i] = init[3]
    end
  elseif context.backend == cuda
    fill!(data.a, init[1])
    fill!(data.b, init[2])
    fill!(data.c, init[3])
  elseif context.backend == rocm
    fill!(data.a, init[1])
    fill!(data.b, init[2])
    fill!(data.c, init[3])
  else
    error("unsupported backend $(backend)")
  end
end

function copy!(data::StreamData{T,C}, context::Context) where {T,C}
  @kernel function kernel(@Const(a::AbstractArray{T}), c)
    i = @index(Global)
    @inbounds c[i] = a[i]
  end
  kernel(context.device, TBSize)(data.a, data.c, ndrange = data.size)
  KernelAbstractions.synchronize(context.device)
end

function mul!(data::StreamData{T,C}, context::Context) where {T,C}
  @kernel function kernel(b::AbstractArray{T}, @Const(c::AbstractArray{T}), scalar::T)
    i = @index(Global)
    @inbounds b[i] = scalar * c[i]
  end
  kernel(context.device, TBSize)(data.b, data.c, data.scalar, ndrange = data.size)
  KernelAbstractions.synchronize(context.device)
end

function add!(data::StreamData{T,C}, context::Context) where {T,C}
  @kernel function kernel(@Const(a::AbstractArray{T}), @Const(b::AbstractArray{T}), c)
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
  end
  kernel(context.device, TBSize)(data.a, data.b, data.c, ndrange = data.size)
  KernelAbstractions.synchronize(context.device)
end

function triad!(data::StreamData{T,C}, context::Context) where {T,C}
  @kernel function kernel(a::AbstractArray{T}, @Const(b::AbstractArray{T}), @Const(c), scalar::T)
    i = @index(Global)
    @inbounds a[i] = b[i] + (scalar * c[i])
  end
  kernel(context.device, TBSize)(
    data.a,
    data.b,
    data.c,
    data.scalar,
    ndrange = data.size,
  )
  KernelAbstractions.synchronize(context.device)
end

function nstream!(data::StreamData{T,C}, context::Context) where {T,C}
  @kernel function kernel(a::AbstractArray{T}, @Const(b::AbstractArray{T}), @Const(c), scalar::T)
    i = @index(Global)
    @inbounds a[i] += b[i] + scalar * c[i]
  end
  kernel(context.device, TBSize)(
    data.a,
    data.b,
    data.c,
    data.scalar,
    ndrange = data.size,
  )
  KernelAbstractions.synchronize(context.device)
end

function dot(data::StreamData{T,C}, context::Context) where {T,C}
  @kernel function kernel(@Const(a::AbstractArray{T}), @Const(b::AbstractArray{T}), size::Int, partial::AbstractArray{T})
    local_i = @index(Local)
    tb_sum = @localmem T TBSize
    @inbounds tb_sum[local_i] = 0.0

    # do dot first
    i = @index(Global)
    while i <= size
      @inbounds tb_sum[local_i] += a[i] * b[i]
      i += TBSize * DotBlocks
    end

    # then tree reduction, constant offsets keep private state out of @synchronize for the CPU backend
    @inbounds for offset in DotOffsets
      @synchronize
      li = @index(Local)
      if li <= offset
        tb_sum[li] += tb_sum[li+offset]
      end
    end

    if @index(Local) == 1
      @inbounds partial[@index(Group)] = tb_sum[1]
    end
  end

  if context.backend == cpu
    partial_sum = Vector{T}(undef, DotBlocks)
  elseif context.backend == cuda
    partial_sum = CuArray{T}(undef, DotBlocks)
  elseif context.backend == rocm
    partial_sum = ROCArray{T}(undef, DotBlocks)
  else
    error("unsupported backend $(backend)")
  end

  kernel(context.device, TBSize)(
    data.a,
    data.b,
    data.size,
    partial_sum,
    ndrange = TBSize * DotBlocks,
  )
  KernelAbstractions.synchronize(context.device)

  return sum(partial_sum)
end

function read_data(data::StreamData{T,C}, _::Context)::VectorData{T} where {T,C}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()
