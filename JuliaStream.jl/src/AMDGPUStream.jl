# AMDGPU.jl doesn't support CPU agents, so this isn't a feature-complete ROCmStream, only AMD GPUs
include("Stream.jl")
using AMDGPU

const ROCData = StreamData{T,ROCArray{T}} where {T}
const TBSize = 1024::Int
const DotBlocks = 256::Int

# AMDGPU.agents()'s internal iteration order isn't stable
function gpu_agents_in_repr_order()
  # XXX if we select anything other than :gpu, we get 
  # HSA_STATUS_ERROR_INVALID_AGENT on the first kernel submission
  sort(AMDGPU.get_agents(:gpu), by = repr)
end

function devices()
  try
    map(repr, gpu_agents_in_repr_order())
  catch
    # probably unsupported
    []
  end
end

function gridsize(data::ROCData{T})::Int where {T}
  return data.size
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::Int,
  silent::Bool,
)::ROCData{T} where {T}

  if arraysize % TBSize != 0
    error("arraysize ($(arraysize)) must be divisible by $(TBSize)!")
  end

  # XXX AMDGPU doesn't expose an API for setting the default like CUDA.device!()
  # but AMDGPU.get_default_agent returns DEFAULT_AGENT so we can do it by hand
  AMDGPU.DEFAULT_AGENT[] = gpu_agents_in_repr_order()[device]

  data = ROCData{T}(
    ROCArray{T}(undef, arraysize),
    ROCArray{T}(undef, arraysize),
    ROCArray{T}(undef, arraysize),
    scalar,
    arraysize,
  )
  selected = AMDGPU.get_default_agent()
  if !silent
    println("Using GPU HSA device: $(AMDGPU.get_name(selected)) ($(repr(selected)))")
    println("Kernel parameters   : <<<$(gridsize(data)),$(TBSize)>>>")
  end
  return data
end

function init_arrays!(data::ROCData{T}, init::Tuple{T,T,T}) where {T}
  AMDGPU.fill!(data.a, init[1])
  AMDGPU.fill!(data.b, init[2])
  AMDGPU.fill!(data.c, init[3])
end

function copy!(data::ROCData{T}) where {T}
  function kernel(a, c)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds c[i] = a[i]
    return
  end
  AMDGPU.wait(
    soft = false, # soft wait causes HSA_REFCOUNT overflow issues
    @roc groupsize = TBSize gridsize = gridsize(data) kernel(data.a, data.c)
  )
end

function mul!(data::ROCData{T}) where {T}
  function kernel(b, c, scalar)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds b[i] = scalar * c[i]
    return
  end
  AMDGPU.wait(
    soft = false, # soft wait causes HSA_REFCOUNT overflow issues
    @roc groupsize = TBSize gridsize = gridsize(data) kernel(data.b, data.c, data.scalar)
  )
end

function add!(data::ROCData{T}) where {T}
  function kernel(a, b, c)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds c[i] = a[i] + b[i]
    return
  end
  AMDGPU.wait(
    soft = false, # soft wait causes HSA_REFCOUNT overflow issues
    @roc groupsize = TBSize gridsize = gridsize(data) kernel(data.a, data.b, data.c)
  )
end

function triad!(data::ROCData{T}) where {T}
  function kernel(a, b, c, scalar)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds a[i] = b[i] + (scalar * c[i])
    return
  end
  AMDGPU.wait(
    soft = false, # soft wait causes HSA_REFCOUNT overflow issues
    @roc groupsize = TBSize gridsize = gridsize(data) kernel(
      data.a,
      data.b,
      data.c,
      data.scalar,
    )
  )
end

function nstream!(data::ROCData{T}) where {T}
  function kernel(a, b, c, scalar)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds a[i] += b[i] + scalar * c[i]
    return
  end
  AMDGPU.wait(
    soft = false, # soft wait causes HSA_REFCOUNT overflow issues
    @roc groupsize = TBSize gridsize = gridsize(data) kernel(
      data.a,
      data.b,
      data.c,
      data.scalar,
    )
  )
end

function dot(data::ROCData{T}) where {T}
  function kernel(a, b, size, partial)
    tb_sum = ROCDeviceArray((TBSize,), alloc_local(:reduce, T, TBSize))
    local_i = workitemIdx().x
    @inbounds tb_sum[local_i] = 0.0

    # do dot first
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    while i <= size
      @inbounds tb_sum[local_i] += a[i] * b[i]
      i += TBSize * DotBlocks # XXX don't use (workgroupDim().x * gridDimWG().x) here
    end

    # then tree reduction
    offset = workgroupDim().x ÷ 2
    while offset > 0
      sync_workgroup()
      if (local_i - 1) < offset
        @inbounds tb_sum[local_i] += tb_sum[local_i+offset]
      end
      offset ÷= 2
    end

    if (local_i == 1)
      @inbounds partial[workgroupIdx().x] = tb_sum[local_i]
    end

    return
  end
  partial_sum = ROCArray{T}(undef, DotBlocks)
  AMDGPU.wait(
    soft = false, # soft wait causes HSA_REFCOUNT overflow issues
    @roc groupsize = TBSize gridsize = TBSize * DotBlocks kernel(
      data.a,
      data.b,
      data.size,
      partial_sum,
    )
  )
  return sum(partial_sum)
end

function read_data(data::ROCData{T})::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()