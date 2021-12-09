# AMDGPU.jl doesn't support CPU agents, so this isn't a feature-complete ROCmStream, only AMD GPUs
include("Stream.jl")
using AMDGPU

const ROCData = StreamData{T,ROCArray{T}} where {T}
const TBSize = 1024::Int
const DotBlocks = 256::Int

function devices()::Vector{DeviceWithRepr}
  try
    # AMDGPU.agents()'s internal iteration order isn't stable
    sorted = sort(AMDGPU.get_agents(:gpu), by = repr)
    map(x -> (x, repr(x), "AMDGPU.jl"), sorted)
  catch
    # probably unsupported
    String[]
  end
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::DeviceWithRepr,
  silent::Bool,
)::Tuple{ROCData{T},Nothing} where {T}

  if arraysize % TBSize != 0
    error("arraysize ($(arraysize)) must be divisible by $(TBSize)!")
  end

  # XXX AMDGPU doesn't expose an API for setting the default like CUDA.device!()
  # but AMDGPU.get_default_agent returns DEFAULT_AGENT so we can do it by hand
  AMDGPU.DEFAULT_AGENT[] = device[1]
  selected = AMDGPU.get_default_agent()
  if !silent
    println("Using GPU HSA device: $(AMDGPU.get_name(selected)) ($(repr(selected)))")
    println("Kernel parameters   : <<<$(arraysize),$(TBSize)>>>")
  end
  return (
    ROCData{T}(
      ROCArray{T}(undef, arraysize),
      ROCArray{T}(undef, arraysize),
      ROCArray{T}(undef, arraysize),
      scalar,
      arraysize,
    ),
    nothing,
  )
end

function init_arrays!(data::ROCData{T}, _, init::Tuple{T,T,T}) where {T}
  AMDGPU.fill!(data.a, init[1])
  AMDGPU.fill!(data.b, init[2])
  AMDGPU.fill!(data.c, init[3])
end

function copy!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, c::AbstractArray{T})
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds c[i] = a[i]
    return
  end
  AMDGPU.wait(
    @roc groupsize = TBSize gridsize = data.size kernel(data.a, data.c)
  )
end

function mul!(data::ROCData{T}, _) where {T}
  function kernel(b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds b[i] = scalar * c[i]
    return
  end
  AMDGPU.wait(
    @roc groupsize = TBSize gridsize = data.size kernel(data.b, data.c, data.scalar)
  )
end

function add!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T})
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds c[i] = a[i] + b[i]
    return
  end
  AMDGPU.wait(
    @roc groupsize = TBSize gridsize = data.size kernel(data.a, data.b, data.c)
  )
end

function triad!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds a[i] = b[i] + (scalar * c[i])
    return
  end
  AMDGPU.wait(
    @roc groupsize = TBSize gridsize = data.size kernel(
      data.a,
      data.b,
      data.c,
      data.scalar,
    )
  )
end

function nstream!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds a[i] += b[i] + scalar * c[i]
    return
  end
  AMDGPU.wait(
    @roc groupsize = TBSize gridsize = data.size kernel(
      data.a,
      data.b,
      data.c,
      data.scalar,
    )
  )
end

function dot(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, size::Int, partial::AbstractArray{T})
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
    @roc groupsize = TBSize gridsize = TBSize * DotBlocks kernel(
      data.a,
      data.b,
      data.size,
      partial_sum,
    )
  )
  return sum(partial_sum)
end

function read_data(data::ROCData{T}, _)::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()