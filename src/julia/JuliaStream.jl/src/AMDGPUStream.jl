# AMDGPU.jl doesn't support CPU agents, so this isn't a feature-complete ROCmStream, only AMD GPUs
include("Stream.jl")
using AMDGPU

const ROCData = StreamData{T,ROCArray{T}} where {T}
const TBSize = 1024::Int
const DotBlocks = 256::Int

function devices()::Vector{DeviceWithRepr}
  return !AMDGPU.functional() ? String[] : map(d -> (d, repr(d), "AMDGPU.jl"), AMDGPU.devices())
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

  AMDGPU.device!(device[1])
  selected = AMDGPU.device()
  if selected != device[1]
    error("Cannot select HIP device, expecting $(device[1]), but got $(selected)")
  end
  if !silent
    println("Using HIP device: $(repr(selected))")
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
  fill!(data.a, init[1])
  fill!(data.b, init[2])
  fill!(data.c, init[3])
end

function copy!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, c::AbstractArray{T})
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds c[i] = a[i]
    return
  end
  @roc groupsize = TBSize gridsize = data.size ÷ TBSize kernel(data.a, data.c)
  AMDGPU.synchronize()
end

function mul!(data::ROCData{T}, _) where {T}
  function kernel(b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds b[i] = scalar * c[i]
    return
  end
  @roc groupsize = TBSize gridsize = data.size ÷ TBSize kernel(data.b, data.c, data.scalar)
  AMDGPU.synchronize()
end

function add!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T})
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds c[i] = a[i] + b[i]
    return
  end
  @roc groupsize = TBSize gridsize = data.size ÷ TBSize kernel(data.a, data.b, data.c)
  AMDGPU.synchronize()
end

function triad!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds a[i] = b[i] + (scalar * c[i])
    return
  end
  @roc groupsize = TBSize gridsize = data.size ÷ TBSize kernel(
    data.a,
    data.b,
    data.c,
    data.scalar,
  )
  AMDGPU.synchronize()
end

function nstream!(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    @inbounds a[i] += b[i] + scalar * c[i]
    return
  end
  @roc groupsize = TBSize gridsize = data.size ÷ TBSize kernel(
    data.a,
    data.b,
    data.c,
    data.scalar,
  )
  AMDGPU.synchronize()
end

function dot(data::ROCData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, size::Int, partial::AbstractArray{T})
    tb_sum = @ROCStaticLocalArray(T, TBSize)
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
  @roc groupsize = TBSize gridsize = DotBlocks kernel(
    data.a,
    data.b,
    data.size,
    partial_sum,
  )
  AMDGPU.synchronize()
  return sum(partial_sum)
end

function read_data(data::ROCData{T}, _)::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()