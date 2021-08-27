include("Stream.jl")
using CUDA

const CuData = StreamData{T,CuArray{T}} where {T}
const TBSize = 1024::Int
const DotBlocks = 256::Int

function devices()::Vector{DeviceWithRepr}
  return !CUDA.functional(false) ? String[] :
         map(d -> (d, "$(CUDA.name(d)) ($(repr(d)))", "CUDA.jl"), CUDA.devices())
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::DeviceWithRepr,
  silent::Bool,
)::Tuple{CuData{T},Nothing} where {T}

  if arraysize % TBSize != 0
    error("arraysize ($(arraysize)) must be divisible by $(TBSize)!")
  end

  CUDA.device!(device[1])
  selected = CUDA.device()
  # show_reason is set to true here so it dumps CUDA info 
  # for us regardless of whether it's functional
  if !CUDA.functional(true)
    error("Non-functional CUDA configuration")
  end
  if !silent
    println("Using CUDA device: $(CUDA.name(selected)) ($(repr(selected)))")
    println("Kernel parameters: <<<$(arraysize ÷ TBSize),$(TBSize)>>>")
  end
  return (
    CuData{T}(
      CuArray{T}(undef, arraysize),
      CuArray{T}(undef, arraysize),
      CuArray{T}(undef, arraysize),
      scalar,
      arraysize,
    ),
    nothing,
  )
end

function init_arrays!(data::CuData{T}, _, init::Tuple{T,T,T}) where {T}
  fill!(data.a, init[1])
  fill!(data.b, init[2])
  fill!(data.c, init[3])
end

function copy!(data::CuData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, c::AbstractArray{T})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds c[i] = a[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(data.a, data.c)
  CUDA.synchronize()
end

function mul!(data::CuData{T}, _) where {T}
  function kernel(b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds b[i] = scalar * c[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(data.b, data.c, data.scalar)
  CUDA.synchronize()
end

function add!(data::CuData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds c[i] = a[i] + b[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(data.a, data.b, data.c)
  CUDA.synchronize()
end

function triad!(data::CuData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds a[i] = b[i] + (scalar * c[i])
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(
    data.a,
    data.b,
    data.c,
    data.scalar,
  )
  CUDA.synchronize()
end

function nstream!(data::CuData{T}, _) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds a[i] += b[i] + scalar * c[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(
    data.a,
    data.b,
    data.c,
    data.scalar,
  )
  CUDA.synchronize()
end

function dot(data::CuData{T}, _) where {T}
  # direct port of the reduction in CUDAStream.cu 
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, size::Int, partial::AbstractArray{T})
    tb_sum = @cuStaticSharedMem(T, TBSize)
    local_i = threadIdx().x
    @inbounds tb_sum[local_i] = 0.0

    # do dot first
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    while i <= size
      @inbounds tb_sum[local_i] += a[i] * b[i]
      i += blockDim().x * gridDim().x
    end

    # then tree reduction
    offset = blockDim().x ÷ 2
    while offset > 0
      sync_threads()
      if (local_i - 1) < offset
        @inbounds tb_sum[local_i] += tb_sum[local_i+offset]
      end
      offset ÷= 2
    end

    if (local_i == 1)
      @inbounds partial[blockIdx().x] = tb_sum[local_i]
    end

    return
  end
  partial_sum = CuArray{T}(undef, DotBlocks)
  @cuda blocks = DotBlocks threads = TBSize kernel(data.a, data.b, data.size, partial_sum)
  return sum(partial_sum)
end

function read_data(data::CuData{T}, _)::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()