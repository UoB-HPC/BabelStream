include("Stream.jl")
using CUDA

const CuData = StreamData{T,CuArray{T}} where {T}
const TBSize = 1024::Int
const DotBlocks = 256::Int

function devices()
  return !CUDA.functional(false) ? [] :
         map(d -> "$(CUDA.name(d)) ($(repr(d)))", CUDA.devices())
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::Int,
  silent::Bool,
)::CuData{T} where {T}

  if arraysize % TBSize != 0
    error("arraysize ($(arraysize)) must be divisible by $(TBSize)!")
  end

  # so CUDA's device is 0 indexed, so -1 from Julia
  CUDA.device!(device - 1)
  selected = CUDA.device()
  # show_reason is set to true here so it dumps CUDA info 
  # for us regardless of whether it's functional
  if !CUDA.functional(true)
    error("Non-functional CUDA configuration")
  end
  data = CuData{T}(
    CuArray{T}(undef, arraysize),
    CuArray{T}(undef, arraysize),
    CuArray{T}(undef, arraysize),
    scalar,
    arraysize,
  )
  if !silent
    println("Using CUDA device: $(CUDA.name(selected)) ($(repr(selected)))")
    println("Kernel parameters: <<<$(data.size ÷ TBSize),$(TBSize)>>>")
  end
  return data
end

function init_arrays!(data::CuData{T}, init::Tuple{T,T,T}) where {T}
  CUDA.fill!(data.a, init[1])
  CUDA.fill!(data.b, init[2])
  CUDA.fill!(data.c, init[3])
end

function copy!(data::CuData{T}) where {T}
  function kernel(a, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x # only blockIdx starts at 1
    @inbounds c[i] = a[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(data.a, data.c)
  CUDA.synchronize()
end

function mul!(data::CuData{T}) where {T}
  function kernel(b, c, scalar)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x # only blockIdx starts at 1
    @inbounds b[i] = scalar * c[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(data.b, data.c, data.scalar)
  CUDA.synchronize()
end

function add!(data::CuData{T}) where {T}
  function kernel(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x # only blockIdx starts at 1
    @inbounds c[i] = a[i] + b[i]
    return
  end
  @cuda blocks = data.size ÷ TBSize threads = TBSize kernel(data.a, data.b, data.c)
  CUDA.synchronize()
end

function triad!(data::CuData{T}) where {T}
  function kernel(a, b, c, scalar)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x # only blockIdx starts at 1
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

function nstream!(data::CuData{T}) where {T}
  function kernel(a, b, c, scalar)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x # only blockIdx starts at 1
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

function dot(data::CuData{T}) where {T}
  # direct port of the reduction in CUDAStream.cu 
  function kernel(a, b, size, partial)
    tb_sum = @cuStaticSharedMem(T, TBSize)
    local_i = threadIdx().x
    @inbounds tb_sum[local_i] = 0.0

    # do dot first
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x # only blockIdx starts at 1
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
  CUDA.synchronize()
  return sum(partial_sum)
end

function read_data(data::CuData{T})::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()