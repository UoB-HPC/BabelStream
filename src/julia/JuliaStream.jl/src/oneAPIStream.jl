using Base.Iterators: println
using Base.Iterators: println
using Printf: Iterators

include("Stream.jl")
using oneAPI

const oneData = StreamData{T,oneArray{T}} where {T}
const DotWGSize = 256::Int

function devices()::Vector{DeviceWithRepr}
  all = map(oneL0.devices, oneL0.drivers()) |> Iterators.flatten |> Iterators.collect
  map(dev -> (dev, repr("text/plain", dev), "oneAPi.jl"), all)
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::DeviceWithRepr,
  silent::Bool,
)::Tuple{oneData{T},Int} where {T}

  oneAPI.allowscalar(false)
  oneAPI.device!(device[1])

  props = oneL0.compute_properties(oneAPI.device())
  groupsize = min(props.maxTotalGroupSize, arraysize)

  if arraysize % groupsize != 0
    error("arraysize ($(arraysize)) must be divisible by $(groupsize)!")
  end

  if !silent
    println("Using L0 device: $(repr("text/plain",device[1]))")
    println("Kernel parameters   : <<<$(arraysize),$(groupsize)>>>")
  end
  return (
    oneData{T}(
      oneArray{T}(undef, arraysize),
      oneArray{T}(undef, arraysize),
      oneArray{T}(undef, arraysize),
      scalar,
      arraysize,
    ),
    groupsize,
  )
end

function init_arrays!(data::oneData{T}, _, init::Tuple{T,T,T}) where {T}
  oneAPI.fill!(data.a, init[1])
  oneAPI.fill!(data.b, init[2])
  oneAPI.fill!(data.c, init[3])
end

function copy!(data::oneData{T}, groupsize::Int) where {T}
  function kernel(a::AbstractArray{T}, c::AbstractArray{T})
    i = get_global_id()
    @inbounds c[i] = a[i]
    return
  end
  @oneapi items = groupsize groups = data.size ÷ groupsize kernel( # 
    data.a,
    data.c,
  )
  oneAPI.synchronize()
end

function mul!(data::oneData{T}, groupsize::Int) where {T}
  function kernel(b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = get_global_id()
    @inbounds b[i] = scalar * c[i]
    return
  end
  @oneapi items = groupsize groups = data.size ÷ groupsize kernel( #
    data.b,
    data.c,
    data.scalar,
  )
  oneAPI.synchronize()
end

function add!(data::oneData{T}, groupsize::Int) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T})
    i = get_global_id()
    @inbounds c[i] = a[i] + b[i]
    return
  end
  @oneapi items = groupsize groups = data.size ÷ groupsize kernel( #
    data.a,
    data.b,
    data.c,
  )
  oneAPI.synchronize()
end

function triad!(data::oneData{T}, groupsize::Int) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = get_global_id()
    @inbounds a[i] = b[i] + (scalar * c[i])
    return
  end
  @oneapi items = groupsize groups = data.size ÷ groupsize kernel( #
    data.a,
    data.b,
    data.c,
    data.scalar,
  )
  oneAPI.synchronize()
end

function nstream!(data::oneData{T}, groupsize::Int) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}, scalar::T)
    i = get_global_id()
    @inbounds a[i] += b[i] + scalar * c[i]
    return
  end
  @oneapi items = groupsize groups = data.size ÷ groupsize kernel( #
    data.a,
    data.b,
    data.c,
    data.scalar,
  )
  oneAPI.synchronize()
end

function dot(data::oneData{T}, groupsize::Int) where {T}
  function kernel(a::AbstractArray{T}, b::AbstractArray{T}, size::Int, partial::AbstractArray{T})
    wg_sum = @LocalMemory(T, (DotWGSize,))
    li = get_local_id()
    @inbounds wg_sum[li] = 0.0

    # do dot first
    i = get_global_id()
    while i <= size
      @inbounds wg_sum[li] += a[i] * b[i]
      i += get_global_size()
    end

    # then tree reduction
    offset = get_local_size() ÷ 2
    while offset > 0
      barrier()
      if li <= offset
        @inbounds wg_sum[li] += wg_sum[li+offset]
      end
      offset ÷= 2
    end

    if li == 1
      @inbounds partial[get_group_id()] =  wg_sum[li]
    end

    return
  end
  partial_sum = oneArray{T}(undef, groupsize)
  @oneapi items = groupsize groups = DotWGSize kernel(
    data.a,
    data.b,
    data.size,
    partial_sum,
  )
  oneAPI.synchronize()
  return sum(partial_sum)
end

function read_data(data::oneData{T}, _)::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()