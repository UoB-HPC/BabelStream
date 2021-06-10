include("Stream.jl")

function devices()
  return ["CPU"]
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::Int,
  silent::Bool,
)::VectorData{T} where {T}
  if device != 1
    error("Only CPU device is supported")
  end
  if !silent
    println("Using max $(Threads.nthreads()) threads")
  end
  return VectorData{T}(1:arraysize, 1:arraysize, 1:arraysize, scalar, arraysize)
end

function init_arrays!(data::VectorData{T}, init::Tuple{T,T,T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.a[i] = init[1]
    @inbounds data.b[i] = init[2]
    @inbounds data.c[i] = init[3]
  end
end

function copy!(data::VectorData{T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.c[i] = data.a[i]
  end
end

function mul!(data::VectorData{T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.b[i] = data.scalar * data.c[i]
  end
end

function add!(data::VectorData{T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.c[i] = data.a[i] + data.b[i]
  end
end

function triad!(data::VectorData{T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.a[i] = data.b[i] + (data.scalar * data.c[i])
  end
end

function nstream!(data::VectorData{T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.a[i] += data.b[i] + data.scalar * data.c[i]
  end
end

function dot(data::VectorData{T}) where {T}
  partial = zeros(T, Threads.nthreads())
  Threads.@threads for i = 1:data.size
    @inbounds partial[Threads.threadid()] += data.a[i] * data.b[i]
  end
  return sum(partial)
end

function read_data(data::VectorData{T})::VectorData{T} where {T}
  return data
end

main()