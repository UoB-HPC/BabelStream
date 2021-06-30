using Distributed

@everywhere using Pkg
@everywhere Pkg.activate("."; io=devnull) # don't spam `Activating environment at...`
@everywhere include("StreamData.jl")
@everywhere include("Stream.jl")
@everywhere using SharedArrays
@everywhere const SharedArrayData = StreamData{T,SharedArray{T}} where {T}

function devices()
  return ["CPU (localhost)"]
end

function make_stream(
  arraysize::Int,
  scalar::T,
  device::Int,
  silent::Bool,
)::SharedArrayData{T} where {T}
  if device != 1
    error("Only CPU device is supported")
  end

  if !silent
    println("Using max $(nworkers()) process(es) + 1 master")
  end
  return SharedArrayData{T}(
    SharedArray{T}(arraysize),
    SharedArray{T}(arraysize),
    SharedArray{T}(arraysize),
    scalar,
    arraysize,
  )
end

function init_arrays!(data::SharedArrayData{T}, init::Tuple{T,T,T}) where {T}

  @sync @distributed for i = 1:data.size
    @inbounds data.a[i] = init[1]
    @inbounds data.b[i] = init[2]
    @inbounds data.c[i] = init[3]
  end
end

function copy!(data::SharedArrayData{T}) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.c[i] = data.a[i]
  end
end

function mul!(data::SharedArrayData{T}) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.b[i] = data.scalar * data.c[i]
  end
end

function add!(data::SharedArrayData{T}) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.c[i] = data.a[i] + data.b[i]
  end
end

function triad!(data::SharedArrayData{T}) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.a[i] = data.b[i] + (data.scalar * data.c[i])
  end
end

function nstream!(data::SharedArrayData{T}) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.a[i] += data.b[i] + data.scalar * data.c[i]
  end
end

function dot(data::SharedArrayData{T}) where {T}
  return @distributed (+) for i = 1:data.size
    @inbounds data.a[i] * data.b[i]
  end
end

function read_data(data::SharedArrayData{T})::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()