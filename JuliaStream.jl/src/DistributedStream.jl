using Distributed

@everywhere using Pkg
@everywhere Pkg.activate("."; io = devnull) # don't spam `Activating environment at...`
@everywhere include("StreamData.jl")
@everywhere include("Stream.jl")
@everywhere using SharedArrays
@everywhere const SharedArrayData = StreamData{T,SharedArray{T}} where {T}

function devices()::Vector{DeviceWithRepr}
  return [(undef, "CPU (localhost) $(nworkers())P", "Distributed.jl")]
end

function make_stream(
  arraysize::Int,
  scalar::T,
  _::DeviceWithRepr,
  silent::Bool,
)::Tuple{SharedArrayData{T},Nothing} where {T}

  if !silent
    println("Using max $(nworkers()) process(es) + 1 master")
  end
  return (
    SharedArrayData{T}(
      SharedArray{T}(arraysize),
      SharedArray{T}(arraysize),
      SharedArray{T}(arraysize),
      scalar,
      arraysize,
    ),
    nothing,
  )
end

function init_arrays!(data::SharedArrayData{T}, _, init::Tuple{T,T,T}) where {T}

  @sync @distributed for i = 1:data.size
    @inbounds data.a[i] = init[1]
    @inbounds data.b[i] = init[2]
    @inbounds data.c[i] = init[3]
  end
end

function copy!(data::SharedArrayData{T}, _) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.c[i] = data.a[i]
  end
end

function mul!(data::SharedArrayData{T}, _) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.b[i] = data.scalar * data.c[i]
  end
end

function add!(data::SharedArrayData{T}, _) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.c[i] = data.a[i] + data.b[i]
  end
end

function triad!(data::SharedArrayData{T}, _) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.a[i] = data.b[i] + (data.scalar * data.c[i])
  end
end

function nstream!(data::SharedArrayData{T}, _) where {T}
  @sync @distributed for i = 1:data.size
    @inbounds data.a[i] += data.b[i] + data.scalar * data.c[i]
  end
end

function dot(data::SharedArrayData{T}, _) where {T}
  return @distributed (+) for i = 1:data.size
    @inbounds data.a[i] * data.b[i]
  end
end

function read_data(data::SharedArrayData{T}, _)::VectorData{T} where {T}
  return VectorData{T}(data.a, data.b, data.c, data.scalar, data.size)
end

main()