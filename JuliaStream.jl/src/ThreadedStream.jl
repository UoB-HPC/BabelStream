include("Stream.jl")

function devices()::Vector{DeviceWithRepr}
  return [(undef, "$(Sys.cpu_info()[1].model) ($(Threads.nthreads())T)", "Threaded")]
end

function make_stream(
  arraysize::Int,
  scalar::T,
  _::DeviceWithRepr,
  silent::Bool,
)::Tuple{VectorData{T},Nothing} where {T}
  if !silent
    println("Using max $(Threads.nthreads()) threads")
  end
  return (
    VectorData{T}(
      Vector{T}(undef, arraysize),
      Vector{T}(undef, arraysize),
      Vector{T}(undef, arraysize),
      scalar,
      arraysize,
    ),
    nothing
  )
end

function init_arrays!(data::VectorData{T}, _, init::Tuple{T,T,T}) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.a[i] = init[1]
    @inbounds data.b[i] = init[2]
    @inbounds data.c[i] = init[3]
  end
end

function copy!(data::VectorData{T}, _) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.c[i] = data.a[i]
  end
end

function mul!(data::VectorData{T}, _) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.b[i] = data.scalar * data.c[i]
  end
end

function add!(data::VectorData{T}, _) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.c[i] = data.a[i] + data.b[i]
  end
end

function triad!(data::VectorData{T}, _) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.a[i] = data.b[i] + (data.scalar * data.c[i])
  end
end

function nstream!(data::VectorData{T}, _) where {T}
  Threads.@threads for i = 1:data.size
    @inbounds data.a[i] += data.b[i] + data.scalar * data.c[i]
  end
end

# Threads.@threads/Threads.@spawn doesn't support OpenMP's firstprivate, etc
function static_par_ranged(f::Function, range::Int, n::Int)
  stride = range ÷ n
  rem = range % n
  strides = map(0:n) do i
    width = stride + (i < rem ? 1 : 0)
    offset = i < rem ? (stride + 1) * i : ((stride + 1) * rem) + (stride * (i - rem))
    (offset, width)
  end
  ccall(:jl_enter_threaded_region, Cvoid, ())
  try
    foreach(wait, map(1:n) do group
      (offset, size) = strides[group]
      task = Task(() -> f(group, offset+1, offset+size))
      task.sticky = true
      ccall(:jl_set_task_tid, Cvoid, (Any, Cint), task, group-1) # ccall, so 0-based for group
      schedule(task)
    end)
  finally
    ccall(:jl_exit_threaded_region, Cvoid, ())
  end
end

function dot(data::VectorData{T}, _) where {T}
  partial = Vector{T}(undef,  Threads.nthreads())
  static_par_ranged(data.size, Threads.nthreads()) do group, startidx, endidx
    acc = zero(T)
    @simd for i = startidx:endidx
      @inbounds acc += data.a[i] * data.b[i]
    end
    @inbounds partial[group] = acc
  end
  return sum(partial)
  # This doesn't do well on aarch64 because of the excessive Threads.threadid() ccall 
  # and inhibited vectorisation from the lack of @simd 
  # partial = zeros(T, Threads.nthreads())
  # Threads.@threads for i = 1:data.size
  #   @inbounds partial[Threads.threadid()] += (data.a[i] * data.b[i])
  # end
  # return sum(partial)
end

function read_data(data::VectorData{T}, _)::VectorData{T} where {T}
  return data
end

main()