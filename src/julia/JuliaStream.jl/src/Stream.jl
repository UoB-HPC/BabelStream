using ArgParse
using Parameters
using Printf
using Base: Float64, Int

include("StreamData.jl")

const VectorData = StreamData{T,Vector{T}} where {T}

const DeviceWithRepr = Tuple{Any,String,Any}

struct Timings
  copy::Vector{Float64}
  mul::Vector{Float64}
  add::Vector{Float64}
  triad::Vector{Float64}
  dot::Vector{Float64}
  Timings(n) = new(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))
end

@enum Benchmark All Triad Nstream


function run_init_arrays!(data::StreamData{T,C}, context, init::Tuple{T,T,T})::Float64 where {T,C}
  return @elapsed init_arrays!(data, context, init)
end

function run_read_data(data::StreamData{T,C}, context)::Tuple{Float64,VectorData{T}} where {T,C}
  elapsed = @elapsed begin
    result = read_data(data, context)
  end
  return (elapsed, result)
end

function run_all!(data::StreamData{T,C}, context, times::Int)::Tuple{Timings,T} where {T,C}
  timings = Timings(times)
  lastSum::T = 0
  for i = 1:times
    @inbounds timings.copy[i] = @elapsed copy!(data, context)
    @inbounds timings.mul[i] = @elapsed mul!(data, context)
    @inbounds timings.add[i] = @elapsed add!(data, context)
    @inbounds timings.triad[i] = @elapsed triad!(data, context)
    @inbounds timings.dot[i] = @elapsed lastSum = dot(data, context)
  end
  return (timings, lastSum)
end

function run_triad!(data::StreamData{T,C}, context, times::Int)::Float64 where {T,C}
  return @elapsed for _ = 1:times
    triad!(data, context)
  end
end

function run_nstream!(data::StreamData{T,C}, context, times::Int)::Vector{Float64} where {T,C}
  timings::Vector{Float64} = zeros(times)
  for i = 1:times
    @inbounds timings[i] = @elapsed nstream!(data, context)
  end
  return timings
end

function check_solutions(
  data::StreamData{T,C},
  times::Int,
  init::Tuple{T,T,T},
  benchmark::Benchmark,
  dot::Union{T,Nothing},
) where {T,C}
  (gold_a, gold_b, gold_c) = init
  for _ = 1:times
    if benchmark == All
      gold_c = gold_a
      gold_b = data.scalar * gold_c
      gold_c = gold_a + gold_b
      gold_a = gold_b + data.scalar * gold_c
    elseif benchmark == Triad
      gold_a = gold_b + data.scalar * gold_c
    elseif benchmark == Nstream
      gold_a += gold_b + data.scalar * gold_c
    else
      error("Unknown benchmark", benchmark)
    end
  end

  tolerance = eps(T) * 100
  function validate_xs(name::String, xs::AbstractArray{T}, from::T)
    error = (map(x -> abs(x - from), xs) |> sum) / length(xs)
    failed = error > tolerance
    if failed
      println("Validation failed on $name. Average error $error")
    end
    !failed
  end
  a_valid = validate_xs("a", data.a, gold_a)
  b_valid = validate_xs("b", data.b, gold_b)
  c_valid = validate_xs("c", data.c, gold_c)
  dot_valid =
    dot !== nothing ?
    begin
      gold_sum = gold_a * gold_b * data.size
      error = abs((dot - gold_sum) / gold_sum)
      failed = error > 1.0e-8
      if failed
        println("Validation failed on sum. Error $error \nSum was $dot but should be $gold_sum")
      end
      !failed
    end : true

  a_valid && b_valid && c_valid && dot_valid
end

@with_kw mutable struct Config
  list::Bool = false
  device::Int = 1
  numtimes::Int = 100
  arraysize::Int = 33554432
  float::Bool = false
  triad_only::Bool = false
  nstream_only::Bool = false
  csv::Bool = false
  mibibytes::Bool = false
end

function parse_options(given::Config)
  s = ArgParseSettings()
  @add_arg_table s begin
    "--list"
    help = "List available devices"
    action = :store_true
    "--device", "-d"
    help = "Select device at DEVICE, NOTE: Julia is 1-indexed"
    arg_type = Int
    default = given.device
    "--numtimes", "-n"
    help = "Run the test NUMTIMES times (NUM >= 2)"
    arg_type = Int
    default = given.numtimes
    "--arraysize", "-s"
    help = "Use ARRAYSIZE elements in the array"
    arg_type = Int
    default = given.arraysize
    "--float"
    help = "Use floats (rather than doubles)"
    action = :store_true
    "--triad_only"
    help = "Only run triad"
    action = :store_true
    "--nstream_only"
    help = "Only run nstream"
    action = :store_true
    "--csv"
    help = "Output as csv table"
    action = :store_true
    "--mibibytes"
    help = "Use MiB=2^20 for bandwidth calculation (default MB=10^6)"
    action = :store_true
  end
  args = parse_args(s)
  # surely there's a better way than doing this:
  for (arg, val) in args
    setproperty!(given, Symbol(arg), val)
  end
end

const DefaultInit = (0.1, 0.2, 0.0)
const DefaultScalar = 0.4
const Version = "5.0"

function main()

  config::Config = Config()
  parse_options(config)

  if config.list
    for (i, (_, repr, impl)) in enumerate(devices())
      println("[$i] ($impl) $repr")
    end
    exit(0)
  end

  ds = devices()
  # TODO implement substring device match
  if config.device < 1 || config.device > length(ds)
    error("Device $(config.device) out of range (1..$(length(ds))), NOTE: Julia is 1-indexed")
  else
    device = ds[config.device]
  end

  type = config.float ? Float32 : Float64

  if config.nstream_only && !config.triad_only
    benchmark = Nstream
  elseif !config.nstream_only && config.triad_only
    benchmark = Triad
  elseif !config.nstream_only && !config.triad_only
    benchmark = All
  elseif config.nstream_only && config.triad_only
    error("Both triad and nstream are enabled, pick one or omit both to run all benchmarks")
  else
    error("Invalid config: $(repr(config))")
  end

  array_bytes = config.arraysize * sizeof(type)
  total_bytes = array_bytes * 3
  (mega_scale, mega_suffix, giga_scale, giga_suffix) =
    !config.mibibytes ? (1.0e-6, "MB", 1.0e-9, "GB") : (2^-20, "MiB", 2^-30, "GiB")

  if !config.csv
    println("""BabelStream
               Version: $Version
               Implementation: Julia; $(PROGRAM_FILE)""")
    println("Running kernels $(config.numtimes) times")
    if benchmark == Triad
      println("Number of elements: $(config.arraysize)")
    end
    println("Precision: $(config.float ?  "float" :   "double")")
    r1 = n -> round(n; digits = 1)
    println(
      "Array size: $(r1(mega_scale * array_bytes)) $mega_suffix(=$(r1(giga_scale * array_bytes)) $giga_suffix)",
    )
    println(
      "Total size: $(r1(mega_scale * total_bytes)) $mega_suffix(=$(r1(giga_scale * total_bytes)) $giga_suffix)",
    )
  end

  function mk_row(xs::Vector{Float64}, name::String, total_bytes::Int)
    tail = Iterators.rest(xs)
    min = Base.minimum(tail)
    max = Base.maximum(tail)
    avg = Base.sum(tail) / Base.length(tail)
    mbps = mega_scale * total_bytes / min
    if config.csv
      return [
        ("function", name),
        ("num_times", config.numtimes),
        ("n_elements", config.arraysize),
        ("sizeof", total_bytes),
        ("max_m$(  config.mibibytes ? "i" : "")bytes_per_sec", mbps),
        ("min_runtime", min),
        ("max_runtime", max),
        ("avg_runtime", avg),
      ]
    else
      return [
        ("Function", name),
        ("M$(config.mibibytes ? "i" : "")Bytes/sec", round(mbps; digits = 3)),
        ("Min (sec)", round(min; digits = 5)),
        ("Max", round(max; digits = 5)),
        ("Average", round(avg; digits = 5)),
      ]
    end
  end

  function tabulate(rows::Vector{Tuple{String,Any}}...)
    header = Base.first(rows)
    padding = config.csv ? 0 : 12
    sep = config.csv ? "," : ""
    map(x -> rpad(x[1], padding), header) |> x -> join(x, sep) |> println
    for row in rows
      map(x -> rpad(x[2], padding), row) |> x -> join(x, sep) |> println
    end
  end

  function show_init(init::Float64, read::Float64)
    setup = [("Init", init, 3 * array_bytes), ("Read", read, 3 * array_bytes)]
    if config.csv
      tabulate(
        map(
          x -> [
            ("phase", x[1]),
            ("n_elements", config.arraysize),
            ("sizeof", x[3]),
            ("max_m$(config.mibibytes ? "i" : "")bytes_per_sec", mega_scale * total_bytes / x[2]),
            ("runtime", x[2]),
          ],
          setup,
        )...,
      )
    else
      for (name, elapsed, total_bytes) in setup
        println(
          "$name: $(round(elapsed; digits=5)) s (=$(round(( mega_scale * total_bytes) / elapsed; digits = 5)) M$(config.mibibytes ? "i" : "")Bytes/sec)",
        )
      end
    end
  end

  init::Tuple{type,type,type} = DefaultInit
  scalar::type = DefaultScalar

  GC.enable(false)

  (data, context) = make_stream(config.arraysize, scalar, device, config.csv)
  tInit = run_init_arrays!(data, context, init)
  if benchmark == All
    (timings, sum) = run_all!(data, context, config.numtimes)
    (tRead, result) = run_read_data(data, context)
    show_init(tInit, tRead)
    valid = check_solutions(result, config.numtimes, init, benchmark, sum)
    tabulate(
      mk_row(timings.copy, "Copy", 2 * array_bytes),
      mk_row(timings.mul, "Mul", 2 * array_bytes),
      mk_row(timings.add, "Add", 3 * array_bytes),
      mk_row(timings.triad, "Triad", 3 * array_bytes),
      mk_row(timings.dot, "Dot", 2 * array_bytes),
    )
  elseif benchmark == Nstream
    timings = run_nstream!(data, context, config.numtimes)
    (tRead, result) = run_read_data(data, context)
    show_init(tInit, tRead)
    valid = check_solutions(result, config.numtimes, init, benchmark, nothing)
    tabulate(mk_row(timings, "Nstream", 4 * array_bytes))
  elseif benchmark == Triad
    elapsed = run_triad!(data, context, config.numtimes)
    (tRead, result) = run_read_data(data, context)
    show_init(tInit, tRead)
    valid = check_solutions(result, config.numtimes, init, benchmark, nothing)
    total_bytes = 3 * array_bytes * config.numtimes
    bandwidth = mega_scale * (total_bytes / elapsed)
    println("Runtime (seconds): $(round(elapsed; digits=5))")
    println("Bandwidth ($giga_suffix/s): $(round(bandwidth; digits=3)) ")
  else
    error("Bad benchmark $(benchmark)")
  end
  GC.enable(true)

  if !valid
    exit(1)
  end

end
