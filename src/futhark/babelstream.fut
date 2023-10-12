module type kernels = {
  type t
  val copy [n] : [n]t -> *[n]t
  val mul [n] : t -> [n]t -> [n]t
  val add [n] : [n]t -> [n]t -> [n]t
  val triad [n] : t -> [n]t -> [n]t -> [n]t
  val dot [n] : [n]t -> [n]t -> t
  -- Uniqueness allows nstream to mutate the 'a' array.
  val nstream [n] : t -> *[n]t -> [n]t -> [n]t -> [n]t
}

module kernels (P: real) : kernels with t = P.t = {
  type t = P.t
  def copy = copy
  def mul scalar c = map (P.*scalar) c
  def add = map2 (P.+)
  def triad scalar b c = map2 (P.+) b (map (P.* scalar) c)
  def dot a b = reduce (P.+) (P.i32 0) (map2 (P.*) a b)
  def nstream scalar a b c = map2 (P.+) a (map2 (P.+) b (map (P.*scalar) c))
}

module f32_kernels = kernels f32
def f32_start_scalar : f32 = 0.4
entry f32_copy = f32_kernels.copy
entry f32_mul = f32_kernels.mul f32_start_scalar
entry f32_add = f32_kernels.add
entry f32_triad = f32_kernels.triad f32_start_scalar
entry f32_nstream = f32_kernels.nstream f32_start_scalar
entry f32_dot = f32_kernels.dot

module f64_kernels = kernels f64
def f64_start_scalar : f64 = 0.4
entry f64_copy = f64_kernels.copy
entry f64_mul = f64_kernels.mul f64_start_scalar
entry f64_add = f64_kernels.add
entry f64_triad = f64_kernels.triad f64_start_scalar
entry f64_nstream = f64_kernels.nstream f64_start_scalar
entry f64_dot = f64_kernels.dot

-- ==
-- entry: f32_copy f32_mul
-- random input { [33554432]f32 }

-- ==
-- entry: f32_add f32_dot f32_triad
-- random input { [33554432]f32 [33554432]f32 }

-- ==
-- entry: f32_nstream
-- random input { [33554432]f32 [33554432]f32 [33554432]f32 }

-- ==
-- entry: f64_copy f64_mul
-- random input { [33554432]f64 }

-- ==
-- entry: f64_add f64_dot f64_triad
-- random input { [33554432]f64 [33554432]f64 }

-- ==
-- entry: f64_nstream
-- random input { [33554432]f64 [33554432]f64 [33554432]f64 }
