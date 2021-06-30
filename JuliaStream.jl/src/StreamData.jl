struct StreamData{T,C<:AbstractArray{T}}
  a::C
  b::C
  c::C
  scalar::T
  size::Int
end
