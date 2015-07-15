
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define DATATYPE double

constant DATATYPE scalar = 3.0;

kernel void copy(global const DATATYPE * restrict a, global DATATYPE * restrict c)
{
	const size_t i = get_global_id(0);
	c[i] = a[i];
}

kernel void mul(global DATATYPE * restrict b, global const DATATYPE * restrict c)
{
	const size_t i = get_global_id(0);
	b[i] = scalar * c[i];
}

kernel void add(global const DATATYPE * restrict a, global const DATATYPE * restrict b, global DATATYPE * restrict c)
{
	const size_t i = get_global_id(0);
	c[i] = a[i] + b[i];
}

kernel void triad(global DATATYPE * restrict a, global const DATATYPE * restrict b, global const DATATYPE * restrict c)
{
	const size_t i = get_global_id(0);
	a[i] = b[i] + scalar * c[i];
}
